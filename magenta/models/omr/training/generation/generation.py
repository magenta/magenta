"""VexFlow labeled data generation.

Wraps the node.js generator, which generates a random measure of music as SVG,
and the ground truth glyphs present in the image as a `Page` message.

Each invocation generates a batch of images. There is a tradeoff between the
startup time of node.js for each invocation, and keeping the output size small
enough to pipe into Python.

The final outputs are positive and negative example patches. Positive examples
are centered on an outputted glyph, and have that glyph's type. Negative
examples are at least a few pixels away from any glyph, and have type NONE.
Since negative examples could be a few pixels away from a glyph, we get negative
examples that overlap with partial glyph(s), but are centered too far away from
a glyph to be considered a positive example. Currently, every single glyph
results in a single positive example, and negative examples are randomly
sampled.

All glyphs are emitted to RecordIO, where they are outputted in a single
collection for training. We currently do not store the entire generated image
anywhere. This could be added later in order to try other classification
approaches.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import random
import subprocess
import sys

# internal imports
import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from magenta.models.omr import engine
from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.staves import staffline_distance
from magenta.models.omr.staves import staffline_extractor

# Every image is expected to contain at least 3 glyphs.
POSITIVE_EXAMPLES_PER_IMAGE = 3


def _normalize_path(filename):
  """Normalizes a relative path to a command to spawn.

  Args:
    filename: String; relative or absolute path.

  Returns:
    The normalized path. This is necessary because in our use case,
    vexflow_generator_pipeline will live in a different directory from
    vexflow_generator, and there are symlinks to both directories in the same
    parent directory. Without normalization, `..` would reference the parent of
    the actual directory that was symlinked. With normalization, it references
    the directory that contains the symlink to the working directory.
  """
  if filename.startswith('/'):
    return filename
  else:
    return os.path.normpath(
        os.path.join(os.path.dirname(sys.argv[0]), filename))


class PageGenerationDoFn(beam.DoFn):
  """Generates the PNG images and ground truth for each batch.

  Takes in a batch number, and outputs a tuple of PNG contents (bytes) and the
  labeled staff (Staff message).
  """

  def __init__(self, num_pages_per_batch, vexflow_generator_command,
               svg_to_png_command):
    self.num_pages_per_batch = num_pages_per_batch
    self.vexflow_generator_command = vexflow_generator_command
    self.svg_to_png_command = svg_to_png_command

  def process(self, batch_num):
    for page in self.get_pages_for_batch(batch_num, self.num_pages_per_batch):
      staff = musicscore_pb2.Staff()
      text_format.Parse(page['page'], staff)
      yield self._svg_to_png(page['svg']), staff

  def get_pages_for_batch(self, batch_num, num_pages_per_batch):
    """Generates the music score pages in a single batch.

    The generator takes in a seed for the RNG for each page, and outputs all
    pages at once. The seeds for all batches are consecutive for determinism,
    starting from 0, but each seed to the Mersenne Twister RNG should result in
    completely different output.

    Args:
      batch_num: The index of the batch to output.
      num_pages_per_batch: The number of pages to generate in each batch.

    Returns:
      A list of dicts holding `svg` (XML text) and `page` (text-format
          `tensorflow.magenta.omr.Staff` proto).
    """
    return self.get_pages(
        range(batch_num * num_pages_per_batch,
              (batch_num + 1) * num_pages_per_batch))

  def get_pages(self, seeds):
    vexflow_generator_command = list(self.vexflow_generator_command)
    # If vexflow_generator_command is relative, it is relative to the pipeline
    # binary.
    vexflow_generator_command[0] = _normalize_path(vexflow_generator_command[0])

    seeds = ','.join(map(str, seeds))
    return json.loads(
        subprocess.check_output(
            vexflow_generator_command + ['--random_seeds=' + seeds]))

  def _svg_to_png(self, svg):
    svg_to_png_command = list(self.svg_to_png_command)
    svg_to_png_command[0] = _normalize_path(svg_to_png_command[0])
    popen = subprocess.Popen(
        svg_to_png_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = popen.communicate(input=svg)
    if popen.returncode != 0:
      raise ValueError('convert failed with status %d\nstderr:\n%s' %
                       (popen.returncode, stderr))
    return stdout


class PatchExampleDoFn(beam.DoFn):
  """Extracts labeled patches from generated VexFlow music scores."""

  def __init__(self,
               negative_example_distance,
               patch_width,
               negative_to_positive_example_ratio,
               noise_fn=lambda x: x):
    self.negative_example_distance = negative_example_distance
    self.patch_width = patch_width
    self.negative_to_positive_example_ratio = negative_to_positive_example_ratio
    self.noise_fn = noise_fn
    self.patch_counter = Metrics.counter(self.__class__, 'num_patches')

  def start_bundle(self):
    # TODO(ringwalt): Expose a cleaner way to set this value.
    # The image is too small for the default min staffline distance score.
    # pylint: disable=protected-access
    staffline_distance._MIN_STAFFLINE_DISTANCE_SCORE = 100

    self.omr = engine.OMREngine()

  def process(self, item):
    png_contents, staff_message = item
    with tf.Session(graph=self.omr.graph) as sess:
      # Load the image, then feed it in to apply noise.
      # Randomly rotate the image and apply noise, then dump it back out as a
      # PNG.
      # TODO(ringwalt): Expose a way to pass in the image contents to the main
      # OMR TF graph.
      img = tf.to_float(tf.image.decode_png(png_contents))
      # Collapse the RGB channels, if any. No-op for a monochrome PNG.
      img = tf.reduce_mean(img[:, :, :3], axis=2)[:, :, None]
      # Fix the stafflines being #999.
      img = tf.clip_by_value(img * 2. - 255., 0., 255.)
      img = self.noise_fn(img)
      # Get a 2D uint8 image array for OMR.
      noisy_image = sess.run(
          tf.cast(tf.clip_by_value(img, 0, 255)[:, :, 0], tf.uint8))
      # Run OMR staffline extraction and staffline distance estimation. The
      # stafflines are used to get patches from the generated image.
      stafflines, image_staffline_distance = sess.run(
          [
              self.omr.glyph_classifier.staffline_extractor.extract_staves(),
              self.omr.structure.staff_detector.staffline_distance[0]
          ],
          feed_dict={
              self.omr.image: noisy_image
          })
    if stafflines.shape[0] != 1:
      raise ValueError('Image should have one detected staff, got shape: ' +
                       str(stafflines.shape))
    positive_example_count = 0
    negative_example_whitelist = np.ones(
        (stafflines.shape[staffline_extractor.Axes.POSITION],
         stafflines.shape[staffline_extractor.Axes.X]), np.bool)
    # Blacklist xs where the patch would overlap with either end.
    negative_example_overlap_from_end = max(self.negative_example_distance,
                                            self.patch_width // 2)
    negative_example_whitelist[:, :negative_example_overlap_from_end] = False
    negative_example_whitelist[:,
                               -negative_example_overlap_from_end - 1:] = False
    all_positive_examples = []
    for glyph in staff_message.glyph:
      staffline = staffline_extractor.get_staffline(glyph.y_position,
                                                    stafflines[0])
      glyph_x = int(
          round(glyph.x *
                self.omr.glyph_classifier.staffline_extractor.target_height /
                (image_staffline_distance * self.omr.glyph_classifier.
                 staffline_extractor.staffline_distance_multiple)))
      example = self._create_example(staffline, glyph_x, glyph.type)
      if example:
        staffline_index = staffline_extractor.y_position_to_index(
            glyph.y_position,
            stafflines.shape[staffline_extractor.Axes.POSITION])
        # Blacklist the area adjacent to the glyph, even if it is not selected
        # as a positive example below.
        negative_example_whitelist[
            staffline_index, glyph_x - self.negative_example_distance + 1:
            glyph_x + self.negative_example_distance] = False
        all_positive_examples.append(example)
        positive_example_count += 1

    for example in random.sample(all_positive_examples,
                                 POSITIVE_EXAMPLES_PER_IMAGE):
      yield example
      self.patch_counter.inc()

    negative_example_staffline, negative_example_x = np.where(
        negative_example_whitelist)
    negative_example_inds = np.random.choice(
        len(negative_example_staffline),
        int(positive_example_count * self.negative_to_positive_example_ratio))
    negative_example_staffline = negative_example_staffline[
        negative_example_inds]
    negative_example_x = negative_example_x[negative_example_inds]
    for staffline, x in zip(negative_example_staffline, negative_example_x):
      example = self._create_example(stafflines[0, staffline], x,
                                     musicscore_pb2.Glyph.NONE)
      assert example, 'Negative example xs should always be in range'
      yield example
      self.patch_counter.inc()

  def _create_example(self, staffline, x, label):
    start_x = x - self.patch_width // 2
    limit_x = x + self.patch_width // 2 + 1
    assert limit_x - start_x == self.patch_width
    # x is the last axis of staffline
    if 0 <= start_x <= limit_x < staffline.shape[-1]:
      patch = staffline[:, start_x:limit_x]
      example = tf.train.Example()
      example.features.feature['patch'].float_list.value.extend(patch.ravel())
      example.features.feature['label'].int64_list.value.append(label)
      example.features.feature['height'].int64_list.value.append(patch.shape[0])
      example.features.feature['width'].int64_list.value.append(patch.shape[1])
      return example
    else:
      return None
