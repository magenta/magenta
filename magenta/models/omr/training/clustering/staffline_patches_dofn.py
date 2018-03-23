"""Extracts non-empty patches of extracted stafflines.

Extracts vertical slices of the image where glyphs are expected
(see `staffline_extractor.py`), and takes horizontal windows of the slice which
will be clustered. Some patches will have a glyph roughly in their center, and
the corresponding cluster centroids will be labeled as such.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path
import random

# internal imports
import apache_beam as beam
from six import moves
import tensorflow as tf

from magenta.models.omr import image
from magenta.models.omr import staves
from magenta.models.omr.staves import removal
from magenta.models.omr.staves import staffline_extractor
from magenta.models.omr.util import patches as util_patches


def pipeline_graph(png_path, staffline_height, patch_width, num_stafflines):
  """Constructs the graph for the staffline patches pipeline.

  Args:
    png_path: Path to the input png. String scalar tensor.
    staffline_height: Height of a staffline. int.
    patch_width: Width of a patch. int.
    num_stafflines: Number of stafflines to extract around each staff. int.

  Returns:
    A tensor representing the staffline patches. float32 with shape
        (num_patches, staffline_height, patch_width).
  """
  image_t = image.decode_music_score_png(tf.read_file(png_path))
  staff_detector = staves.StaffDetector(image_t)
  staff_remover = removal.StaffRemover(staff_detector)
  stafflines = tf.identity(
      staffline_extractor.StafflineExtractor(
          staff_remover.remove_staves,
          staff_detector,
          target_height=staffline_height,
          num_sections=num_stafflines).extract_staves(),
      name='stafflines')
  return _extract_patches(stafflines, patch_width)


def _extract_patches(stafflines, patch_width, min_num_dark_pixels=10):
  patches = util_patches.patches_1d(stafflines, patch_width)
  # Limit to patches that have min_num_dark_pixels.
  num_dark_pixels = tf.reduce_sum(
      tf.where(
          tf.less(patches, 0.5),
          tf.ones_like(patches, dtype=tf.int32),
          tf.zeros_like(patches, dtype=tf.int32)),
      axis=[-2, -1])
  return tf.boolean_mask(patches,
                         tf.greater_equal(num_dark_pixels, min_num_dark_pixels))


class StafflinePatchesDoFn(beam.DoFn):
  """Runs the staffline patches graph."""

  def __init__(self, patch_height, patch_width, num_stafflines, timeout_ms,
               max_patches_per_page):
    self.patch_height = patch_height
    self.patch_width = patch_width
    self.num_stafflines = num_stafflines
    self.timeout_ms = timeout_ms
    self.max_patches_per_page = max_patches_per_page

  def start_bundle(self):
    self.session = tf.Session()
    with self.session:
      # Construct the graph.
      self.png_path = tf.placeholder(tf.string, shape=(), name='png_path')
      self.patches = pipeline_graph(self.png_path, self.patch_height,
                                    self.patch_width, self.num_stafflines)

  def process(self, png_path):
    basename = os.path.basename(png_path)
    run_options = tf.RunOptions(timeout_in_ms=self.timeout_ms)
    try:
      patches = self.session.run(
          self.patches,
          feed_dict={self.png_path: png_path},
          options=run_options)
    # pylint: disable=broad-except
    except Exception:
      logging.exception('Skipping failed music score (%s)', png_path)
      return
    # Subsample patches.
    if 0 < self.max_patches_per_page < len(patches):
      patch_inds = random.sample(
          moves.range(len(patches)), self.max_patches_per_page)
      patches = patches[patch_inds]
    # Serialize each patch as an Example.
    for i, patch in enumerate(patches):
      patch_name = (basename + '#' + str(i)).encode('utf-8')
      example = tf.train.Example()
      example.features.feature['name'].bytes_list.value.append(patch_name)
      example.features.feature['features'].float_list.value.extend(
          patch.ravel())
      example.features.feature['height'].int64_list.value.append(patch.shape[0])
      example.features.feature['width'].int64_list.value.append(patch.shape[1])
      yield example

  def finish_bundle(self):
    self.session.close()
    del self.session
