"""The optical music recognition engine.

Parses PNG music score images.

The engine holds a TensorFlow graph containing all structural information and
classifier predictions from an image.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import six
import tensorflow as tf

from magenta.models.omr import conversions
from magenta.models.omr import image
from magenta.models.omr import page_processors
from magenta.models.omr import score_processors
from magenta.models.omr import structure as structure_module
from magenta.models.omr.glyphs import saved_classifier_fn
from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.staves import base as staves_base
from magenta.models.omr.structure import beams
from magenta.models.omr.structure import components
from magenta.models.omr.structure import verticals


# TODO(ringwalt): Get OMR running on GPU. It seems to create too many small
# ops/allocations and brings down my machine.
CONFIG = tf.ConfigProto(device_count={'GPU': 0})


class OMREngine(object):
  """The OMR engine.

  The engine reads one music score page image at a time, and extracts musical
  elements from it.

  The glyph classifier can be chosen by a custom glyph_classifier_fn.

  Attributes:
    graph: The TensorFlow graph used by OMR.
    png_path: A scalar tensor representing the PNG image filename to load. It
      should be used as a key in the `feed_dict` to `get_page()`. The image is
      expected to contain an entire music score page.
    image: A 2D uint8 tensor representing the music score page image. The
      background is white (255) and the foreground is black (0). It may be fed
      in to `get_page()` instead of specifying a PNG filename with `png_path`.
    structure: The `Structure` holds the tensors that represent structural
      information in the music score.
    glyph_classifier: An instance of `BaseGlyphClassifier`, which holds the
      tensor representing the detected glyphs for the entire page. Defaults to
      nearest-neighbor classification using the pre-packaged labeled clusters.
  """

  def __init__(self, glyph_classifier_fn=None):
    """Creates the engine and TF graph for running OMR.

    Args:
      glyph_classifier_fn: Callable that loads the glyph classifier into the
        graph. Accepts a `Structure` as the single argument, and returns an
        instance of `BaseGlyphClassifier`. The function typically loads a TF
        saved model or other external data, and wraps the classification in a
        concrete glyph classifier subclass. If the classifier uses a
        `StafflineExtractor` for classification, it must set the
        `staffline_extractor` attribute of the `Structure`. Otherwise, glyph x
        coordinates will not be scaled back to image coordinates.
    """
    glyph_classifier_fn = (
        glyph_classifier_fn or saved_classifier_fn.build_classifier_fn())
    self.graph = tf.Graph()
    self.session = tf.Session(graph=self.graph)
    with self.graph.as_default():
      with self.session.as_default():
        with tf.name_scope('OMREngine'):
          self.png_path = tf.placeholder(tf.string, name='png_path', shape=())
          self.image = image.decode_music_score_png(
              tf.read_file(self.png_path, name='page_image'))
          self.structure = structure_module.create_structure(self.image)
        # Loading saved models happens outside of the name scope, because scopes
        # can rename tensors from the model and cause dangling references.
        # TODO(ringwalt): TF should be able to load models gracefully within a
        # name scope.
        self.glyph_classifier = glyph_classifier_fn(self.structure)

  def run(self, input_pngs, output_notesequence=False):
    """Converts input PNGs into a `Score` message.

    Args:
      input_pngs: A list of PNG filenames to process.
      output_notesequence: Whether to return a NoteSequence, as opposed to a
          Score containing Pages with Glyphs.

    Returns:
      A NoteSequence message, or a Score message holding Pages for each input
          image (with their detected Glyphs).
    """
    if isinstance(input_pngs, six.string_types):
      input_pngs = [input_pngs]
    score = musicscore_pb2.Score()
    with tf.Session(graph=self.graph, config=CONFIG):
      score.page.extend(
          self._get_page(feed_dict={
              self.png_path: png
          }) for png in input_pngs)
    score = score_processors.process(score)
    return (conversions.score_to_notesequence(score)
            if output_notesequence else score)

  def process_image(self, image_arr, process_structure=True):
    """Processes a uint8 image array.

    Args:
      image_arr: A 2D (H, W) uint8 NumPy array. Must have a white (255)
        background and black (0) foreground.
      process_structure: Whether to add structural information to the page.

    Returns:
      A `Page` message constructed from the contents of the image.
    """
    with tf.Session(graph=self.graph, config=CONFIG):
      return self._get_page(
          feed_dict={self.image: image_arr},
          process_structure=process_structure)

  def _get_page(self, feed_dict=None, process_structure=True):
    """Returns the Page holding Glyphs for the page.

    Args:
      feed_dict: The feed dict to use for the TensorFlow graph. The image must
        be fed in.
      process_structure: If True, run the page_processors, which add staff
        locations and other structural information. If False, return a Page
        containing only Glyphs for each staff.

    Returns:
      A Page message holding Staff protos which have location information and
          detected Glyphs.
    """
    # If structure is given, output all structural information in addition to
    # the Page message.
    structure_data = (
        _nested_ndarrays_to_tensors(self.structure.data)
        if self.structure else [])
    structure_data, glyphs = tf.get_default_session().run(
        [structure_data,
         self.glyph_classifier.get_detected_glyphs()],
        feed_dict=feed_dict)
    computed_staves, computed_beams, computed_verticals, computed_components = (
        structure_data)

    # Construct and return a computed Structure.
    computed_structure = structure_module.Structure(
        staves_base.ComputedStaves(*computed_staves),
        beams.ComputedBeams(*computed_beams),
        verticals.ComputedVerticals(*computed_verticals),
        components.ComputedComponents(*computed_components))

    # The Page without staff location information.
    labels_page = self.glyph_classifier.glyph_predictions_to_page(glyphs)

    # Process the Page using the computed structure.
    if process_structure:
      processed_page = page_processors.process(
          labels_page, computed_structure,
          self.glyph_classifier.staffline_extractor)
    else:
      processed_page = labels_page
    return processed_page


def _nested_ndarrays_to_tensors(data):
  """Converts possibly nested lists of np.ndarrays and Tensors to Tensors.

  This is necessary in case some data in the Structure is already computed. We
  just pass everything to tf.Session.run, and some data may be a tf.constant
  which is just spit back out.

  Args:
    data: An np.ndarray, Tensor, or list recursively containing the same types.

  Returns:
    data with all np.ndarrays converted to Tensors.

  Raises:
    ValueError: If unexpected data was given.
  """
  if isinstance(data, list):
    return [_nested_ndarrays_to_tensors(element) for element in data]
  elif isinstance(data, np.ndarray):
    return tf.constant(data)
  elif isinstance(data, tf.Tensor):
    return data
  else:
    raise ValueError('Unexpected data: %s' % data)
