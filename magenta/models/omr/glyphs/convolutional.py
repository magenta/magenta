"""Base glyph classifier model."""
# TODO(ringwalt): Replace subclasses with a saved TF model. Hardcode the
# stafflines and predictions tensor names, so that we define the classifier
# separately. It can either be defined in the same graph before constructing the
# Convolutional1DGlyphClassifier, or loaded from a saved model after training
# externally.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.glyphs import base
from magenta.models.omr.protobuf import musicscore_pb2


class Convolutional1DGlyphClassifier(base.BaseGlyphClassifier):
  """The base 1D convolutional glyph classifier model."""

  def __init__(self, run_min_length=3):
    """Base classifier model.

    Args:
      run_min_length: Must have this many consecutive pixels with the same
          non-NONE predicted glyph to emit the glyph.
    """
    super(Convolutional1DGlyphClassifier, self).__init__()
    self.run_min_length = run_min_length

  @property
  @abc.abstractmethod
  def staffline_predictions(self):
    """The staffline predictions tensor.

    Convolutional1DGlyphClassifier wraps this output, which would be the output
    of a 1D convolutional model, and extracts individual glyphs to be added to
    the Page message.

    Shape (num_staves, num_stafflines, width).
    """
    pass

  def _build_detected_glyphs(self, predictions_arr):
    """Takes the convolutional output ndarray and builds the individual glyphs.

    At each staff and y position, looks for short runs of the same detected
    glyph, and then outputs a single glyph at the x coordinate of the center of
    the run.

    Args:
      predictions_arr: A NumPy array with the result of `staffline_predictions`.
          Shape (num_staves, num_stafflines, width).

    Returns:
      A 2D array of the glyph coordinates. Shape (num_glyphs, 4) with columns
          corresponding to base.GlyphsTensorColumns.
    """
    glyphs = []
    num_staves, num_stafflines, width = predictions_arr.shape
    for staff in range(num_staves):
      for staffline in range(num_stafflines):
        y_position = num_stafflines // 2 - staffline
        run_start = -1
        run_value = musicscore_pb2.Glyph.NONE
        for x in range(width + 1):
          if x < width:
            value = predictions_arr[staff, staffline, x]
          if x == width or value != run_value:
            if run_value > musicscore_pb2.Glyph.NONE:
              # Process the current run if it is at least run_min_length pixels.
              if x - run_start >= self.run_min_length:
                glyph_center_x = (run_start + x) // 2
                glyphs.append(
                    self._create_glyph_arr(staff, y_position, glyph_center_x,
                                           run_value))
            run_value = value
            run_start = x
    # Convert to a 2D array.
    glyphs = np.asarray(glyphs, np.int32)
    return np.reshape(glyphs, (-1, 4))

  def _create_glyph_arr(self, staff_index, y_position, x, type_value):
    glyph = np.empty(len(base.GlyphsTensorColumns), np.int32)
    glyph[base.GlyphsTensorColumns.STAFF_INDEX] = staff_index
    glyph[base.GlyphsTensorColumns.Y_POSITION] = y_position
    glyph[base.GlyphsTensorColumns.X] = x
    glyph[base.GlyphsTensorColumns.TYPE] = type_value
    return glyph

  def get_detected_glyphs(self):
    """Extracts the individual glyphs as a Tensor.

    This is run in the TensorFlow graph, so we have to wrap the Python glyph
    logic in a `py_func`.

    Returns:
      A Tensor of glyphs, with shape (num_glyphs, 4). The columns are indexed by
        `base.GlyphsTensorColumns`.
    """
    return tf.py_func(self._build_detected_glyphs, [self.staffline_predictions],
                      tf.int32)
