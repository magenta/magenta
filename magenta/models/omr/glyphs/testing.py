"""Testing utilities for glyph classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.glyphs import convolutional
from magenta.models.omr.protobuf import musicscore_pb2


# Sample glyph predictions.
# Shape (num_staves, num_stafflines, width).
PREDICTIONS = np.asarray(
    [[[1, 1, 1, 1, 2, 1],
      [3, 1, 5, 1, 1, 1],
      [1, 4, 1, 1, 1, 1]],
     [[1, 1, 3, 1, 1, 1],
      [1, 1, 5, 1, 1, 1],
      [1, 1, 1, 1, 3, 5]]])  # pyformat: disable
# Page corresponding to the glyphs in PREDICTIONS.
GLYPHS_PAGE = musicscore_pb2.Page(system=[
    musicscore_pb2.StaffSystem(staff=[
        musicscore_pb2.Staff(glyph=[
            musicscore_pb2.Glyph(x=0, y_position=0, type=3),
            musicscore_pb2.Glyph(x=1, y_position=-1, type=4),
            musicscore_pb2.Glyph(x=2, y_position=0, type=5),
            musicscore_pb2.Glyph(x=4, y_position=1, type=2),
        ]),
        musicscore_pb2.Staff(glyph=[
            musicscore_pb2.Glyph(x=2, y_position=1, type=3),
            musicscore_pb2.Glyph(x=2, y_position=0, type=5),
            musicscore_pb2.Glyph(x=4, y_position=-1, type=3),
            musicscore_pb2.Glyph(x=5, y_position=-1, type=5),
        ]),
    ]),
])


class DummyGlyphClassifier(convolutional.Convolutional1DGlyphClassifier):
  """A 1D convolutional glyph classifier with constant predictions.

  The predictions have shape (num_staves, num_stafflines, width).
  """

  def __init__(self, predictions):
    super(DummyGlyphClassifier, self).__init__(run_min_length=1)
    self.predictions = predictions

  @property
  def staffline_predictions(self):
    return tf.constant(self.predictions)
