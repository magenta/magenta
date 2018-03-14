"""Tests for Convolutional1DGlyphClassifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import pandas as pd
import tensorflow as tf

from magenta.models.omr.glyphs import base
from magenta.models.omr.glyphs import convolutional
from magenta.models.omr.glyphs import testing
from magenta.models.omr.protobuf import musicscore_pb2

STAFF_INDEX = base.GlyphsTensorColumns.STAFF_INDEX
Y_POSITION = base.GlyphsTensorColumns.Y_POSITION
X = base.GlyphsTensorColumns.X
TYPE = base.GlyphsTensorColumns.TYPE


class ConvolutionalTest(tf.test.TestCase):

  def testGetGlyphsPage(self):
    # Refer to testing.py for the glyphs array.
    # pyformat: disable
    glyphs = pd.DataFrame(
        [
            {STAFF_INDEX: 0, Y_POSITION: 0, X: 0, TYPE: 3},
            {STAFF_INDEX: 0, Y_POSITION: -1, X: 1, TYPE: 4},
            {STAFF_INDEX: 0, Y_POSITION: 0, X: 2, TYPE: 5},
            {STAFF_INDEX: 0, Y_POSITION: 1, X: 4, TYPE: 2},
            {STAFF_INDEX: 1, Y_POSITION: 1, X: 2, TYPE: 3},
            {STAFF_INDEX: 1, Y_POSITION: 0, X: 2, TYPE: 5},
            {STAFF_INDEX: 1, Y_POSITION: -1, X: 4, TYPE: 3},
            {STAFF_INDEX: 1, Y_POSITION: -1, X: 5, TYPE: 5},
        ],
        columns=[STAFF_INDEX, Y_POSITION, X, TYPE])
    # Compare glyphs (rows in the glyphs array) regardless of their position in
    # the array (they are not required to be sorted).
    self.assertEqual(
        set(
            map(tuple,
                convolutional.Convolutional1DGlyphClassifier(
                    run_min_length=1)._build_detected_glyphs(
                        testing.PREDICTIONS))),
        set(map(tuple, glyphs.values)))

  def testNoGlyphs_dummyClassifier(self):

    class DummyClassifier(convolutional.Convolutional1DGlyphClassifier):
      """Outputs the classifications for no glyphs on multiple staves."""

      @property
      def staffline_predictions(self):
        return tf.fill([5, 9, 100], musicscore_pb2.Glyph.NONE)

    with self.test_session():
      self.assertAllEqual(
          DummyClassifier().get_detected_glyphs().eval(),
          np.zeros((0, 4), np.int32))


if __name__ == '__main__':
  tf.test.main()
