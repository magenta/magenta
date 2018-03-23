"""Tests for binary morphology."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.vision import morphology


class MorphologyTest(tf.test.TestCase):

  def testMorphology_false(self):
    for op in [morphology.binary_erosion, morphology.binary_dilation]:
      with self.test_session():
        self.assertAllEqual(
            op(tf.zeros((5, 3), tf.bool), n=1).eval(), np.zeros((5, 3),
                                                                np.bool))

  def testErosion_small(self):
    with self.test_session():
      self.assertAllEqual(
          morphology.binary_erosion(
              tf.cast([[0, 1, 0], [1, 1, 1], [0, 1, 0]], tf.bool), n=1).eval(),
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]])

  def testErosion(self):
    with self.test_session():
      self.assertAllEqual(
          morphology.binary_erosion(
              tf.cast(
                  [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]],
                  tf.bool),
              n=1).eval(),
          np.asarray(
              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
              np.bool))  # pyformat: disable

  def testDilation(self):
    with self.test_session():
      self.assertAllEqual(
          morphology.binary_dilation(
              tf.cast(
                  [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]],
                  tf.bool),
              n=1).eval(),
          np.asarray(
              [[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]],
              np.bool))  # pyformat: disable


if __name__ == '__main__':
  tf.test.main()
