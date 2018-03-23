"""Tests for image utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.vision import images


class ImagesTest(tf.test.TestCase):

  def testTranslate(self):
    with self.test_session():
      arr = tf.reshape(tf.range(9), (3, 3))
      self.assertAllEqual(
          images.translate(arr, 0, -1).eval(), [[3, 4, 5], [6, 7, 8], [0, 0,
                                                                       0]])
      self.assertAllEqual(
          images.translate(arr, 0, 1).eval(), [[0, 0, 0], [0, 1, 2], [3, 4, 5]])
      self.assertAllEqual(
          images.translate(arr, -1, 0).eval(), [[1, 2, 0], [4, 5, 0], [7, 8,
                                                                       0]])
      self.assertAllEqual(
          images.translate(arr, 1, 0).eval(), [[0, 0, 1], [0, 3, 4], [0, 6, 7]])


if __name__ == '__main__':
  tf.test.main()
