"""Tests for fastgen."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import fastgen


class FastegenTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {'shape': [3, 10, 10, 3]},
      {'shape': [4, 5, 5, 1]})
  def testPixelNorm(self, shape, eps=1e-8):
    images_np = np.random.randn(*shape)
    images_tf = tf.convert_to_tensor(images_np)
    with self.test_session() as sess:
      r_tf = sess.run(images_tf)
      self.assertEqual(images_np.shape, r_tf.shape)


if __name__ == '__main__':
  tf.test.main()
