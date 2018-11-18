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
      {'batch_size': 1},
      {'batch_size': 10})
  def testLoadFastgenNsynth(self, batch_size):
    net = fastgen.load_fastgen_nsynth(batch_size=batch_size)
    with self.test_session() as sess:
      sess.run(net['init_ops'])
      self.assertEqual(net['X'].shape, (batch_size, 1))
      self.assertEqual(net['encoding'].shape, (batch_size, 16))
      self.assertEqual(net['predictions'].shape, (batch_size, 256))

  @parameterized.parameters(
      {'batch_size': 1, 'sample_length': 1024 * 10},
      {'batch_size': 10, 'sample_length': 1024 * 10},
      {'batch_size': 10, 'sample_length': 1024 * 20},
  )
  def testLoadNsynth(self, batch_size, sample_length):
    net = fastgen.load_nsynth(batch_size=batch_size, sample_length=sample_length)
    encodings_length = int(sample_length/512)
    with self.test_session() as sess:
      self.assertEqual(net['X'].shape, (batch_size, sample_length))
      self.assertEqual(net['encoding'].shape, (batch_size, encodings_length, 16))
      self.assertEqual(net['predictions'].shape, (batch_size * sample_length, 256))


if __name__ == '__main__':
  tf.test.main()
