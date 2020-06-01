# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for nade."""

from magenta.common.nade import Nade
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class NadeTest(tf.test.TestCase):

  def testInternalBias(self):
    batch_size = 4
    num_hidden = 6
    num_dims = 8
    test_inputs = tf.random_normal(shape=(batch_size, num_dims))
    nade = Nade(num_dims, num_hidden, internal_bias=True)
    log_prob, cond_probs = nade.log_prob(test_inputs)
    sample, sample_prob = nade.sample(n=batch_size)
    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      self.assertEqual(log_prob.eval().shape, (batch_size,))
      self.assertEqual(cond_probs.eval().shape, (batch_size, num_dims))
      self.assertEqual(sample.eval().shape, (batch_size, num_dims))
      self.assertEqual(sample_prob.eval().shape, (batch_size,))

  def testExternalBias(self):
    batch_size = 4
    num_hidden = 6
    num_dims = 8
    test_inputs = tf.random_normal(shape=(batch_size, num_dims))
    test_b_enc = tf.random_normal(shape=(batch_size, num_hidden))
    test_b_dec = tf.random_normal(shape=(batch_size, num_dims))

    nade = Nade(num_dims, num_hidden)
    log_prob, cond_probs = nade.log_prob(test_inputs, test_b_enc, test_b_dec)
    sample, sample_prob = nade.sample(b_enc=test_b_enc, b_dec=test_b_dec)
    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      self.assertEqual(log_prob.eval().shape, (batch_size,))
      self.assertEqual(cond_probs.eval().shape, (batch_size, num_dims))
      self.assertEqual(sample.eval().shape, (batch_size, num_dims))
      self.assertEqual(sample_prob.eval().shape, (batch_size,))


if __name__ == '__main__':
  tf.test.main()
