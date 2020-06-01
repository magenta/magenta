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

"""Tests for Magenta's Tensor2Tensor modalities."""
import functools
from magenta.models.score2perf import modalities
import numpy as np
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import expert_utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ModalitiesTest(tf.test.TestCase):

  def testBottomInputs(self):
    """Adapted from tensor2tensor/layers/modalities_test.py."""
    batch_size = 10
    num_datashards = 5
    length = 5
    vocab_size = [2000, 500, 2500]
    hidden_size = 9
    model_hparams = common_hparams.basic_params1()
    model_hparams.hidden_size = hidden_size
    model_hparams.mode = tf.estimator.ModeKeys.TRAIN
    x = np.stack([
        -1 + np.random.random_integers(
            vocab_size[i], size=(batch_size, length, 1))
        for i in range(len(vocab_size))
    ], axis=3)
    data_parallelism = expert_utils.Parallelism(
        ['/device:CPU:0'] * num_datashards)
    bottom = functools.partial(modalities.bottom,
                               model_hparams=model_hparams,
                               vocab_size=vocab_size)
    with self.test_session() as session:
      xs = tf.split(x, num_datashards)
      sharded_output = data_parallelism(bottom, xs)
      output = tf.concat(sharded_output, 0)
      session.run(tf.global_variables_initializer())
      res = session.run(output)
    self.assertEqual(res.shape, (batch_size, length, 1, hidden_size))


if __name__ == '__main__':
  tf.test.main()
