# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tf_lib."""

# internal imports
import tensorflow as tf

from magenta.common import tf_lib


class TfLibTest(tf.test.TestCase):

  def testHParams(self):
    hparams = tf_lib.HParams(batch_size=128, hidden_size=256)
    hparams.parse('{"hidden_size":512, "rnn_layers":[128, 128]}')
    self.assertEquals(hparams.batch_size, 128)
    self.assertEquals(
        hparams.values(),
        {'batch_size': 128, 'hidden_size': 512, 'rnn_layers': [128, 128]})


if __name__ == '__main__':
  tf.test.main()
