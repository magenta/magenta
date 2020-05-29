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

# Lint as: python3
"""Forked classes and functions from `tf.contrib.cudnn_rnn`."""
from magenta.contrib import rnn as contrib_rnn


class CudnnCompatibleLSTMCell(contrib_rnn.LSTMBlockCell):
  """Cudnn Compatible LSTMCell.

  A simple wrapper around `tf.contrib.rnn.LSTMBlockCell` to use along with
  `tf.contrib.cudnn_rnn.CudnnLSTM`. The latter's params can be used by
  this cell seamlessly.
  """

  def __init__(self, num_units, reuse=None):
    super(CudnnCompatibleLSTMCell, self).__init__(
        num_units,
        forget_bias=0,
        cell_clip=None,
        use_peephole=False,
        reuse=reuse,
        name="cudnn_compatible_lstm_cell")
    self._names.update({"scope": "cudnn_compatible_lstm_cell"})
