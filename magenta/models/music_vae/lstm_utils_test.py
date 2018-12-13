# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for MusicVAE lstm_utils library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from magenta.models.music_vae import lstm_utils
from tensorflow.contrib import rnn
from tensorflow.python.util import nest


class LstmUtilsTest(tf.test.TestCase):

  def testStateTupleToCudnnLstmState(self):
    with self.test_session():
      h, c = lstm_utils.state_tuples_to_cudnn_lstm_state(
          (rnn.LSTMStateTuple(h=np.arange(10).reshape(5, 2),
                              c=np.arange(10, 20).reshape(5, 2)),))
      self.assertAllEqual(np.arange(10).reshape(1, 5, 2), h.eval())
      self.assertAllEqual(np.arange(10, 20).reshape(1, 5, 2), c.eval())

      h, c = lstm_utils.state_tuples_to_cudnn_lstm_state(
          (rnn.LSTMStateTuple(h=np.arange(10).reshape(5, 2),
                              c=np.arange(20, 30).reshape(5, 2)),
           rnn.LSTMStateTuple(h=np.arange(10, 20).reshape(5, 2),
                              c=np.arange(30, 40).reshape(5, 2))))
      self.assertAllEqual(np.arange(20).reshape(2, 5, 2), h.eval())
      self.assertAllEqual(np.arange(20, 40).reshape(2, 5, 2), c.eval())

  def testCudnnLstmState(self):
    with self.test_session() as sess:
      lstm_state = lstm_utils.cudnn_lstm_state_to_state_tuples(
          (np.arange(10).reshape(1, 5, 2), np.arange(10, 20).reshape(1, 5, 2)))
      nest.map_structure(
          self.assertAllEqual,
          (rnn.LSTMStateTuple(h=np.arange(10).reshape(5, 2),
                              c=np.arange(10, 20).reshape(5, 2)),),
          sess.run(lstm_state))

      lstm_state = lstm_utils.cudnn_lstm_state_to_state_tuples(
          (np.arange(20).reshape(2, 5, 2), np.arange(20, 40).reshape(2, 5, 2)))
      nest.map_structure(
          self.assertAllEqual,
          (rnn.LSTMStateTuple(h=np.arange(10).reshape(5, 2),
                              c=np.arange(20, 30).reshape(5, 2)),
           rnn.LSTMStateTuple(h=np.arange(10, 20).reshape(5, 2),
                              c=np.arange(30, 40).reshape(5, 2))),
          sess.run(lstm_state))

  def testGetFinal(self):
    with self.test_session():
      sequences = np.arange(40).reshape((4, 5, 2))
      lengths = np.array([0, 1, 2, 5])
      expected_values = np.array([[0, 1], [10, 11], [22, 23], [38, 39]])

      self.assertAllEqual(
          expected_values,
          lstm_utils.get_final(sequences, lengths, time_major=False).eval())

      self.assertAllEqual(
          expected_values,
          lstm_utils.get_final(
              np.transpose(sequences, [1, 0, 2]),
              lengths,
              time_major=True).eval())

  def testSetFinal(self):
    with self.test_session():
      sequences = np.arange(40, dtype=np.float32).reshape(4, 5, 2)
      lengths = np.array([0, 1, 2, 5])
      final_values = np.arange(40, 48, dtype=np.float32).reshape(4, 2)
      expected_result = sequences.copy()
      for i, l in enumerate(lengths):
        expected_result[i, l:] = 0.0
        expected_result[i, max(0, l-1)] = final_values[i]
      expected_result[range(4), np.maximum(0, lengths - 1)] = final_values

      self.assertAllEqual(
          expected_result,
          lstm_utils.set_final(
              sequences, lengths, final_values, time_major=False).eval())

      self.assertAllEqual(
          np.transpose(expected_result, [1, 0, 2]),
          lstm_utils.set_final(
              np.transpose(sequences, [1, 0, 2]),
              lengths,
              final_values,
              time_major=True).eval())

  def testMaybeSplitSequenceLengths(self):
    with self.test_session():
      # Test unsplit.
      sequence_length = tf.constant([8, 0, 8], tf.int32)
      num_splits = 4
      total_length = 8
      expected_split_length = np.array([[2, 2, 2, 2],
                                        [0, 0, 0, 0],
                                        [2, 2, 2, 2]])
      split_length = lstm_utils.maybe_split_sequence_lengths(
          sequence_length, num_splits, total_length).eval()
      self.assertAllEqual(expected_split_length, split_length)

      # Test already split.
      presplit_length = np.array([[0, 2, 1, 2],
                                  [0, 0, 0, 0],
                                  [1, 1, 1, 1]], np.int32)
      split_length = lstm_utils.maybe_split_sequence_lengths(
          tf.constant(presplit_length), num_splits, total_length).eval()
      self.assertAllEqual(presplit_length, split_length)

      # Test invalid total length.
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sequence_length = tf.constant([8, 0, 7])
        lstm_utils.maybe_split_sequence_lengths(
            sequence_length, num_splits, total_length).eval()

      # Test invalid segment length.
      with self.assertRaises(tf.errors.InvalidArgumentError):
        presplit_length = np.array([[0, 2, 3, 1],
                                    [0, 0, 0, 0],
                                    [1, 1, 1, 1]], np.int32)
        lstm_utils.maybe_split_sequence_lengths(
            tf.constant(presplit_length), num_splits, total_length).eval()

if __name__ == '__main__':
  tf.test.main()
