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

"""Tests for MusicVAE lstm_utils library."""
from magenta.models.music_vae import lstm_utils
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


class LstmUtilsTest(tf.test.TestCase):

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
