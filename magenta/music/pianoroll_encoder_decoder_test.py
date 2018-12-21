# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Tests for pianoroll_encoder_decoder."""

import numpy as np
import tensorflow as tf

from magenta.music import pianoroll_encoder_decoder


class PianorollEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = pianoroll_encoder_decoder.PianorollEncoderDecoder(5)

  def testProperties(self):
    self.assertEqual(5, self.enc.input_size)
    self.assertEqual(32, self.enc.num_classes)
    self.assertEqual(0, self.enc.default_event_label)

  def testEncodeInput(self):
    events = [(), (1, 2), (2,)]
    self.assertTrue(np.array_equal(
        np.zeros(5, np.bool), self.enc.events_to_input(events, 0)))
    self.assertTrue(np.array_equal(
        [0, 1, 1, 0, 0], self.enc.events_to_input(events, 1)))
    self.assertTrue(np.array_equal(
        [0, 0, 1, 0, 0], self.enc.events_to_input(events, 2)))

  def testEncodeLabel(self):
    events = [[], [1, 2], [2]]
    self.assertEqual(0, self.enc.events_to_label(events, 0))
    self.assertEqual(6, self.enc.events_to_label(events, 1))
    self.assertEqual(4, self.enc.events_to_label(events, 2))

  def testDecodeLabel(self):
    self.assertEqual((), self.enc.class_index_to_event(0, None))
    self.assertEqual((1, 2), self.enc.class_index_to_event(6, None))
    self.assertEqual((2,), self.enc.class_index_to_event(4, None))

  def testExtendEventSequences(self):
    seqs = ([(0,), (1, 2)], [(), ()])
    samples = ([0, 0, 0, 0, 0], [1, 1, 0, 0, 1])
    self.enc.extend_event_sequences(seqs, samples)
    self.assertEqual(([(0,), (1, 2), ()], [(), (), (0, 1, 4)]), seqs)


if __name__ == '__main__':
  tf.test.main()
