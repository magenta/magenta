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
"""Tests for basic_rnn_encoder_decoder."""

# internal imports
import tensorflow as tf

from magenta.models.basic_rnn import basic_rnn_encoder_decoder
from magenta.music import melodies_lib

NOTE_OFF = melodies_lib.MELODY_NOTE_OFF
NO_EVENT = melodies_lib.MELODY_NO_EVENT


class BasicRnnEncoderDecoderTest(tf.test.TestCase):

  def testDefaultRange(self):
    basic_rnn_encoder_decoder.MIN_NOTE = 48
    basic_rnn_encoder_decoder.MAX_NOTE = 84
    self.assertEqual(basic_rnn_encoder_decoder.TRANSPOSE_TO_KEY, 0)

    melody_encoder_decoder = basic_rnn_encoder_decoder.MelodyEncoderDecoder()
    self.assertEqual(melody_encoder_decoder.input_size, 38)
    self.assertEqual(melody_encoder_decoder.num_classes, 38)

    melody = melodies_lib.MonophonicMelody()
    melody_events = [48, NO_EVENT, 49, 83, NOTE_OFF]
    melody.from_event_list(melody_events)

    expected_inputs = [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    expected_labels = [2, 0, 3, 37, 1]
    for i in xrange(len(melody_events)):
      self.assertListEqual(melody_encoder_decoder.events_to_input(melody, i),
                           expected_inputs[i])
      self.assertEqual(melody_encoder_decoder.events_to_label(melody, i),
                       expected_labels[i])
      self.assertEqual(
          melody_encoder_decoder.class_index_to_event(expected_labels[i], None),
          melody_events[i])
      partial_melody = melodies_lib.MonophonicMelody()
      partial_melody.from_event_list(melody_events[:i])
      softmax = [[[0.0] * melody_encoder_decoder.num_classes]]
      softmax[0][0][expected_labels[i]] = 1.0
      melody_encoder_decoder.extend_event_sequences([partial_melody], softmax)
      self.assertEqual(list(partial_melody)[-1], melody_events[i])

    melodies = [melody, melody]
    expected_full_length_inputs_batch = [expected_inputs, expected_inputs]
    expected_last_event_inputs_batch = [expected_inputs[-1:],
                                        expected_inputs[-1:]]
    self.assertListEqual(
        expected_full_length_inputs_batch,
        melody_encoder_decoder.get_inputs_batch(melodies, True))
    self.assertListEqual(
        expected_last_event_inputs_batch,
        melody_encoder_decoder.get_inputs_batch(melodies))

  def testCustomRange(self):
    basic_rnn_encoder_decoder.MIN_NOTE = 24
    basic_rnn_encoder_decoder.MAX_NOTE = 36

    melody_encoder_decoder = basic_rnn_encoder_decoder.MelodyEncoderDecoder()
    self.assertEqual(melody_encoder_decoder.input_size, 14)
    self.assertEqual(melody_encoder_decoder.num_classes, 14)

    melody = melodies_lib.MonophonicMelody()
    melody_events = [24, NO_EVENT, 25, 35, NOTE_OFF]
    melody.from_event_list(melody_events)

    expected_inputs = [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    expected_labels = [2, 0, 3, 13, 1]

    for i in xrange(len(melody_events)):
      self.assertListEqual(melody_encoder_decoder.events_to_input(melody, i),
                           expected_inputs[i])
      self.assertEqual(melody_encoder_decoder.events_to_label(melody, i),
                       expected_labels[i])
      self.assertEqual(
          melody_encoder_decoder.class_index_to_event(expected_labels[i], None),
          melody_events[i])
      partial_melody = melodies_lib.MonophonicMelody()
      partial_melody.from_event_list(melody_events[:i])
      softmax = [[[0.0] * melody_encoder_decoder.num_classes]]
      softmax[0][0][expected_labels[i]] = 1.0
      melody_encoder_decoder.extend_event_sequences([partial_melody], softmax)
      self.assertEqual(list(partial_melody)[-1], melody_events[i])

    melodies = [melody, melody]
    expected_full_length_inputs_batch = [expected_inputs, expected_inputs]
    expected_last_event_inputs_batch = [expected_inputs[-1:],
                                        expected_inputs[-1:]]
    self.assertListEqual(
        expected_full_length_inputs_batch,
        melody_encoder_decoder.get_inputs_batch(melodies, True))
    self.assertListEqual(
        expected_last_event_inputs_batch,
        melody_encoder_decoder.get_inputs_batch(melodies))


if __name__ == '__main__':
  tf.test.main()
