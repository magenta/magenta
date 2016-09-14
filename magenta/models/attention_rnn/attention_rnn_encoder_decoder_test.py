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
"""Tests for attention_rnn_encoder_decoder."""

# internal imports
import attention_rnn_encoder_decoder
import tensorflow as tf
from magenta.lib import melodies_lib

NOTE_OFF = melodies_lib.NOTE_OFF
NO_EVENT = melodies_lib.NO_EVENT


class AttentionRnnEncoderDecoderTest(tf.test.TestCase):

  def testDefaultRange(self):
    attention_rnn_encoder_decoder.MIN_NOTE = 48
    attention_rnn_encoder_decoder.MAX_NOTE = 84
    self.assertEqual(attention_rnn_encoder_decoder.TRANSPOSE_TO_KEY, 0)

    melody_encoder_decoder = (
        attention_rnn_encoder_decoder.MelodyEncoderDecoder())
    self.assertEqual(melody_encoder_decoder.input_size, 74)
    self.assertEqual(melody_encoder_decoder.num_classes, 40)

    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list([48, NO_EVENT, 49, 83, NOTE_OFF] + [NO_EVENT] * 11 +
                           [48, NOTE_OFF] + [NO_EVENT] * 14 +
                           [48, NOTE_OFF, 49, 82])

    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 0),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
         1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
         1.0, 1.0, 0.0, 1.0, 0.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 1),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
         1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
         1.0, 1.0, 0.0, 1.0, 0.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 2),
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
         1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 0.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 3),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
         -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
         0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 4),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
         0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 15),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
         -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
         0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 16),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0,
         1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 17),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0,
         -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 32),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0,
         1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 33),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0,
         -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 34),
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
         1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 35),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
         -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 0.0])

    for i, j in zip([0, 1, 2, 3, 4, 15, 16, 17, 32, 33, 34, 35],
                    [0, 39, 1, 35, 37, 39, 38, 37, 39, 38, 39, 34]):
      self.assertEqual(melody_encoder_decoder.melody_to_label(melody, i), j)

    temp_events = melody._events[:]
    for i, j, k in zip([0, 1, 2, 3, 4, 15, 16, 17, 32, 33, 34, 35],
                       [0, 39, 1, 35, 37, 39, 38, 37, 39, 38, 39, 34],
                       [48, NO_EVENT, 49, 83, NOTE_OFF,
                        NO_EVENT, 48, NOTE_OFF,
                        48, NOTE_OFF, 49, 82]):
      melody._events = temp_events[:i]
      self.assertEqual(
          melody_encoder_decoder.class_index_to_melody_event(j, melody), k)

  def testCustomRange(self):
    attention_rnn_encoder_decoder.MIN_NOTE = 24
    attention_rnn_encoder_decoder.MAX_NOTE = 36
    self.assertEqual(attention_rnn_encoder_decoder.TRANSPOSE_TO_KEY, 0)

    melody_encoder_decoder = (
        attention_rnn_encoder_decoder.MelodyEncoderDecoder())
    self.assertEqual(melody_encoder_decoder.input_size, 50)
    self.assertEqual(melody_encoder_decoder.num_classes, 16)

    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list([24, NO_EVENT, 25, 35, NOTE_OFF] + [NO_EVENT] * 11 +
                           [24, NOTE_OFF] + [NO_EVENT] * 14 +
                           [24, NOTE_OFF, 25, 34])

    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 0),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0,
         1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
         1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 1),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0,
         1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
         1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 2),
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 3),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
         1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0,
         1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 4),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0,
         1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 15),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 16),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         1.0, -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 1.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 17),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
         1.0, -1.0, 0.0, 0.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 1.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 32),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 1.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 33),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
         1.0, -1.0, 1.0, 0.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 1.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 34),
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
         1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    self.assertEqual(
        melody_encoder_decoder.melody_to_input(melody, 35),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
         1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    for i, j in zip([0, 1, 2, 3, 4, 15, 16, 17, 32, 33, 34, 35],
                    [0, 15, 1, 11, 13, 15, 14, 13, 15, 14, 15, 10]):
      self.assertEqual(melody_encoder_decoder.melody_to_label(melody, i), j)

    temp_events = melody._events[:]
    for i, j, k in zip([0, 1, 2, 3, 4, 15, 16, 17, 32, 33, 34, 35],
                       [0, 15, 1, 11, 13, 15, 14, 13, 15, 14, 15, 10],
                       [24, NO_EVENT, 25, 35, NOTE_OFF,
                        NO_EVENT, 24, NOTE_OFF,
                        24, NOTE_OFF, 25, 34]):
      melody._events = temp_events[:i]
      self.assertEqual(
          melody_encoder_decoder.class_index_to_melody_event(j, melody), k)


if __name__ == '__main__':
  tf.test.main()
