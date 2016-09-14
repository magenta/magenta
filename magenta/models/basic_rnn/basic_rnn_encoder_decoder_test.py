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
import basic_rnn_encoder_decoder
import tensorflow as tf
from magenta.lib import melodies_lib

NOTE_OFF = melodies_lib.NOTE_OFF
NO_EVENT = melodies_lib.NO_EVENT


class BasicRnnEncoderDecoderTest(tf.test.TestCase):

  def testDefaultRange(self):
    basic_rnn_encoder_decoder.MIN_NOTE = 48
    basic_rnn_encoder_decoder.MAX_NOTE = 84
    self.assertEqual(basic_rnn_encoder_decoder.TRANSPOSE_TO_KEY, 0)

    melody_encoder_decoder = basic_rnn_encoder_decoder.MelodyEncoderDecoder()
    self.assertEqual(melody_encoder_decoder.input_size, 38)
    self.assertEqual(melody_encoder_decoder.num_classes, 38)

    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list([48, NO_EVENT, 49, 83, NOTE_OFF])

    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 0),
                     [1.0 if i == 2 else 0.0 for i in xrange(38)])
    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 1),
                     [1.0 if i == 0 else 0.0 for i in xrange(38)])
    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 2),
                     [1.0 if i == 3 else 0.0 for i in xrange(38)])
    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 3),
                     [1.0 if i == 37 else 0.0 for i in xrange(38)])
    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 4),
                     [1.0 if i == 1 else 0.0 for i in xrange(38)])

    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 0), 2)
    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 1), 0)
    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 2), 3)
    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 3), 37)
    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 4), 1)

    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(2, None), 48)
    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(0, None), NO_EVENT)
    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(3, None), 49)
    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(37, None), 83)
    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(1, None), NOTE_OFF)

  def testCustomRange(self):
    basic_rnn_encoder_decoder.MIN_NOTE = 24
    basic_rnn_encoder_decoder.MAX_NOTE = 36

    melody_encoder_decoder = basic_rnn_encoder_decoder.MelodyEncoderDecoder()
    self.assertEqual(melody_encoder_decoder.input_size, 14)
    self.assertEqual(melody_encoder_decoder.num_classes, 14)

    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list([24, NO_EVENT, 25, 35, NOTE_OFF])

    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 0),
                     [1.0 if i == 2 else 0.0 for i in xrange(14)])
    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 1),
                     [1.0 if i == 0 else 0.0 for i in xrange(14)])
    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 2),
                     [1.0 if i == 3 else 0.0 for i in xrange(14)])
    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 3),
                     [1.0 if i == 13 else 0.0 for i in xrange(14)])
    self.assertEqual(melody_encoder_decoder.melody_to_input(melody, 4),
                     [1.0 if i == 1 else 0.0 for i in xrange(14)])

    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 0), 2)
    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 1), 0)
    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 2), 3)
    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 3), 13)
    self.assertEqual(melody_encoder_decoder.melody_to_label(melody, 4), 1)

    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(2, None), 24)
    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(0, None), NO_EVENT)
    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(3, None), 25)
    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(13, None), 35)
    self.assertEqual(
        melody_encoder_decoder.class_index_to_melody_event(1, None), NOTE_OFF)


if __name__ == '__main__':
  tf.test.main()
