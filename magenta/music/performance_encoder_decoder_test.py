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
"""Tests for performance_encoder_decoder."""

from math import cos
from math import pi
from math import sin

# internal imports

import tensorflow as tf

from magenta.music import performance_encoder_decoder
from magenta.music import performance_lib
from magenta.music.performance_lib import PerformanceEvent


class PerformanceOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = performance_encoder_decoder.PerformanceOneHotEncoding(
        num_velocity_bins=16)

  def testEncodeDecode(self):
    expected_pairs = [
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_ON, event_value=60), 60),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_ON, event_value=0), 0),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_ON, event_value=127), 127),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=72), 200),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=0), 128),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=127), 255),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=10), 265),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=1), 256),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=100), 355),
        (PerformanceEvent(
            event_type=PerformanceEvent.VELOCITY, event_value=5), 360),
        (PerformanceEvent(
            event_type=PerformanceEvent.VELOCITY, event_value=1), 356),
        (PerformanceEvent(
            event_type=PerformanceEvent.VELOCITY, event_value=16), 371)
    ]

    for expected_event, expected_index in expected_pairs:
      index = self.enc.encode_event(expected_event)
      self.assertEqual(expected_index, index)
      event = self.enc.decode_event(expected_index)
      self.assertEqual(expected_event, event)

  def testEventToNumSteps(self):
    self.assertEqual(0, self.enc.event_to_num_steps(
        PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=60)))
    self.assertEqual(0, self.enc.event_to_num_steps(
        PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=67)))
    self.assertEqual(0, self.enc.event_to_num_steps(
        PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=10)))

    self.assertEqual(1, self.enc.event_to_num_steps(
        PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=1)))
    self.assertEqual(45, self.enc.event_to_num_steps(
        PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=45)))
    self.assertEqual(100, self.enc.event_to_num_steps(
        PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=100)))


class PerformanceModuloEncodingTest(tf.test.TestCase):
  """Test class for ModuloEventSequenceEncoder.

  The PerformanceModuloEncoding class mainly modifies the input encoding
  of events and otherwise acts very similarly to the PerformanceOneHotEncoding
  class. The encode_modulo_event() method of PerformanceModuloEncoding is tested
  through unit tests for class ModuloPerformanceEventSequenceEncoderDecoder.
  """

  def setUp(self):
    self._num_velocity_bins = 16
    self._max_shift_steps = performance_lib.DEFAULT_MAX_SHIFT_STEPS
    self.enc = performance_encoder_decoder.PerformanceModuloEncoding(
        num_velocity_bins=self._num_velocity_bins,
        max_shift_steps=self._max_shift_steps)

    self._pitch_encoder_width = 5
    self._velocity_encoder_width = 3
    self._time_shift_encoder_width = 3
    self._expected_input_size = (2 * self._pitch_encoder_width +
                                 self._velocity_encoder_width +
                                 self._time_shift_encoder_width)

    self._expected_num_classes = (self._num_velocity_bins +
                                  self._max_shift_steps +
                                  (performance_lib.MAX_MIDI_PITCH -
                                   performance_lib.MIN_MIDI_PITCH + 1) * 2)

  def testNumClasses(self):
    self.assertEqual(self._expected_num_classes, self.enc.num_classes)

  def testInputSize(self):
    self.assertEquals(self._expected_input_size, self.enc.input_size)

  def testEmbedNote(self):
    # The following are true only for semitone_steps = 1.
    expected_pairs = [
        (0, (cos(0.0), sin(0.0))),
        (1, (cos(pi / 6.0), sin(pi / 6.0))),
        (2, (cos(pi / 3.0), sin(pi / 3.0))),
        (3, (cos(pi / 2.0), sin(pi / 2.0))),
        (4, (cos(2.0 * pi / 3.0), sin(2.0 * pi / 3.0))),
        (5, (cos(5.0 * pi / 6.0), sin(5.0 * pi / 6.0))),
        (6, (cos(pi), sin(pi))),
        (7, (cos(7.0 * pi / 6.0), sin(7.0 * pi / 6.0))),
        (8, (cos(4.0 * pi / 3.0), sin(4.0 * pi / 3.0))),
        (9, (cos(3.0 * pi / 2.0), sin(3.0 * pi / 2.0))),
        (10, (cos(5.0 * pi / 3.0), sin(5.0 * pi / 3.0))),
        (11, (cos(11.0 * pi / 6.0), sin(11.0 * pi / 6.0)))]

    for note, expected_embedding in expected_pairs:
      actual_embedding = self.enc.embed_note(note)
      self.assertEqual(actual_embedding[0], expected_embedding[0])
      self.assertEqual(actual_embedding[1], expected_embedding[1])

  def testEncodeDecode(self):
    expected_pairs = [
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_ON, event_value=60), 60),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_ON, event_value=0), 0),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_ON, event_value=127), 127),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=72), 200),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=0), 128),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=127), 255),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=10), 265),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=1), 256),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=100), 355),
        (PerformanceEvent(
            event_type=PerformanceEvent.VELOCITY, event_value=5), 360),
        (PerformanceEvent(
            event_type=PerformanceEvent.VELOCITY, event_value=1), 356),
        (PerformanceEvent(
            event_type=PerformanceEvent.VELOCITY, event_value=16), 371)
    ]

    for expected_event, expected_index in expected_pairs:
      index = self.enc.encode_event(expected_event)
      self.assertEqual(expected_index, index)
      event = self.enc.decode_event(expected_index)
      self.assertEqual(expected_event, event)

  def testEventToNumSteps(self):
    self.assertEqual(0, self.enc.event_to_num_steps(
        PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=60)))
    self.assertEqual(0, self.enc.event_to_num_steps(
        PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=67)))
    self.assertEqual(0, self.enc.event_to_num_steps(
        PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=10)))

    self.assertEqual(1, self.enc.event_to_num_steps(
        PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=1)))
    self.assertEqual(45, self.enc.event_to_num_steps(
        PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=45)))
    self.assertEqual(100, self.enc.event_to_num_steps(
        PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=100)))

  def testEncodeModuloEvent(self):
    num_note_bins = (performance_lib.MAX_MIDI_PITCH -
                     performance_lib.MIN_MIDI_PITCH + 1)
    num_shift_bins = self._max_shift_steps
    num_velocity_bins = self._num_velocity_bins
    expected_pairs = [
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=60),
         (0, 5, PerformanceEvent.NOTE_ON, 60, num_note_bins)),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=0),
         (0, 5, PerformanceEvent.NOTE_ON, 0, num_note_bins)),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=127),
         (0, 5, PerformanceEvent.NOTE_ON, 127, num_note_bins)),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=72),
         (5, 5, PerformanceEvent.NOTE_OFF, 72, num_note_bins)),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=0),
         (5, 5, PerformanceEvent.NOTE_OFF, 0, num_note_bins)),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=127),
         (5, 5, PerformanceEvent.NOTE_OFF, 127, num_note_bins)),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=10),
         (10, 3, PerformanceEvent.TIME_SHIFT, 9, num_shift_bins)),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=1),
         (10, 3, PerformanceEvent.TIME_SHIFT, 0, num_shift_bins)),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=100),
         (10, 3, PerformanceEvent.TIME_SHIFT, 99, num_shift_bins)),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=5),
         (13, 3, PerformanceEvent.VELOCITY, 4, num_velocity_bins)),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=1),
         (13, 3, PerformanceEvent.VELOCITY, 0, num_velocity_bins)),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=16),
         (13, 3, PerformanceEvent.VELOCITY, 15, num_velocity_bins)),
    ]

    # expected_encoded_modulo_event is of the following form:
    # (offset, encoder_width, event_type, value, bins)
    for event, expected_encoded_modulo_event in expected_pairs:
      actual_encoded_modulo_event = self.enc.encode_modulo_event(event)
      self.assertEqual(actual_encoded_modulo_event,
                       expected_encoded_modulo_event)


class NoteDensityOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = performance_encoder_decoder.NoteDensityOneHotEncoding(
        density_bin_ranges=[1.0, 5.0])

  def testEncodeDecode(self):
    self.assertEqual(0, self.enc.encode_event(0.0))
    self.assertEqual(0, self.enc.encode_event(0.5))
    self.assertEqual(1, self.enc.encode_event(1.0))
    self.assertEqual(1, self.enc.encode_event(2.0))
    self.assertEqual(2, self.enc.encode_event(5.0))
    self.assertEqual(2, self.enc.encode_event(10.0))

    self.assertEqual(0.0, self.enc.decode_event(0))
    self.assertEqual(1.0, self.enc.decode_event(1))
    self.assertEqual(5.0, self.enc.decode_event(2))


if __name__ == '__main__':
  tf.test.main()
