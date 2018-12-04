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

import math

import tensorflow as tf

from magenta.music import performance_encoder_decoder
from magenta.music import performance_lib
from magenta.music.performance_encoder_decoder import ModuloPerformanceEventSequenceEncoderDecoder
from magenta.music.performance_encoder_decoder import NotePerformanceEventSequenceEncoderDecoder
from magenta.music.performance_encoder_decoder import PerformanceModuloEncoding
from magenta.music.performance_lib import PerformanceEvent


cos = math.cos
sin = math.sin
pi = math.pi


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
  """Test class for PerformanceModuloEncoding."""

  def setUp(self):
    self._num_velocity_bins = 16
    self._max_shift_steps = performance_lib.DEFAULT_MAX_SHIFT_STEPS
    self.enc = PerformanceModuloEncoding(
        num_velocity_bins=self._num_velocity_bins,
        max_shift_steps=self._max_shift_steps)

    self._expected_input_size = (
        2 * performance_encoder_decoder.MODULO_PITCH_ENCODER_WIDTH +
        performance_encoder_decoder.MODULO_VELOCITY_ENCODER_WIDTH +
        performance_encoder_decoder.MODULO_TIME_SHIFT_ENCODER_WIDTH)

    self._expected_num_classes = (self._num_velocity_bins +
                                  self._max_shift_steps +
                                  (performance_lib.MAX_MIDI_PITCH -
                                   performance_lib.MIN_MIDI_PITCH + 1) * 2)

  def testInputSize(self):
    self.assertEqual(self._expected_input_size, self.enc.input_size)

  def testEmbedPitchClass(self):
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
      actual_embedding = self.enc.embed_pitch_class(note)
      self.assertEqual(actual_embedding[0], expected_embedding[0])
      self.assertEqual(actual_embedding[1], expected_embedding[1])

  def testEmbedNote(self):
    # The following are true only for semitone_steps = 1.
    base = 72.0
    expected_pairs = [
        (0, (cos(0.0), sin(0.0))),
        (13, (cos(pi * 13.0 / base), sin(pi * 13.0 / base))),
        (26, (cos(pi * 26.0 / base), sin(pi * 26.0 / base))),
        (39, (cos(pi * 39.0 / base), sin(pi * 39.0 / base))),
        (52, (cos(pi * 52.0 / base), sin(pi * 52.0 / base))),
        (65, (cos(pi * 65.0 / base), sin(pi * 65.0 / base))),
        (78, (cos(pi * 78.0 / base), sin(pi * 78.0 / base))),
        (91, (cos(pi * 91.0 / base), sin(pi * 91.0 / base))),
        (104, (cos(pi * 104.0 / base), sin(pi * 104.0 / base))),
        (117, (cos(pi * 117.0 / base), sin(pi * 117.0 / base))),
        (130, (cos(pi * 130.0 / base), sin(pi * 130.0 / base))),
        (143, (cos(pi * 143.0 / base), sin(pi * 143.0 / base)))]

    for note, expected_embedding in expected_pairs:
      actual_embedding = self.enc.embed_note(note)
      self.assertEqual(actual_embedding[0], expected_embedding[0])
      self.assertEqual(actual_embedding[1], expected_embedding[1])

  def testEmbedTimeShift(self):
    # The following are true only for semitone_steps = 1.
    base = self._max_shift_steps  # 100
    expected_pairs = [
        (0, (cos(0.0), sin(0.0))),
        (2, (cos(2.0 * pi * 2.0 / base), sin(2.0 * pi * 2.0 / base))),
        (5, (cos(2.0 * pi * 5.0 / base), sin(2.0 * pi * 5.0 / base))),
        (13, (cos(2.0 * pi * 13.0 / base), sin(2.0 * pi * 13.0 / base))),
        (20, (cos(2.0 * pi * 20.0 / base), sin(2.0 * pi * 20.0 / base))),
        (45, (cos(2.0 * pi * 45.0 / base), sin(2.0 * pi * 45.0 / base))),
        (70, (cos(2.0 * pi * 70.0 / base), sin(2.0 * pi * 70.0 / base))),
        (99, (cos(2.0 * pi * 99.0 / base), sin(2.0 * pi * 99.0 / base)))]

    for time_shift, expected_embedding in expected_pairs:
      actual_embedding = self.enc.embed_time_shift(time_shift)
      self.assertEqual(actual_embedding[0], expected_embedding[0])
      self.assertEqual(actual_embedding[1], expected_embedding[1])

  def testEmbedVelocity(self):
    # The following are true only for semitone_steps = 1.
    base = self._num_velocity_bins  # 16
    expected_pairs = [
        (0, (cos(0.0), sin(0.0))),
        (2, (cos(2.0 * pi * 2.0 / base), sin(2.0 * pi * 2.0 / base))),
        (5, (cos(2.0 * pi * 5.0 / base), sin(2.0 * pi * 5.0 / base))),
        (7, (cos(2.0 * pi * 7.0 / base), sin(2.0 * pi * 7.0 / base))),
        (10, (cos(2.0 * pi * 10.0 / base), sin(2.0 * pi * 10.0 / base))),
        (13, (cos(2.0 * pi * 13.0 / base), sin(2.0 * pi * 13.0 / base))),
        (15, (cos(2.0 * pi * 15.0 / base), sin(2.0 * pi * 15.0 / base)))]

    for velocity, expected_embedding in expected_pairs:
      actual_embedding = self.enc.embed_velocity(velocity)
      self.assertEqual(actual_embedding[0], expected_embedding[0])
      self.assertEqual(actual_embedding[1], expected_embedding[1])

  def testEncodeModuloEvent(self):
    expected_pairs = [
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=60),
         (0, PerformanceEvent.NOTE_ON, 60)),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=0),
         (0, PerformanceEvent.NOTE_ON, 0)),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=127),
         (0, PerformanceEvent.NOTE_ON, 127)),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=72),
         (5, PerformanceEvent.NOTE_OFF, 72)),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=0),
         (5, PerformanceEvent.NOTE_OFF, 0)),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=127),
         (5, PerformanceEvent.NOTE_OFF, 127)),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=10),
         (10, PerformanceEvent.TIME_SHIFT, 9)),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=1),
         (10, PerformanceEvent.TIME_SHIFT, 0)),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=100),
         (10, PerformanceEvent.TIME_SHIFT, 99)),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=5),
         (13, PerformanceEvent.VELOCITY, 4)),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=1),
         (13, PerformanceEvent.VELOCITY, 0)),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=16),
         (13, PerformanceEvent.VELOCITY, 15)),
    ]

    # expected_encoded_modulo_event is of the following form:
    # (offset, encoder_width, event_type, value, bins)
    for event, expected_encoded_modulo_event in expected_pairs:
      actual_encoded_modulo_event = self.enc.encode_modulo_event(event)
      self.assertEqual(actual_encoded_modulo_event,
                       expected_encoded_modulo_event)


class ModuloPerformanceEventSequenceEncoderTest(tf.test.TestCase):
  """Test class for ModuloPerformanceEventSequenceEncoder.

  ModuloPerformanceEventSequenceEncoderDecoder is tightly coupled with the
  PerformanceModuloEncoding, and PerformanceOneHotEncoding classes. As a result,
  in the test set up, the test object is initialized with one of each objects
  and tested accordingly. Since this class only modifies the input encoding
  of performance events, and otherwise its treatment of labels is the same as
  OneHotEventSequenceEncoderDecoder, the events_to_labels(), and
  class_index_to_event() methods of the class are not tested.
  """

  def setUp(self):
    self._num_velocity_bins = 32
    self._max_shift_steps = 100
    self.enc = ModuloPerformanceEventSequenceEncoderDecoder(
        num_velocity_bins=self._num_velocity_bins,
        max_shift_steps=self._max_shift_steps)

    self._expected_input_size = (
        2 * performance_encoder_decoder.MODULO_PITCH_ENCODER_WIDTH +
        performance_encoder_decoder.MODULO_VELOCITY_ENCODER_WIDTH +
        performance_encoder_decoder.MODULO_TIME_SHIFT_ENCODER_WIDTH)

    self._expected_num_classes = (self._num_velocity_bins +
                                  self._max_shift_steps +
                                  2 * (performance_lib.MAX_MIDI_PITCH -
                                       performance_lib.MIN_MIDI_PITCH + 1))

  def testInputSize(self):
    self.assertEqual(self._expected_input_size, self.enc.input_size)

  def testNumClasses(self):
    self.assertEqual(self._expected_num_classes, self.enc.num_classes)

  def testDefaultEventLabel(self):
    label = self._expected_num_classes - self._num_velocity_bins - 1
    self.assertEqual(label, self.enc.default_event_label)

  def testEventsToInput(self):
    num_shift_bins = self._max_shift_steps
    num_velocity_bins = self._num_velocity_bins
    slow_base = 2.0 * pi / 144.0
    fast_base = 2.0 * pi / 12.0
    shift_base = 2.0 * pi / num_shift_bins
    velocity_base = 2.0 * pi / num_velocity_bins

    expected_pairs = [
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=60),
         [1.0, cos(60.0 * slow_base), sin(60.0 * slow_base),
          cos(60.0 * fast_base), sin(60.0 * fast_base),
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=0),
         [1.0, cos(0.0 * slow_base), sin(0.0 * slow_base),
          cos(0.0 * fast_base), sin(0.0 * fast_base),
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=127),
         [1.0, cos(127.0 * slow_base), sin(127.0 * slow_base),
          cos(127.0 * fast_base), sin(127.0 * fast_base),
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=72),
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
          cos(72.0 * slow_base), sin(72.0 * slow_base),
          cos(72.0 * fast_base), sin(72.0 * fast_base),
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=0),
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
          cos(0.0 * slow_base), sin(0.0 * slow_base),
          cos(0.0 * fast_base), sin(0.0 * fast_base),
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (PerformanceEvent(
            event_type=PerformanceEvent.NOTE_OFF, event_value=127),
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
          cos(127.0 * slow_base), sin(127.0 * slow_base),
          cos(127.0 * fast_base), sin(127.0 * fast_base),
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=10),
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          1.0, cos(9.0 * shift_base), sin(9.0 * shift_base),
          0.0, 0.0, 0.0]),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=1),
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          1.0, cos(0.0 * shift_base), sin(0.0 * shift_base),
          0.0, 0.0, 0.0]),
        (PerformanceEvent(
            event_type=PerformanceEvent.TIME_SHIFT, event_value=100),
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          1.0, cos(99.0 * shift_base), sin(99.0 * shift_base),
          0.0, 0.0, 0.0]),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=5),
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          1.0, cos(4.0 * velocity_base), sin(4.0 * velocity_base)]),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=1),
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          1.0, cos(0.0 * velocity_base), sin(0.0 * velocity_base)]),
        (PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=16),
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          1.0, cos(15.0 * velocity_base), sin(15.0 * velocity_base)]),
    ]

    events = []
    position = 0
    for event, expected_encoded_modulo_event in expected_pairs:
      events += [event]
      actual_encoded_modulo_event = self.enc.events_to_input(events, position)
      position += 1
      for i in range(self._expected_input_size):
        self.assertAlmostEqual(expected_encoded_modulo_event[i],
                               actual_encoded_modulo_event[i])


class NotePerformanceEventSequenceEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = NotePerformanceEventSequenceEncoderDecoder(
        num_velocity_bins=16, max_shift_steps=99, max_duration_steps=500)

    self.assertEqual(10, self.enc.shift_steps_segments)
    self.assertEqual(20, self.enc.duration_steps_segments)

  def testEncodeDecode(self):
    pe = PerformanceEvent
    performance = [
        (pe(pe.TIME_SHIFT, 0), pe(pe.NOTE_ON, 60),
         pe(pe.VELOCITY, 13), pe(pe.DURATION, 401)),
        (pe(pe.TIME_SHIFT, 55), pe(pe.NOTE_ON, 64),
         pe(pe.VELOCITY, 13), pe(pe.DURATION, 310)),
        (pe(pe.TIME_SHIFT, 99), pe(pe.NOTE_ON, 67),
         pe(pe.VELOCITY, 16), pe(pe.DURATION, 100)),
        (pe(pe.TIME_SHIFT, 0), pe(pe.NOTE_ON, 67),
         pe(pe.VELOCITY, 16), pe(pe.DURATION, 1)),
        (pe(pe.TIME_SHIFT, 0), pe(pe.NOTE_ON, 67),
         pe(pe.VELOCITY, 16), pe(pe.DURATION, 500)),
    ]

    labels = [self.enc.events_to_label(performance, i)
              for i in range(len(performance))]

    expected_labels = [
        (0, 0, 60, 12, 16, 0),
        (5, 5, 64, 12, 12, 9),
        (9, 9, 67, 15, 3, 24),
        (0, 0, 67, 15, 0, 0),
        (0, 0, 67, 15, 19, 24),
    ]

    self.assertEqual(expected_labels, labels)

    inputs = [self.enc.events_to_input(performance, i)
              for i in range(len(performance))]

    for input_ in inputs:
      self.assertEqual(6, input_.nonzero()[0].shape[0])

    decoded_performance = [self.enc.class_index_to_event(label, None)
                           for label in labels]

    self.assertEqual(performance, decoded_performance)


if __name__ == '__main__':
  tf.test.main()
