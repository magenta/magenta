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
"""Tests for encoder_decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import cos
from math import pi
from math import sin

# internal imports
import numpy as np
import tensorflow as tf

from magenta.common import sequence_example_lib
from magenta.music import constants
from magenta.music import encoder_decoder
from magenta.music import performance_lib
from magenta.music import testing_lib
from magenta.music.melody_encoder_decoder import MelodyOneHotEncoding
from magenta.music.performance_encoder_decoder import PerformanceModuloEncoding
from magenta.music.performance_lib import PerformanceEvent


class OneHotEventSequenceEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = encoder_decoder.OneHotEventSequenceEncoderDecoder(
        testing_lib.TrivialOneHotEncoding(3, num_steps=range(3)))

  def testInputSize(self):
    self.assertEquals(3, self.enc.input_size)

  def testNumClasses(self):
    self.assertEqual(3, self.enc.num_classes)

  def testEventsToInput(self):
    events = [0, 1, 0, 2, 0]
    self.assertEqual([1.0, 0.0, 0.0], self.enc.events_to_input(events, 0))
    self.assertEqual([0.0, 1.0, 0.0], self.enc.events_to_input(events, 1))
    self.assertEqual([1.0, 0.0, 0.0], self.enc.events_to_input(events, 2))
    self.assertEqual([0.0, 0.0, 1.0], self.enc.events_to_input(events, 3))
    self.assertEqual([1.0, 0.0, 0.0], self.enc.events_to_input(events, 4))

  def testEventsToLabel(self):
    events = [0, 1, 0, 2, 0]
    self.assertEqual(0, self.enc.events_to_label(events, 0))
    self.assertEqual(1, self.enc.events_to_label(events, 1))
    self.assertEqual(0, self.enc.events_to_label(events, 2))
    self.assertEqual(2, self.enc.events_to_label(events, 3))
    self.assertEqual(0, self.enc.events_to_label(events, 4))

  def testClassIndexToEvent(self):
    events = [0, 1, 0, 2, 0]
    self.assertEqual(0, self.enc.class_index_to_event(0, events))
    self.assertEqual(1, self.enc.class_index_to_event(1, events))
    self.assertEqual(2, self.enc.class_index_to_event(2, events))

  def testLabelsToNumSteps(self):
    labels = [0, 1, 0, 2, 0]
    self.assertEqual(3, self.enc.labels_to_num_steps(labels))

  def testEncode(self):
    events = [0, 1, 0, 2, 0]
    sequence_example = self.enc.encode(events)
    expected_inputs = [[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]]
    expected_labels = [1, 0, 2, 0]
    expected_sequence_example = sequence_example_lib.make_sequence_example(
        expected_inputs, expected_labels)
    self.assertEqual(sequence_example, expected_sequence_example)

  def testGetInputsBatch(self):
    event_sequences = [[0, 1, 0, 2, 0], [0, 1, 2]]
    expected_inputs_1 = [[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0]]
    expected_inputs_2 = [[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]
    expected_full_length_inputs_batch = [expected_inputs_1, expected_inputs_2]
    expected_last_event_inputs_batch = [expected_inputs_1[-1:],
                                        expected_inputs_2[-1:]]
    self.assertListEqual(
        expected_full_length_inputs_batch,
        self.enc.get_inputs_batch(event_sequences, True))
    self.assertListEqual(
        expected_last_event_inputs_batch,
        self.enc.get_inputs_batch(event_sequences))

  def testExtendEventSequences(self):
    events1 = [0]
    events2 = [0]
    events3 = [0]
    event_sequences = [events1, events2, events3]
    softmax = [[[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]
    self.enc.extend_event_sequences(event_sequences, softmax)
    self.assertListEqual(list(events1), [0, 2])
    self.assertListEqual(list(events2), [0, 0])
    self.assertListEqual(list(events3), [0, 1])

  def testEvaluateLogLikelihood(self):
    events1 = [0, 1, 0]
    events2 = [1, 2, 2]
    event_sequences = [events1, events2]
    softmax = [[[0.0, 0.5, 0.5], [0.3, 0.4, 0.3]],
               [[0.0, 0.6, 0.4], [0.0, 0.4, 0.6]]]
    p = self.enc.evaluate_log_likelihood(event_sequences, softmax)
    self.assertListEqual([np.log(0.5) + np.log(0.3),
                          np.log(0.4) + np.log(0.6)], p)


class LookbackEventSequenceEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = encoder_decoder.LookbackEventSequenceEncoderDecoder(
        testing_lib.TrivialOneHotEncoding(3, num_steps=range(3)), [1, 2], 2)

  def testInputSize(self):
    self.assertEqual(13, self.enc.input_size)

  def testNumClasses(self):
    self.assertEqual(5, self.enc.num_classes)

  def testEventsToInput(self):
    events = [0, 1, 0, 2, 0]
    self.assertEqual([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                      1.0, -1.0, 0.0, 0.0],
                     self.enc.events_to_input(events, 0))
    self.assertEqual([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                      -1.0, 1.0, 0.0, 0.0],
                     self.enc.events_to_input(events, 1))
    self.assertEqual([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                      1.0, 1.0, 0.0, 1.0],
                     self.enc.events_to_input(events, 2))
    self.assertEqual([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                      -1.0, -1.0, 0.0, 0.0],
                     self.enc.events_to_input(events, 3))
    self.assertEqual([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                      1.0, -1.0, 0.0, 1.0],
                     self.enc.events_to_input(events, 4))

  def testEventsToLabel(self):
    events = [0, 1, 0, 2, 0]
    self.assertEqual(4, self.enc.events_to_label(events, 0))
    self.assertEqual(1, self.enc.events_to_label(events, 1))
    self.assertEqual(4, self.enc.events_to_label(events, 2))
    self.assertEqual(2, self.enc.events_to_label(events, 3))
    self.assertEqual(4, self.enc.events_to_label(events, 4))

  def testClassIndexToEvent(self):
    events = [0, 1, 0, 2, 0]
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:1]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:1]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:1]))
    self.assertEqual(0, self.enc.class_index_to_event(3, events[:1]))
    self.assertEqual(0, self.enc.class_index_to_event(4, events[:1]))
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:2]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:2]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:2]))
    self.assertEqual(1, self.enc.class_index_to_event(3, events[:2]))
    self.assertEqual(0, self.enc.class_index_to_event(4, events[:2]))
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:3]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:3]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:3]))
    self.assertEqual(0, self.enc.class_index_to_event(3, events[:3]))
    self.assertEqual(1, self.enc.class_index_to_event(4, events[:3]))
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:4]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:4]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:4]))
    self.assertEqual(2, self.enc.class_index_to_event(3, events[:4]))
    self.assertEqual(0, self.enc.class_index_to_event(4, events[:4]))
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:5]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:5]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:5]))
    self.assertEqual(0, self.enc.class_index_to_event(3, events[:5]))
    self.assertEqual(2, self.enc.class_index_to_event(4, events[:5]))

  def testLabelsToNumSteps(self):
    labels = [0, 1, 0, 2, 0]
    self.assertEqual(3, self.enc.labels_to_num_steps(labels))

    labels = [0, 1, 3, 2, 4]
    self.assertEqual(5, self.enc.labels_to_num_steps(labels))

  def testEmptyLookback(self):
    enc = encoder_decoder.LookbackEventSequenceEncoderDecoder(
        testing_lib.TrivialOneHotEncoding(3), [], 2)
    self.assertEqual(5, enc.input_size)
    self.assertEqual(3, enc.num_classes)

    events = [0, 1, 0, 2, 0]

    self.assertEqual([1.0, 0.0, 0.0, 1.0, -1.0],
                     enc.events_to_input(events, 0))
    self.assertEqual([0.0, 1.0, 0.0, -1.0, 1.0],
                     enc.events_to_input(events, 1))
    self.assertEqual([1.0, 0.0, 0.0, 1.0, 1.0],
                     enc.events_to_input(events, 2))
    self.assertEqual([0.0, 0.0, 1.0, -1.0, -1.0],
                     enc.events_to_input(events, 3))
    self.assertEqual([1.0, 0.0, 0.0, 1.0, -1.0],
                     enc.events_to_input(events, 4))

    self.assertEqual(0, enc.events_to_label(events, 0))
    self.assertEqual(1, enc.events_to_label(events, 1))
    self.assertEqual(0, enc.events_to_label(events, 2))
    self.assertEqual(2, enc.events_to_label(events, 3))
    self.assertEqual(0, enc.events_to_label(events, 4))

    self.assertEqual(0, self.enc.class_index_to_event(0, events[:1]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:1]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:1]))
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:2]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:2]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:2]))
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:3]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:3]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:3]))
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:4]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:4]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:4]))
    self.assertEqual(0, self.enc.class_index_to_event(0, events[:5]))
    self.assertEqual(1, self.enc.class_index_to_event(1, events[:5]))
    self.assertEqual(2, self.enc.class_index_to_event(2, events[:5]))


class ConditionalEventSequenceEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = encoder_decoder.ConditionalEventSequenceEncoderDecoder(
        encoder_decoder.OneHotEventSequenceEncoderDecoder(
            testing_lib.TrivialOneHotEncoding(2)),
        encoder_decoder.OneHotEventSequenceEncoderDecoder(
            testing_lib.TrivialOneHotEncoding(3)))

  def testInputSize(self):
    self.assertEquals(5, self.enc.input_size)

  def testNumClasses(self):
    self.assertEqual(3, self.enc.num_classes)

  def testEventsToInput(self):
    control_events = [1, 1, 1, 0, 0]
    target_events = [0, 1, 0, 2, 0]
    self.assertEqual(
        [0.0, 1.0, 1.0, 0.0, 0.0],
        self.enc.events_to_input(control_events, target_events, 0))
    self.assertEqual(
        [0.0, 1.0, 0.0, 1.0, 0.0],
        self.enc.events_to_input(control_events, target_events, 1))
    self.assertEqual(
        [1.0, 0.0, 1.0, 0.0, 0.0],
        self.enc.events_to_input(control_events, target_events, 2))
    self.assertEqual(
        [1.0, 0.0, 0.0, 0.0, 1.0],
        self.enc.events_to_input(control_events, target_events, 3))

  def testEventsToLabel(self):
    target_events = [0, 1, 0, 2, 0]
    self.assertEqual(0, self.enc.events_to_label(target_events, 0))
    self.assertEqual(1, self.enc.events_to_label(target_events, 1))
    self.assertEqual(0, self.enc.events_to_label(target_events, 2))
    self.assertEqual(2, self.enc.events_to_label(target_events, 3))
    self.assertEqual(0, self.enc.events_to_label(target_events, 4))

  def testClassIndexToEvent(self):
    target_events = [0, 1, 0, 2, 0]
    self.assertEqual(0, self.enc.class_index_to_event(0, target_events))
    self.assertEqual(1, self.enc.class_index_to_event(1, target_events))
    self.assertEqual(2, self.enc.class_index_to_event(2, target_events))

  def testEncode(self):
    control_events = [1, 1, 1, 0, 0]
    target_events = [0, 1, 0, 2, 0]
    sequence_example = self.enc.encode(control_events, target_events)
    expected_inputs = [[0.0, 1.0, 1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 1.0, 0.0],
                       [1.0, 0.0, 1.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0, 1.0]]
    expected_labels = [1, 0, 2, 0]
    expected_sequence_example = sequence_example_lib.make_sequence_example(
        expected_inputs, expected_labels)
    self.assertEqual(sequence_example, expected_sequence_example)

  def testGetInputsBatch(self):
    control_event_sequences = [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]]
    target_event_sequences = [[0, 1, 0, 2], [0, 1]]
    expected_inputs_1 = [[0.0, 1.0, 1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 1.0, 0.0],
                         [1.0, 0.0, 1.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0, 0.0, 1.0]]
    expected_inputs_2 = [[0.0, 1.0, 1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 1.0, 0.0]]
    expected_full_length_inputs_batch = [expected_inputs_1, expected_inputs_2]
    expected_last_event_inputs_batch = [expected_inputs_1[-1:],
                                        expected_inputs_2[-1:]]
    self.assertListEqual(
        expected_full_length_inputs_batch,
        self.enc.get_inputs_batch(
            control_event_sequences, target_event_sequences, True))
    self.assertListEqual(
        expected_last_event_inputs_batch,
        self.enc.get_inputs_batch(
            control_event_sequences, target_event_sequences))

  def testExtendEventSequences(self):
    target_events_1 = [0]
    target_events_2 = [0]
    target_events_3 = [0]
    target_event_sequences = [target_events_1, target_events_2, target_events_3]
    softmax = [[[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]
    self.enc.extend_event_sequences(target_event_sequences, softmax)
    self.assertListEqual(list(target_events_1), [0, 2])
    self.assertListEqual(list(target_events_2), [0, 0])
    self.assertListEqual(list(target_events_3), [0, 1])

  def testEvaluateLogLikelihood(self):
    target_events_1 = [0, 1, 0]
    target_events_2 = [1, 2, 2]
    target_event_sequences = [target_events_1, target_events_2]
    softmax = [[[0.0, 0.5, 0.5], [0.3, 0.4, 0.3]],
               [[0.0, 0.6, 0.4], [0.0, 0.4, 0.6]]]
    p = self.enc.evaluate_log_likelihood(target_event_sequences, softmax)
    self.assertListEqual([np.log(0.5) + np.log(0.3),
                          np.log(0.4) + np.log(0.6)], p)


class OptionalEventSequenceEncoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = encoder_decoder.OptionalEventSequenceEncoder(
        encoder_decoder.OneHotEventSequenceEncoderDecoder(
            testing_lib.TrivialOneHotEncoding(3)))

  def testInputSize(self):
    self.assertEquals(4, self.enc.input_size)

  def testEventsToInput(self):
    events = [(False, 0), (False, 1), (False, 0), (True, 2), (True, 0)]
    self.assertEqual(
        [0.0, 1.0, 0.0, 0.0],
        self.enc.events_to_input(events, 0))
    self.assertEqual(
        [0.0, 0.0, 1.0, 0.0],
        self.enc.events_to_input(events, 1))
    self.assertEqual(
        [0.0, 1.0, 0.0, 0.0],
        self.enc.events_to_input(events, 2))
    self.assertEqual(
        [1.0, 0.0, 0.0, 0.0],
        self.enc.events_to_input(events, 3))
    self.assertEqual(
        [1.0, 0.0, 0.0, 0.0],
        self.enc.events_to_input(events, 4))


class MultipleEventSequenceEncoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = encoder_decoder.MultipleEventSequenceEncoder([
        encoder_decoder.OneHotEventSequenceEncoderDecoder(
            testing_lib.TrivialOneHotEncoding(2)),
        encoder_decoder.OneHotEventSequenceEncoderDecoder(
            testing_lib.TrivialOneHotEncoding(3))])

  def testInputSize(self):
    self.assertEquals(5, self.enc.input_size)

  def testEventsToInput(self):
    events = [(1, 0), (1, 1), (1, 0), (0, 2), (0, 0)]
    self.assertEqual(
        [0.0, 1.0, 1.0, 0.0, 0.0],
        self.enc.events_to_input(events, 0))
    self.assertEqual(
        [0.0, 1.0, 0.0, 1.0, 0.0],
        self.enc.events_to_input(events, 1))
    self.assertEqual(
        [0.0, 1.0, 1.0, 0.0, 0.0],
        self.enc.events_to_input(events, 2))
    self.assertEqual(
        [1.0, 0.0, 0.0, 0.0, 1.0],
        self.enc.events_to_input(events, 3))
    self.assertEqual(
        [1.0, 0.0, 1.0, 0.0, 0.0],
        self.enc.events_to_input(events, 4))


class ModuloPerformanceEventSequenceEncoderTest(tf.test.TestCase):
  """Test class for ModuloPerformanceEventSequenceEncoder.

  ModuloPerformanceEventSequenceEncoderDecoder is tightly coupled with the
  PerformanceModuloEncoding class. As a result, in the test set up, the test
  object is initialized with a PerformanceModuloEncoding object, and tested
  accordingly. Since this class mainly modifies the input encoding of events
  and otherwise acts very similarly to the OneHotEventSequenceEncoderDecoder,
  the events_to_labels(), and class_index_to_event() methods of the class are
  not tested.
  """

  def setUp(self):
    self._num_velocity_bins = 32
    self._max_shift_steps = 32
    self.enc = encoder_decoder.ModuloPerformanceEventSequenceEncoderDecoder(
        PerformanceModuloEncoding(
            num_velocity_bins=self._num_velocity_bins,
            max_shift_steps=self._max_shift_steps))

    self._pitch_encoder_width = 5
    self._velocity_encoder_width = 3
    self._time_shift_encoder_width = 3
    self._expected_input_size = (2 * self._pitch_encoder_width +
                                 self._velocity_encoder_width +
                                 self._time_shift_encoder_width)

    self._expected_num_classes = (self._num_velocity_bins +
                                  self._max_shift_steps +
                                  2 * (performance_lib.MAX_MIDI_PITCH -
                                       performance_lib.MIN_MIDI_PITCH + 1))

  def testInputSize(self):
    self.assertEquals(self._expected_input_size, self.enc.input_size)

  def testNumClasses(self):
    self.assertEqual(self._expected_num_classes, self.enc.num_classes)

  def testDefaultEventLabel(self):
    label = self._expected_num_classes - self._num_velocity_bins - 1
    self.assertEquals(label, self.enc.default_event_label)

  def testEventToNumSteps(self):
    self.assertEquals(self.enc.event_to_num_steps(1), 1)

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


class ModuloEventSequenceEncoderTest(tf.test.TestCase):
  """Test class for ModuloEventSequenceEncoder.

  The ModuloEventSequenceEncoderDecoder is expected to be used with the
  MelodyOneHotEncoding object. As such, in the test set up, the test object is
  initialized with a MelodyOneHotEncoding object, and tested accordingly. Since
  this class mainly modifies the input encoding of events
  and otherwise acts very similarly to the OneHotEventSequenceEncoderDecoder,
  the events_to_labels(), and class_index_to_event() methods of this class are
  not tested.
  """

  def setUp(self):
    self._min_note = 1
    self._max_note = 60  # max_note is excluded
    self.enc = encoder_decoder.ModuloEventSequenceEncoderDecoder(
        MelodyOneHotEncoding(self._min_note, self._max_note))
    self._expected_input_size = constants.NUM_SPECIAL_MELODY_EVENTS + 4
    self._expected_num_classes = self._max_note - self._min_note + 2

  def testInputSize(self):
    self.assertEquals(self._expected_input_size, self.enc.input_size)

  def testNumClasses(self):
    self.assertEqual(self._expected_num_classes, self.enc.num_classes)

  def testDefaultEventLabel(self):
    self.assertAllEqual(0, self.enc.default_event_label)

  def testLabelsToNumSteps(self):
    labels = [0, 1, 0, 2, 0]
    self.assertEqual(5, self.enc.labels_to_num_steps(labels))

  def testEventsToInput(self):
    # Create a list containing all possible events ordered by event number,
    # starting with no-event=-2, note-off=-1, and first-note=self._min_note.
    events = (range(self._expected_num_classes) -
              np.ones(self._expected_num_classes) * 2)
    events[2:] += self._min_note

    slow_base = 2.0 * pi / 144.0
    fast_base = 2.0 * pi / 12.0

    # A list of pairs of positions in the above events list, and the expected
    # encoding for the event in that position. The list include the special
    # events, the first 12 notes, and a sample of notes in higher octaves.
    expected_pairs = [
        (0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # no-event
        (1, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # note-off
        (2, [0.0, 0.0, cos(0.0 * slow_base), sin(0.0 * slow_base),  # min-note
             cos(0.0 * fast_base), sin(0.0 * fast_base)]),
        (3, [0.0, 0.0, cos(1.0 * slow_base), sin(1.0 * slow_base),
             cos(1.0 * fast_base), sin(1.0 * fast_base)]),
        (4, [0.0, 0.0, cos(2.0 * slow_base), sin(2.0 * slow_base),
             cos(2.0 * fast_base), sin(2.0 * fast_base)]),
        (5, [0.0, 0.0, cos(3.0 * slow_base), sin(3.0 * slow_base),
             cos(3.0 * fast_base), sin(3.0 * fast_base)]),
        (6, [0.0, 0.0, cos(4.0 * slow_base), sin(4.0 * slow_base),
             cos(4.0 * fast_base), sin(4.0 * fast_base)]),
        (7, [0.0, 0.0, cos(5.0 * slow_base), sin(5.0 * slow_base),
             cos(5.0 * fast_base), sin(5.0 * fast_base)]),
        (8, [0.0, 0.0, cos(6.0 * slow_base), sin(6.0 * slow_base),
             cos(6.0 * fast_base), sin(6.0 * fast_base)]),
        (9, [0.0, 0.0, cos(7.0 * slow_base), sin(7.0 * slow_base),
             cos(7.0 * fast_base), sin(7.0 * fast_base)]),
        (10, [0.0, 0.0, cos(8.0 * slow_base), sin(8.0 * slow_base),
              cos(8.0 * fast_base), sin(8.0 * fast_base)]),
        (11, [0.0, 0.0, cos(9.0 * slow_base), sin(9.0 * slow_base),
              cos(9.0 * fast_base), sin(9.0 * fast_base)]),
        (12, [0.0, 0.0, cos(10.0 * slow_base), sin(10.0 * slow_base),
              cos(10.0 * fast_base), sin(10.0 * fast_base)]),
        (13, [0.0, 0.0, cos(11.0 * slow_base), sin(11.0 * slow_base),
              cos(11.0 * fast_base), sin(11.0 * fast_base)]),  # min_note+11
        (19, [0.0, 0.0, cos(17.0 * slow_base), sin(17.0 * slow_base),
              cos(17.0 * fast_base), sin(17.0 * fast_base)]),
        (30, [0.0, 0.0, cos(28.0 * slow_base), sin(28.0 * slow_base),
              cos(28.0 * fast_base), sin(28.0 * fast_base)]),
        (41, [0.0, 0.0, cos(39.0 * slow_base), sin(39.0 * slow_base),
              cos(39.0 * fast_base), sin(39.0 * fast_base)]),
        (60, [0.0, 0.0, cos(58.0 * slow_base), sin(58.0 * slow_base),
              cos(58.0 * fast_base), sin(58.0 * fast_base)])]

    for position, expected_encoded_modulo_event in expected_pairs:
      actual_encoded_modulo_event = self.enc.events_to_input(events, position)
      for i in range(self._expected_input_size):
        self.assertAlmostEqual(expected_encoded_modulo_event[i],
                               actual_encoded_modulo_event[i])


if __name__ == '__main__':
  tf.test.main()
