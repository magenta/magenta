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

import numpy as np
import tensorflow as tf

from magenta.common import sequence_example_lib
from magenta.music import encoder_decoder
from magenta.music import testing_lib


class OneHotEventSequenceEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = encoder_decoder.OneHotEventSequenceEncoderDecoder(
        testing_lib.TrivialOneHotEncoding(3, num_steps=range(3)))

  def testInputSize(self):
    self.assertEqual(3, self.enc.input_size)

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


class OneHotIndexEventSequenceEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = encoder_decoder.OneHotIndexEventSequenceEncoderDecoder(
        testing_lib.TrivialOneHotEncoding(3, num_steps=range(3)))

  def testInputSize(self):
    self.assertEqual(1, self.enc.input_size)

  def testInputDepth(self):
    self.assertEqual(3, self.enc.input_depth)

  def testEventsToInput(self):
    events = [0, 1, 0, 2, 0]
    self.assertEqual([0], self.enc.events_to_input(events, 0))
    self.assertEqual([1], self.enc.events_to_input(events, 1))
    self.assertEqual([0], self.enc.events_to_input(events, 2))
    self.assertEqual([2], self.enc.events_to_input(events, 3))
    self.assertEqual([0], self.enc.events_to_input(events, 4))

  def testEncode(self):
    events = [0, 1, 0, 2, 0]
    sequence_example = self.enc.encode(events)
    expected_inputs = [[0], [1], [0], [2]]
    expected_labels = [1, 0, 2, 0]
    expected_sequence_example = sequence_example_lib.make_sequence_example(
        expected_inputs, expected_labels)
    self.assertEqual(sequence_example, expected_sequence_example)

  def testGetInputsBatch(self):
    event_sequences = [[0, 1, 0, 2, 0], [0, 1, 2]]
    expected_inputs_1 = [[0], [1], [0], [2], [0]]
    expected_inputs_2 = [[0], [1], [2]]
    expected_full_length_inputs_batch = [expected_inputs_1, expected_inputs_2]
    expected_last_event_inputs_batch = [expected_inputs_1[-1:],
                                        expected_inputs_2[-1:]]
    self.assertListEqual(
        expected_full_length_inputs_batch,
        self.enc.get_inputs_batch(event_sequences, True))
    self.assertListEqual(
        expected_last_event_inputs_batch,
        self.enc.get_inputs_batch(event_sequences))


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
    self.assertEqual(5, self.enc.input_size)

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
    softmax = np.array(
        [[[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
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
    self.assertEqual(4, self.enc.input_size)

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
    self.assertEqual(5, self.enc.input_size)

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


if __name__ == '__main__':
  tf.test.main()
