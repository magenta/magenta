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

# internal imports

import tensorflow as tf

from magenta.music import performance_encoder_decoder
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


if __name__ == '__main__':
  tf.test.main()
