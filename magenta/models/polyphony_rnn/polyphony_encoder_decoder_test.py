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

"""Tests for polyphony_encoder_decoder."""

from magenta.models.polyphony_rnn import polyphony_encoder_decoder
from magenta.models.polyphony_rnn.polyphony_lib import PolyphonicEvent
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class PolyphonyOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = polyphony_encoder_decoder.PolyphonyOneHotEncoding()

  def testEncodeDecode(self):
    start = PolyphonicEvent(
        event_type=PolyphonicEvent.START, pitch=0)
    step_end = PolyphonicEvent(
        event_type=PolyphonicEvent.STEP_END, pitch=0)
    new_note = PolyphonicEvent(
        event_type=PolyphonicEvent.NEW_NOTE, pitch=0)
    continued_note = PolyphonicEvent(
        event_type=PolyphonicEvent.CONTINUED_NOTE, pitch=60)
    continued_max_note = PolyphonicEvent(
        event_type=PolyphonicEvent.CONTINUED_NOTE, pitch=127)

    index = self.enc.encode_event(start)
    self.assertEqual(0, index)
    event = self.enc.decode_event(index)
    self.assertEqual(start, event)

    index = self.enc.encode_event(step_end)
    self.assertEqual(2, index)
    event = self.enc.decode_event(index)
    self.assertEqual(step_end, event)

    index = self.enc.encode_event(new_note)
    self.assertEqual(3, index)
    event = self.enc.decode_event(index)
    self.assertEqual(new_note, event)

    index = self.enc.encode_event(continued_note)
    self.assertEqual(191, index)
    event = self.enc.decode_event(index)
    self.assertEqual(continued_note, event)

    index = self.enc.encode_event(continued_max_note)
    self.assertEqual(258, index)
    event = self.enc.decode_event(index)
    self.assertEqual(continued_max_note, event)

  def testEventToNumSteps(self):
    self.assertEqual(0, self.enc.event_to_num_steps(
        PolyphonicEvent(event_type=PolyphonicEvent.START, pitch=0)))
    self.assertEqual(0, self.enc.event_to_num_steps(
        PolyphonicEvent(event_type=PolyphonicEvent.END, pitch=0)))
    self.assertEqual(1, self.enc.event_to_num_steps(
        PolyphonicEvent(event_type=PolyphonicEvent.STEP_END, pitch=0)))
    self.assertEqual(0, self.enc.event_to_num_steps(
        PolyphonicEvent(event_type=PolyphonicEvent.NEW_NOTE, pitch=60)))
    self.assertEqual(0, self.enc.event_to_num_steps(
        PolyphonicEvent(event_type=PolyphonicEvent.CONTINUED_NOTE, pitch=72)))


if __name__ == '__main__':
  tf.test.main()
