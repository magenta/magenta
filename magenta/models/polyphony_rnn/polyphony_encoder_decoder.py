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
"""Classes for converting between polyphonic input and model input/output."""

from __future__ import division

# internal imports

from magenta.models.polyphony_rnn import polyphony_lib
from magenta.models.polyphony_rnn.polyphony_lib import PolyphonicEvent
from magenta.music import encoder_decoder

EVENT_CLASSES_WITHOUT_PITCH = [
    PolyphonicEvent.START,
    PolyphonicEvent.END,
    PolyphonicEvent.STEP_END,
]

EVENT_CLASSES_WITH_PITCH = [
    PolyphonicEvent.NEW_NOTE,
    PolyphonicEvent.CONTINUED_NOTE,
]

PITCH_CLASSES = polyphony_lib.MAX_MIDI_PITCH + 1


class PolyphonyOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for polyphonic events."""

  @property
  def num_classes(self):
    return len(EVENT_CLASSES_WITHOUT_PITCH) + (
        len(EVENT_CLASSES_WITH_PITCH) * PITCH_CLASSES)

  @property
  def default_event(self):
    return PolyphonicEvent(
        event_type=PolyphonicEvent.STEP_END, pitch=0)

  def encode_event(self, event):
    if event.event_type in EVENT_CLASSES_WITHOUT_PITCH:
      return EVENT_CLASSES_WITHOUT_PITCH.index(event.event_type)
    elif event.event_type in EVENT_CLASSES_WITH_PITCH:
      return len(EVENT_CLASSES_WITHOUT_PITCH) + (
          EVENT_CLASSES_WITH_PITCH.index(event.event_type) * PITCH_CLASSES +
          event.pitch)
    else:
      raise ValueError('Unknown event type: %s' % event.event_type)

  def decode_event(self, index):
    if index < len(EVENT_CLASSES_WITHOUT_PITCH):
      return PolyphonicEvent(
          event_type=EVENT_CLASSES_WITHOUT_PITCH[index], pitch=0)

    pitched_index = index - len(EVENT_CLASSES_WITHOUT_PITCH)
    if pitched_index < len(EVENT_CLASSES_WITH_PITCH) * PITCH_CLASSES:
      event_type = len(EVENT_CLASSES_WITHOUT_PITCH) + (
          pitched_index // PITCH_CLASSES)
      pitch = pitched_index % PITCH_CLASSES
      return PolyphonicEvent(
          event_type=event_type, pitch=pitch)

    raise ValueError('Unknown event index: %s' % index)

  def event_to_num_steps(self, event):
    if event.event_type == PolyphonicEvent.STEP_END:
      return 1
    else:
      return 0
