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
"""Classes for converting between performance input and model input/output."""

from __future__ import division

# internal imports

from magenta.models.performance_rnn import performance_lib
from magenta.models.performance_rnn.performance_lib import PerformanceEvent
from magenta.music import encoder_decoder


# Value ranges for event types, as (event_type, min_value, max_value) tuples.
EVENT_RANGES = [
    (PerformanceEvent.NOTE_ON,
     performance_lib.MIN_MIDI_PITCH, performance_lib.MAX_MIDI_PITCH),
    (PerformanceEvent.NOTE_OFF,
     performance_lib.MIN_MIDI_PITCH, performance_lib.MAX_MIDI_PITCH),
    (PerformanceEvent.TIME_SHIFT, 1, performance_lib.MAX_SHIFT_STEPS),
]


class PerformanceOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for performance events."""

  def __init__(self, num_velocity_bins=0):
    if num_velocity_bins > 0:
      self._event_ranges = EVENT_RANGES + [
          (PerformanceEvent.VELOCITY, 1, num_velocity_bins)]
    else:
      self._event_ranges = EVENT_RANGES

  @property
  def num_classes(self):
    return sum(max_value - min_value + 1
               for event_type, min_value, max_value in self._event_ranges)

  @property
  def default_event(self):
    return PerformanceEvent(
        event_type=PerformanceEvent.TIME_SHIFT,
        event_value=performance_lib.MAX_SHIFT_STEPS)

  def encode_event(self, event):
    offset = 0
    for event_type, min_value, max_value in self._event_ranges:
      if event.event_type == event_type:
        return offset + event.event_value - min_value
      offset += max_value - min_value + 1

    raise ValueError('Unknown event type: %s' % event.event_type)

  def decode_event(self, index):
    offset = 0
    for event_type, min_value, max_value in self._event_ranges:
      if offset <= index <= offset + max_value - min_value:
        return PerformanceEvent(
            event_type=event_type, event_value=min_value + index - offset)
      offset += max_value - min_value + 1

    raise ValueError('Unknown event index: %s' % index)
