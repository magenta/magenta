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

from magenta.music import constants
from magenta.music import encoder_decoder
from magenta.music import performance_lib
from magenta.music.performance_lib import PerformanceEvent

NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE


# Value ranges for event types, as (event_type, min_value, max_value) tuples.
EVENT_RANGES = [
    (PerformanceEvent.NOTE_ON,
     performance_lib.MIN_MIDI_PITCH, performance_lib.MAX_MIDI_PITCH),
    (PerformanceEvent.NOTE_OFF,
     performance_lib.MIN_MIDI_PITCH, performance_lib.MAX_MIDI_PITCH),
]


class PerformanceOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for performance events."""

  def __init__(self, num_velocity_bins=0,
               max_shift_steps=performance_lib.DEFAULT_MAX_SHIFT_STEPS):
    self._event_ranges = EVENT_RANGES + [
        (PerformanceEvent.TIME_SHIFT, 1, max_shift_steps)
    ]
    if num_velocity_bins > 0:
      self._event_ranges.append(
          (PerformanceEvent.VELOCITY, 1, num_velocity_bins))
    self._max_shift_steps = max_shift_steps

  @property
  def num_classes(self):
    return sum(max_value - min_value + 1
               for event_type, min_value, max_value in self._event_ranges)

  @property
  def default_event(self):
    return PerformanceEvent(
        event_type=PerformanceEvent.TIME_SHIFT,
        event_value=self._max_shift_steps)

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

  def event_to_num_steps(self, event):
    if event.event_type == PerformanceEvent.TIME_SHIFT:
      return event.event_value
    else:
      return 0


class NoteDensityOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for performance note density events.

  Encodes by quantizing note density events. When decoding, always decodes to
  the minimum value for each bin. The first bin starts at zero note density.
  """

  def __init__(self, density_bin_ranges):
    """Initialize a NoteDensityOneHotEncoding.

    Args:
      density_bin_ranges: List of note density (notes per second) bin boundaries
          to use when quantizing. The number of bins will be one larger than the
          list length.
    """
    self._density_bin_ranges = density_bin_ranges

  @property
  def num_classes(self):
    return len(self._density_bin_ranges) + 1

  @property
  def default_event(self):
    return 0.0

  def encode_event(self, event):
    for idx, density in enumerate(self._density_bin_ranges):
      if event < density:
        return idx
    return len(self._density_bin_ranges)

  def decode_event(self, index):
    if index == 0:
      return 0.0
    else:
      return self._density_bin_ranges[index - 1]


class PitchHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
  """An encoder for pitch class histogram sequences.

  This class has no label encoding and is only a trivial input encoder that
  merely uses each histogram as the input vector.
  """

  @property
  def input_size(self):
    return NOTES_PER_OCTAVE

  @property
  def num_classes(self):
    raise NotImplementedError

  @property
  def default_event_label(self):
    raise NotImplementedError

  def events_to_input(self, events, position):
    return events[position]

  def events_to_label(self, events, position):
    raise NotImplementedError

  def class_index_to_event(self, class_index, events):
    raise NotImplementedError
