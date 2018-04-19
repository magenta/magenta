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

import math
from numpy import zeros

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


# Number of floats used to encode NOTE_ON and NOTE_OFF events, using modulo-12
# encoding. 5 floats for: valid, octave_cos, octave_sin, note_cos, note_sin.
MODULO_PITCH_ENCODER_WIDTH = 5

# Number of floats used to encode TIME_SHIFT and VELOCITY events using
# module-bins encoding. 3 floats for: valid, event_cos, event_sin.
MODULO_VELOCITY_ENCODER_WIDTH = 3
MODULO_TIME_SHIFT_ENCODER_WIDTH = 3

MODULO_EVENT_RANGES = [
    (PerformanceEvent.NOTE_ON, performance_lib.MIN_MIDI_PITCH,
     performance_lib.MAX_MIDI_PITCH, MODULO_PITCH_ENCODER_WIDTH),
    (PerformanceEvent.NOTE_OFF, performance_lib.MIN_MIDI_PITCH,
     performance_lib.MAX_MIDI_PITCH, MODULO_PITCH_ENCODER_WIDTH),
]


class PerformanceModuloEncoding(encoder_decoder.OneHotEncoding):
  """Modulo encoding for performance events."""

  def __init__(self, num_velocity_bins=0,
               max_shift_steps=performance_lib.DEFAULT_MAX_SHIFT_STEPS):
    """Initiaizer for PerformanceModuloEncoding.

    Args:
      num_velocity_bins: Number of velocity bins.
      max_shift_steps: Maximum number of shift steps supported.
    """

    self._event_ranges = MODULO_EVENT_RANGES + [
        (PerformanceEvent.TIME_SHIFT, 1, max_shift_steps,
         MODULO_TIME_SHIFT_ENCODER_WIDTH)
    ]
    if num_velocity_bins > 0:
      self._event_ranges.append(
          (PerformanceEvent.VELOCITY, 1, num_velocity_bins,
           MODULO_VELOCITY_ENCODER_WIDTH))
    self._max_shift_steps = max_shift_steps

    # Create a lookup table for modulo-12 encoding of notes.
    # Possible values for semitone_steps are 1 and 7. A value of 1 corresponds
    # to placing notes consecutively on the unit circle. A value of 7
    # corresponds to following each note with one that is 7 semitones above it.
    # semitone_steps = 1 seems to produce better results, and is the recommended
    # value. Moreover, unit tests are provided only for semitone_steps = 1. If
    # in the future you plan to enable support for semitone_steps = 7, then
    # please make semitone_steps a parameter of this method, and add unit tests
    # for it.
    semitone_steps = 1
    self._table = zeros((12, 2))
    for i in range(12):
      row = (i * semitone_steps) % 12
      angle = (float(row) * math.pi) / 6.0
      self._table[row] = [math.cos(angle), math.sin(angle)]

  @property
  def num_classes(self):
    return sum(max_value - min_value + 1
               for _, min_value, max_value, _ in self._event_ranges)

  @property
  def input_size(self):
    total = 0
    for _, _, _, encoder_width in self._event_ranges:
      total += encoder_width
    return total

  @property
  def default_event(self):
    return PerformanceEvent(
        event_type=PerformanceEvent.TIME_SHIFT,
        event_value=self._max_shift_steps)

  def encode_modulo_event(self, event):
    offset = 0
    for event_type, min_value, max_value, encoder_width in self._event_ranges:
      if event.event_type == event_type:
        value = event.event_value - min_value
        bins = max_value - min_value + 1
        return offset, encoder_width, event_type, value, bins
      offset += encoder_width

    raise ValueError('Unknown event type: %s' % event.event_type)

  def embed_note(self, value):
    if value < 0 or value > 11:
      raise ValueError('Unexpected note class number: %s' % value)
    return self._table[value]

  def encode_event(self, event):
    offset = 0
    for event_type, min_value, max_value, _ in self._event_ranges:
      if event.event_type == event_type:
        return offset + event.event_value - min_value
      offset += max_value - min_value + 1

    raise ValueError('Unknown event type: %s' % event.event_type)

  def decode_event(self, index):
    offset = 0
    for event_type, min_value, max_value, _ in self._event_ranges:
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
