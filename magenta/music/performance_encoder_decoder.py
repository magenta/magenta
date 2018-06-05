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

# internal imports
import numpy as np

from magenta.music import encoder_decoder
from magenta.music import performance_lib
from magenta.music.encoder_decoder import EventSequenceEncoderDecoder
from magenta.music.performance_lib import PerformanceEvent


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


class PerformanceModuloEncoding(object):
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
    self._num_velocity_bins = num_velocity_bins

    # Create a lookup table for modulo-12 encoding of pitch classes.
    # Possible values for semitone_steps are 1 and 7. A value of 1 corresponds
    # to placing notes consecutively on the unit circle. A value of 7
    # corresponds to following each note with one that is 7 semitones above it.
    # semitone_steps = 1 seems to produce better results, and is the recommended
    # value. Moreover, unit tests are provided only for semitone_steps = 1. If
    # in the future you plan to enable support for semitone_steps = 7, then
    # please make semitone_steps a parameter of this method, and add unit tests
    # for it.
    semitone_steps = 1
    self._pitch_class_table = np.zeros((12, 2))
    for i in range(12):
      row = (i * semitone_steps) % 12
      angle = (float(row) * math.pi) / 6.0
      self._pitch_class_table[row] = [math.cos(angle), math.sin(angle)]

    # Create a lookup table for modulo-144 encoding of notes. Encode each note
    # on a unit circle of 144 notes, spanning 12 octaves. Since there are only
    # 128 midi notes, the last 16 positions on the unit circle will not be used.
    self._note_table = np.zeros((144, 2))
    for i in range(144):
      angle = (float(i) * math.pi) / 72.0
      self._note_table[i] = [math.cos(angle), math.sin(angle)]

    # Create a lookup table for modulo-bins encoding of time_shifts.
    self._time_shift_table = np.zeros((max_shift_steps, 2))
    for i in range(max_shift_steps):
      angle = (float(i) * 2.0 * math.pi) / float(max_shift_steps)
      self._time_shift_table[i] = [math.cos(angle), math.sin(angle)]

    # Create a lookup table for modulo-bins encoding of velocities.
    if num_velocity_bins > 0:
      self._velocity_table = np.zeros((num_velocity_bins, 2))
      for i in range(num_velocity_bins):
        angle = (float(i) * 2.0 * math.pi) / float(num_velocity_bins)
        self._velocity_table[i] = [math.cos(angle), math.sin(angle)]

  @property
  def input_size(self):
    total = 0
    for _, _, _, encoder_width in self._event_ranges:
      total += encoder_width
    return total

  def encode_modulo_event(self, event):
    offset = 0
    for event_type, min_value, _, encoder_width in self._event_ranges:
      if event.event_type == event_type:
        value = event.event_value - min_value
        return offset, event_type, value
      offset += encoder_width

    raise ValueError('Unknown event type: %s' % event.event_type)

  def embed_pitch_class(self, value):
    if value < 0 or value >= 12:
      raise ValueError('Unexpected pitch class value: %s' % value)
    return self._pitch_class_table[value]

  def embed_note(self, value):
    if value < 0 or value >= 144:
      raise ValueError('Unexpected note value: %s' % value)
    return self._note_table[value]

  def embed_time_shift(self, value):
    if value < 0 or value >= self._max_shift_steps:
      raise ValueError('Unexpected time shift value: %s' % value)
    return self._time_shift_table[value]

  def embed_velocity(self, value):
    if value < 0 or value >= self._num_velocity_bins:
      raise ValueError('Unexpected velocity value: %s' % value)
    return self._velocity_table[value]


class ModuloPerformanceEventSequenceEncoderDecoder(EventSequenceEncoderDecoder):
  """An EventSequenceEncoderDecoder for modulo encoding performance events.

  ModuloPerformanceEventSequenceEncoderDecoder is an EventSequenceEncoderDecoder
  that uses modulo/circular encoding for encoding performance input events, and
  otherwise uses one hot encoding for encoding and decoding of labels.
  """

  def __init__(self, num_velocity_bins=0,
               max_shift_steps=performance_lib.DEFAULT_MAX_SHIFT_STEPS):
    """Initialize a ModuloPerformanceEventSequenceEncoderDecoder object.

    Args:
      num_velocity_bins: Number of velocity bins.
      max_shift_steps: Maximum number of shift steps supported.
    """

    self._modulo_encoding = PerformanceModuloEncoding(
        num_velocity_bins=num_velocity_bins, max_shift_steps=max_shift_steps)
    self._one_hot_encoding = PerformanceOneHotEncoding(
        num_velocity_bins=num_velocity_bins, max_shift_steps=max_shift_steps)

  @property
  def input_size(self):
    return self._modulo_encoding.input_size

  @property
  def num_classes(self):
    return self._one_hot_encoding.num_classes

  @property
  def default_event_label(self):
    return self._one_hot_encoding.encode_event(
        self._one_hot_encoding.default_event)

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the event sequence.

    Returns a modulo/circular encoding for the given position in the performance
      event sequence.

    Args:
      events: A list-like sequence of events.
      position: An integer event position in the event sequence.

    Returns:
      An input vector, a list of floats.
    """
    input_ = [0.0] * self.input_size
    offset, event_type, value = (self._modulo_encoding
                                 .encode_modulo_event(events[position]))
    input_[offset] = 1.0  # valid bit for the event
    offset += 1
    if (event_type == performance_lib.PerformanceEvent.NOTE_ON or
        event_type == performance_lib.PerformanceEvent.NOTE_OFF):

      # Encode the note on a circle of 144 notes, covering 12 octaves.
      cosine_sine_pair = self._modulo_encoding.embed_note(value)
      input_[offset] = cosine_sine_pair[0]
      input_[offset + 1] = cosine_sine_pair[1]
      offset += 2

      # Encode the note's pitch class, using the encoder's lookup table.
      value %= 12
      cosine_sine_pair = self._modulo_encoding.embed_pitch_class(value)
      input_[offset] = cosine_sine_pair[0]
      input_[offset + 1] = cosine_sine_pair[1]
    else:
      # This must be a velocity, or a time-shift event. Encode it using
      # modulo-bins embedding.
      if event_type == performance_lib.PerformanceEvent.TIME_SHIFT:
        cosine_sine_pair = self._modulo_encoding.embed_time_shift(value)
      else:
        cosine_sine_pair = self._modulo_encoding.embed_velocity(value)
      input_[offset] = cosine_sine_pair[0]
      input_[offset + 1] = cosine_sine_pair[1]
    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the event sequence.

    Returns the zero-based index value for the given position in the event
    sequence, as determined by the one hot encoding.

    Args:
      events: A list-like sequence of events.
      position: An integer event position in the event sequence.

    Returns:
      A label, an integer.
    """
    return self._one_hot_encoding.encode_event(events[position])

  def class_index_to_event(self, class_index, events):
    """Returns the event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A list-like sequence of events. This object is not used in this
          implementation.

    Returns:
      An event value.
    """
    return self._one_hot_encoding.decode_event(class_index)

  def labels_to_num_steps(self, labels):
    """Returns the total number of time steps for a sequence of class labels.

    Args:
      labels: A list-like sequence of integers in the range
          [0, self.num_classes).

    Returns:
      The total number of time steps for the label sequence, as determined by
      the one-hot encoding.
    """
    events = []
    for label in labels:
      events.append(self.class_index_to_event(label, events))
    return sum(self._one_hot_encoding.event_to_num_steps(event)
               for event in events)


class PerformanceOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for performance events."""

  def __init__(self, num_velocity_bins=0,
               max_shift_steps=performance_lib.DEFAULT_MAX_SHIFT_STEPS,
               min_pitch=performance_lib.MIN_MIDI_PITCH,
               max_pitch=performance_lib.MAX_MIDI_PITCH):
    self._event_ranges = [
        (PerformanceEvent.NOTE_ON, min_pitch, max_pitch),
        (PerformanceEvent.NOTE_OFF, min_pitch, max_pitch),
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
