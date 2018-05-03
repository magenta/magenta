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
from numpy import zeros

from magenta.music import encoder_decoder
from magenta.music import performance_lib
from magenta.music.encoder_decoder import EventSequenceEncoderDecoder
from magenta.music.performance_lib import PerformanceEvent


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
  def input_size(self):
    total = 0
    for _, _, _, encoder_width in self._event_ranges:
      total += encoder_width
    return total

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


class ModuloPerformanceEventSequenceEncoderDecoder(EventSequenceEncoderDecoder):
  """An EventSequenceEncoderDecoder for modulo encoding performance events.

  ModuloPerformanceEventSequenceEncoderDecoder is an EventSequenceEncoderDecoder
  that uses Modulo Encoding for individual events. This encoding can be used in
  place of OneHotEventSequenceEncoderDecoder in models that use time-shift
  and/or velocity events in addition to note-on/note-off events. The input
  vectors are modulo-12 encodings of the most recent event. The output labels
  are one-hot encodings of the next event.

  The only method of this class that is different from that of
  OneHotEventSequenceEncoderDecoder is events_to_inputs().
  """

  def __init__(self, modulo_encoding, one_hot_encoding):
    """Initialize a ModuloPerformanceEventSequenceEncoderDecoder object.

    Args:
      modulo_encoding: A ModuloEncoding object that encodes input events using
          modulo encoding.
      one_hot_encoding: A OneHotEncoding object that encodes and decodes event
          labels using one-hot encoding.
    """

    # TODO(vidavakil): ideally, we would want to make sure that the two encoders
    # have compatible event_ranges, since at generation time we would be feeding
    # the output of the model back to its input. However, to do that we would
    # need access to private variables of the two objects (e.g., their
    # _event_ranges).
    self._modulo_encoding = modulo_encoding
    self._one_hot_encoding = one_hot_encoding

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

    Returns a modulo encoding for the given position in the performance event
      sequence, by modulo-encoding the output of the one-hot-encoder applied to
      that event position.

    Args:
      events: A list-like sequence of events.
      position: An integer event position in the event sequence.

    Returns:
      An input vector, a list of floats.
    """
    input_ = [0.0] * self.input_size
    offset, _, event_type, value, bins = (self._modulo_encoding
                                          .encode_modulo_event(
                                              events[position]))
    input_[offset] = 1.0  # valid bit for the event
    offset += 1
    angle = 0.0
    if (event_type == performance_lib.PerformanceEvent.NOTE_ON or
        event_type == performance_lib.PerformanceEvent.NOTE_OFF):

      # Encode the note on a circle.
      angle = (float(value) * math.pi) / 72.0  # 12 octaves, up to 144 notes
      input_[offset] = math.cos(angle)
      input_[offset + 1] = math.sin(angle)
      offset += 2

      # Encode the note's pitch class, using the encoder's lookup table.
      value %= 12
      cosine = self._modulo_encoding.embed_note(value)[0]
      sine = self._modulo_encoding.embed_note(value)[1]
      input_[offset] = cosine
      input_[offset + 1] = sine
    else:
      # This must be a velocity, or a time-shift event. Compute its
      # modulo-bins embedding.
      angle = (float(value) * 2.0 * math.pi) / float(bins)
      input_[offset] = math.cos(angle)
      input_[offset + 1] = math.sin(angle)
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

  def event_to_num_steps(self, unused_event):
    """Returns the number of time steps corresponding to an event value.

    This is used for normalization when computing metrics. Subclasses with
    variable step size should override this method.

    Args:
      unused_event: An event value for which to return the number of steps.

    Returns:
      The number of steps corresponding to the given event value, defaulting to
      one.
    """
    return 1


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
