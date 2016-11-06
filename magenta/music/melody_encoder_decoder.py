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
"""Classes for converting between Melody objects and models inputs/outputs.

MelodyOneHotEncoding is an encoder_decoder.OneHotEncoding that specifies a one-
hot encoding for Melody events, i.e. MIDI pitch values plus note-off and no-
event.

KeyMelodyEncoderDecoder is an encoder_decoder.EventSequenceEncoderDecoder that
specifies an encoding of Melody objects into input vectors and output labels for
use by melody models.
"""

import collections

# internal imports
from magenta.music import constants
from magenta.music import encoder_decoder
from magenta.music import melodies_lib

NUM_SPECIAL_MELODY_EVENTS = constants.NUM_SPECIAL_MELODY_EVENTS
MELODY_NOTE_OFF = constants.MELODY_NOTE_OFF
MELODY_NO_EVENT = constants.MELODY_NO_EVENT
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH
MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR

DEFAULT_LOOKBACK_DISTANCES = encoder_decoder.DEFAULT_LOOKBACK_DISTANCES


class MelodyOneHotEncoding(encoder_decoder.OneHotEncoding):
  """Basic one hot encoding for melody events.

  Encodes melody events as follows:
    0 = no event,
    1 = note-off event,
    [2, self.num_classes) = note-on event for that pitch relative to the
        [self._min_note, self._max_note) range.
  """

  def __init__(self, min_note, max_note):
    """Initializes a MelodyOneHotEncoding object.

    Args:
      min_note: The minimum midi pitch the encoded melody events can have.
      max_note: The maximum midi pitch (exclusive) the encoded melody events
          can have.

    Raises:
      ValueError: If `min_note` or `max_note` are outside the midi range, or if
          `max_note` is not greater than `min_note`.
    """
    if min_note < MIN_MIDI_PITCH:
      raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > MAX_MIDI_PITCH + 1:
      raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note <= min_note:
      raise ValueError('max_note must be greater than min_note')

    self._min_note = min_note
    self._max_note = max_note

  @property
  def num_classes(self):
    return self._max_note - self._min_note + NUM_SPECIAL_MELODY_EVENTS

  @property
  def default_event(self):
    return MELODY_NO_EVENT

  def encode_event(self, event):
    """Collapses a melody event value into a zero-based index range.

    Args:
      event: A Melody event value. -2 = no event, -1 = note-off event,
          [0, 127] = note-on event for that midi pitch.

    Returns:
      An int in the range [0, self.num_classes). 0 = no event,
      1 = note-off event, [2, self.num_classes) = note-on event for
      that pitch relative to the [self._min_note, self._max_note) range.

    Raises:
      ValueError: If `event` is a MIDI note not between self._min_note and
          self._max_note, or an invalid special event value.
    """
    if event < -NUM_SPECIAL_MELODY_EVENTS:
      raise ValueError('invalid melody event value: %d' % event)
    if (event >= 0) and (event < self._min_note):
      raise ValueError('melody event less than min note: %d < %d' % (
          event, self._min_note))
    if event >= self._max_note:
      raise ValueError('melody event greater than max note: %d >= %d' % (
          event, self._max_note))

    if event < 0:
      return event + NUM_SPECIAL_MELODY_EVENTS
    return event - self._min_note + NUM_SPECIAL_MELODY_EVENTS

  def decode_event(self, index):
    """Expands a zero-based index value to its equivalent melody event value.

    Args:
      index: An int in the range [0, self._num_model_events).
          0 = no event, 1 = note-off event,
          [2, self._num_model_events) = note-on event for that pitch relative
          to the [self._min_note, self._max_note) range.

    Returns:
      A Melody event value. -2 = no event, -1 = note-off event,
      [0, 127] = note-on event for that midi pitch.
    """
    if index < NUM_SPECIAL_MELODY_EVENTS:
      return index - NUM_SPECIAL_MELODY_EVENTS
    return index - NUM_SPECIAL_MELODY_EVENTS + self._min_note


class KeyMelodyEncoderDecoder(encoder_decoder.EventSequenceEncoderDecoder):
  """A MelodyEncoderDecoder that encodes repeated events, time, and key."""

  def __init__(self, min_note, max_note, lookback_distances=None,
               binary_counter_bits=7):
    """Initializes the KeyMelodyEncoderDecoder.

    Args:
      min_note: The minimum midi pitch the encoded melody events can have.
      max_note: The maximum midi pitch (exclusive) the encoded melody events can
          have.
      lookback_distances: A list of step intervals to look back in history to
          encode both the following event and whether the current step is a
          repeat. If None, use default lookback distances.
      binary_counter_bits: The number of input bits to use as a counter for the
          metric position of the next note.
    """
    self._lookback_distances = (lookback_distances
                                if lookback_distances is not None
                                else DEFAULT_LOOKBACK_DISTANCES)
    self._binary_counter_bits = binary_counter_bits
    self._min_note = min_note
    self._note_range = max_note - min_note

  @property
  def input_size(self):
    return (self._note_range +                # current note
            2 +                               # note vs. silence
            1 +                               # attack or not
            1 +                               # ascending or not
            len(self._lookback_distances) +   # whether note matches lookbacks
            self._binary_counter_bits +       # binary counters
            1 +                               # start of bar or not
            NOTES_PER_OCTAVE +                # total key estimate
            NOTES_PER_OCTAVE)                 # recent key estimate

  @property
  def num_classes(self):
    return (self._note_range + NUM_SPECIAL_MELODY_EVENTS +
            len(self._lookback_distances))

  @property
  def default_event_label(self):
    return self._note_range

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the melody.

    Returns a self.input_size length list of floats. Assuming
    self._min_note = 48, self._note_range = 36, two lookback distances, and
    seven binary counters, then self.input_size = 74. Each index represents a
    different input signal to the model.

    Indices [0, 73]:
    [0, 35]: A note is playing at that pitch [48, 84).
    36: Any note is playing.
    37: Silence is playing.
    38: The current event is the note-on event of the currently playing note.
    39: Whether the melody is currently ascending or descending.
    40: The last event is repeating (first lookback distance).
    41: The last event is repeating (second lookback distance).
    [42, 48]: Time keeping toggles.
    49: The next event is the start of a bar.
    [50, 61]: The keys the current melody is in.
    [62, 73]: The keys the last 3 notes are in.
    Args:
      events: A magenta.music.Melody object.
      position: An integer event position in the melody.
    Returns:
      An input vector, an self.input_size length list of floats.
    """
    current_note = None
    is_attack = False
    is_ascending = None
    last_3_notes = collections.deque(maxlen=3)
    sub_melody = melodies_lib.Melody(events[:position + 1])
    for note in sub_melody:
      if note == MELODY_NO_EVENT:
        is_attack = False
      elif note == MELODY_NOTE_OFF:
        current_note = None
      else:
        is_attack = True
        current_note = note
        if last_3_notes:
          if note > last_3_notes[-1]:
            is_ascending = True
          if note < last_3_notes[-1]:
            is_ascending = False
        if note in last_3_notes:
          last_3_notes.remove(note)
        last_3_notes.append(note)

    input_ = [0.0] * self.input_size
    offset = 0
    if current_note:
      # The pitch of current note if a note is playing.
      input_[offset + current_note - self._min_note] = 1.0
      # A note is playing.
      input_[offset + self._note_range] = 1.0
    else:
      # Silence is playing.
      input_[offset + self._note_range + 1] = 1.0
    offset += self._note_range + 2

    # The current event is the note-on event of the currently playing note.
    if is_attack:
      input_[offset] = 1.0
    offset += 1

    # Whether the melody is currently ascending or descending.
    if is_ascending is not None:
      input_[offset] = 1.0 if is_ascending else -1.0
    offset += 1

    # Last event is repeating N bars ago.
    for i, lookback_distance in enumerate(self._lookback_distances):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        input_[offset] = 1.0
      offset += 1

    # Binary time counter giving the metric location of the *next* note.
    n = len(sub_melody)
    for i in range(self._binary_counter_bits):
      input_[offset] = 1.0 if (n / 2 ** i) % 2 else -1.0
      offset += 1

    # The next event is the start of a bar.
    if len(sub_melody) % DEFAULT_STEPS_PER_BAR == 0:
      input_[offset] = 1.0
    offset += 1

    # The keys the current melody is in.
    key_histogram = sub_melody.get_major_key_histogram()
    max_val = max(key_histogram)
    for i, key_val in enumerate(key_histogram):
      if key_val == max_val:
        input_[offset] = 1.0
      offset += 1

    # The keys the last 3 notes are in.
    last_3_note_melody = melodies_lib.Melody(list(last_3_notes))
    key_histogram = last_3_note_melody.get_major_key_histogram()
    max_val = max(key_histogram)
    for i, key_val in enumerate(key_histogram):
      if key_val == max_val:
        input_[offset] = 1.0
      offset += 1

    assert offset == self.input_size

    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the melody.

    Returns an int in the range [0, self.num_classes). Assuming
    self._min_note = 48, self._note_range = 36, and two lookback distances,
    then self.num_classes = 40.
    Values [0, 39]:
    [0, 35]: Note-on event for midi pitch [48, 84).
    36: No event.
    37: Note-off event.
    38: Repeat first lookback (takes precedence over above values).
    39: Repeat second lookback (takes precedence over above values).

    Args:
      events: A magenta.music.Melody object.
      position: An integer event position in the melody.
    Returns:
      A label, an integer.
    """
    if (position < self._lookback_distances[-1] and
        events[position] == MELODY_NO_EVENT):
      return self._note_range + len(self._lookback_distances) + 1

    # If the last event repeated N bars ago.
    for i, lookback_distance in reversed(
        list(enumerate(self._lookback_distances))):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        return self._note_range + 2 + i

    # If last event was a note-off event.
    if events[position] == MELODY_NOTE_OFF:
      return self._note_range + 1

    # If last event was a no event.
    if events[position] == MELODY_NO_EVENT:
      return self._note_range

    # If last event was a note-on event, the pitch of that note.
    return events[position] - self._min_note

  def class_index_to_event(self, class_index, events):
    """Returns the melody event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An int in the range [0, self.num_classes).
      events: The magenta.music.Melody events list of the current melody.
    Returns:
      A magenta.music.Melody event value.
    """
    # Repeat N bars ago.
    for i, lookback_distance in reversed(
        list(enumerate(self._lookback_distances))):
      if class_index == self._note_range + 2 + i:
        if len(events) < lookback_distance:
          return MELODY_NO_EVENT
        return events[-lookback_distance]

    # Note-off event.
    if class_index == self._note_range + 1:
      return MELODY_NOTE_OFF

    # No event:
    if class_index == self._note_range:
      return MELODY_NO_EVENT

    # Note-on event for that midi pitch.
    return self._min_note + class_index
