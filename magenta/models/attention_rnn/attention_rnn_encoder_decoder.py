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
"""A MelodyEncoderDecoder specific to the attention RNN model."""

import collections

# internal imports
import magenta

NUM_SPECIAL_MELODY_EVENTS = magenta.music.NUM_SPECIAL_MELODY_EVENTS
MELODY_NOTE_OFF = magenta.music.MELODY_NOTE_OFF
MELODY_NO_EVENT = magenta.music.MELODY_NO_EVENT
MIN_MIDI_PITCH = magenta.music.MIN_MIDI_PITCH
NOTES_PER_OCTAVE = magenta.music.NOTES_PER_OCTAVE
STEPS_PER_BAR = 16  # This code assumes the melodies have 16 steps per bar.

MIN_NOTE = 48  # Inclusive
MAX_NOTE = 84  # Exclusive
TRANSPOSE_TO_KEY = 0  # C Major

# The number of special input indices and label values other than the events
# in the note range.
NUM_SPECIAL_INPUTS = 14 + NOTES_PER_OCTAVE * 2
NUM_SPECIAL_CLASSES = 2
NUM_BINARY_TIME_COUNTERS = 7

# Must be sorted in ascending order.
LOOKBACK_DISTANCES = [STEPS_PER_BAR, STEPS_PER_BAR * 2]


class MelodyEncoderDecoder(magenta.music.MelodyEncoderDecoder):
  """A MelodyEncoderDecoder specific to the attention RNN model.

  Attributes:
    note_range: The number of different note pitches used by this model.
  """

  def __init__(self):
    """Initializes the MelodyEncoderDecoder."""
    super(MelodyEncoderDecoder, self).__init__(MIN_NOTE, MAX_NOTE,
                                               TRANSPOSE_TO_KEY)
    self.note_range = self.max_note - self.min_note

  @property
  def input_size(self):
    return self.note_range + NUM_SPECIAL_INPUTS

  @property
  def num_classes(self):
    return self.note_range + NUM_SPECIAL_MELODY_EVENTS + NUM_SPECIAL_CLASSES

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the melody.

    Returns a self.input_size length list of floats. Assuming MIN_NOTE = 48
    and MAX_NOTE = 84, then self.input_size = 74. Each index represents a
    different input signal to the model.

    Indices [0, 74):
    [0, 36): A note is playing at that pitch [48, 84).
    36: Any note is playing.
    37: Silence is playing.
    38: The current event is the note-on event of the currently playing note.
    39: Whether the melody is currently ascending or descending.
    40: The last event is repeating 1 bar ago.
    41: The last event is repeating 2 bars ago.
    [42, 49): Time keeping toggles.
    49: The next event is the start of a bar.
    [50, 62): The keys the current melody is in.
    [62, 74): The keys the last 3 notes are in.

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
    sub_melody = magenta.music.Melody(events[:position + 1])
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
    if current_note:
      # The pitch of current note if a note is playing.
      input_[current_note - self.min_note] = 1.0
      # A note is playing.
      input_[self.note_range] = 1.0
    else:
      # Silence is playing.
      input_[self.note_range + 1] = 1.0

    # The current event is the note-on event of the currently playing note.
    if is_attack:
      input_[self.note_range + 2] = 1.0

    # Whether the melody is currently ascending or descending.
    if is_ascending is not None:
      input_[self.note_range + 3] = 1.0 if is_ascending else -1.0

    # Last event is repeating N bars ago.
    for i, lookback_distance in enumerate(LOOKBACK_DISTANCES):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        input_[self.note_range + 4 + i] = 1.0

    # Binary time counter giving the metric location of the *next* note.
    n = len(sub_melody)
    for i in range(NUM_BINARY_TIME_COUNTERS):
      input_[self.note_range + 6 + i] = 1.0 if (n / 2 ** i) % 2 else -1.0

    # The next event is the start of a bar.
    if len(sub_melody) % STEPS_PER_BAR == 0:
      input_[self.note_range + 13] = 1.0

    # The keys the current melody is in.
    key_histogram = sub_melody.get_major_key_histogram()
    max_val = max(key_histogram)
    for i, key_val in enumerate(key_histogram):
      if key_val == max_val:
        input_[self.note_range + 14 + i] = 1.0

    # The keys the last 3 notes are in.
    last_3_note_melody = magenta.music.Melody(list(last_3_notes))
    key_histogram = last_3_note_melody.get_major_key_histogram()
    max_val = max(key_histogram)
    for i, key_val in enumerate(key_histogram):
      if key_val == max_val:
        input_[self.note_range + 14 + NOTES_PER_OCTAVE + i] = 1.0

    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the melody.

    Returns an int the range [0, self.num_classes). Assuming MIN_NOTE = 48
    and MAX_NOTE = 84, then self.num_classes = 40.

    Values [0, 40):
    [0, 36): Note-on event for midi pitch [48, 84).
    36: No event.
    37: Note-off event.
    38: Repeat 1 bar ago (takes precedence over above values).
    39: Repeat 2 bars ago (takes precedence over above values).

    Args:
      events: A magenta.music.Melody object.
      position: An integer event position in the melody.

    Returns:
      A label, an integer.
    """
    if (position < LOOKBACK_DISTANCES[-1] and
        events[position] == MELODY_NO_EVENT):
      return self.note_range + len(LOOKBACK_DISTANCES) + 1

    # If the last event repeated N bars ago.
    for i, lookback_distance in reversed(list(enumerate(LOOKBACK_DISTANCES))):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        return self.note_range + 2 + i

    # If last event was a note-off event.
    if events[position] == MELODY_NOTE_OFF:
      return self.note_range + 1

    # If last event was a no event.
    if events[position] == MELODY_NO_EVENT:
      return self.note_range

    # If last event was a note-on event, the pitch of that note.
    return events[position] - self.min_note

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
    for i, lookback_distance in reversed(list(enumerate(LOOKBACK_DISTANCES))):
      if class_index == self.note_range + 2 + i:
        if len(events) < lookback_distance:
          return MELODY_NO_EVENT
        return events[-lookback_distance]

    # Note-off event.
    if class_index == self.note_range + 1:
      return MELODY_NOTE_OFF

    # No event:
    if class_index == self.note_range:
      return MELODY_NO_EVENT

    # Note-on event for that midi pitch.
    return self.min_note + class_index
