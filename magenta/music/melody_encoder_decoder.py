
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
"""Classes for converting between Melody objects and models inputs/outputs."""

import abc
import collections

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import melodies_lib

NUM_SPECIAL_MELODY_EVENTS = constants.NUM_SPECIAL_MELODY_EVENTS
MELODY_NOTE_OFF = constants.MELODY_NOTE_OFF
MELODY_NO_EVENT = constants.MELODY_NO_EVENT
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH
MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_LOOKBACK_DISTANCES = [DEFAULT_STEPS_PER_BAR, DEFAULT_STEPS_PER_BAR * 2]


class MelodyEncoderDecoder(events_lib.EventsEncoderDecoder):
  """An abstract class for translating between melodies and model data.

  When building your dataset, the `encode` method takes in a melody and
  returns a SequenceExample of inputs and labels. These SequenceExamples are
  fed into the model during training and evaluation.

  During melody generation, the `get_inputs_batch` method takes in a list of
  the current melodies and returns an inputs batch which is fed into the
  model to predict what the next note should be for each melody.
  The `extend_event_sequences` method takes in the list of melodies and the
  softmax returned by the model and extends each melody by one step by sampling
  from the softmax probabilities. This loop (`get_inputs_batch` -> inputs batch
  is fed through the model to get a softmax -> `extend_event_sequences`) is
  repeated until the generated melodies have reached the desired length.

  Attributes:
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.
    transpose_to_key: The key that encoded melodies will be transposed into.

  Properties:
    input_size: The length of the list returned by self.events_to_input.
    num_classes: The range of ints that can be returned by
        self.events_to_label.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, min_note=48, max_note=84, transpose_to_key=0):
    """Initializes a MelodyEncoderDecoder object.

    You can change `min_note` and `max_note` to increase/decrease the melody
    range. Since melodies are transposed into this range to be run through
    the model and then transposed back into their original range after the
    melodies have been extended, the location of the range is somewhat
    arbitrary, but the size of the range determines the possible size of the
    generated melodies range. `transpose_to_key` should be set to the key
    that if melodies were transposed into that key, they would best sit
    between `min_note` and `max_note` with having as few notes outside that
    range. The same `min_note`, `max_note`, and `transpose_to_key` values
    should be used when creating your dataset, training your model,
    and generating melodies from it. If you change `min_note`, `max_note`,
    or `transpose_to_key`, you will have to recreate your dataset and retrain
    your model before you can accurately generate melodies from it.

    Args:
      min_note: The minimum midi pitch the encoded melodies can have.
      max_note: The maximum midi pitch (exclusive) the encoded melodies can
          have.
      transpose_to_key: The key that encoded melodies will be transposed into,
          or None if it should not be transposed.

    Raises:
      ValueError: If `min_note` or `max_note` are outside the midi range, or
          if the [`min_note`, `max_note`) range is less than an octave. A range
          of at least an octave is required to be able to octave shift notes
          into that range while preserving their scale value.
    """
    if min_note < MIN_MIDI_PITCH:
      raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > MAX_MIDI_PITCH + 1:
      raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note - min_note < NOTES_PER_OCTAVE:
      raise ValueError('max_note - min_note must be >= 12. min_note is %d. '
                       'max_note is %d. max_note - min_note is %d.' %
                       (min_note, max_note, max_note - min_note))
    if (transpose_to_key is not None and
        (transpose_to_key < 0 or transpose_to_key > NOTES_PER_OCTAVE - 1)):
      raise ValueError('transpose_to_key must be >= 0 and <= 11. '
                       'transpose_to_key is %d.' % transpose_to_key)

    self._min_note = min_note
    self._max_note = max_note
    self._transpose_to_key = transpose_to_key

  @property
  def num_melody_events(self):
    """The number of melody events in the encoding."""
    return self._max_note - self._min_note + NUM_SPECIAL_MELODY_EVENTS

  @property
  def min_note(self):
    """The minimum midi pitch the encoded melodies can have."""
    return self._min_note

  @property
  def max_note(self):
    """The maximum midi pitch (exclusive) the encoded melodies can have."""
    return self._max_note

  @property
  def transpose_to_key(self):
    """The key that encoded melodies will be transposed into.

    Returns:
      An integer in the range [0, 12), the key that encoded melodies will be
      transposed into, or None if it should not be transposed.
    """
    return self._transpose_to_key

  @property
  def no_event_label(self):
    """The class label that represents a NO_EVENT Melody event.

    Returns:
      An int, the class label that represents a NO_EVENT.
    """
    melody = melodies_lib.Melody([MELODY_NO_EVENT])
    return self.events_to_label(melody, 0)

  def melody_event_to_index(self, melody_event):
    """Collapses a melody event value into a zero-based index range.

    Args:
      melody_event: A Melody event value. -2 = no event, -1 = note-off event,
          [0, 127] = note-on event for that midi pitch.

    Returns:
      An int in the range [0, self.num_melody_events). 0 = no event,
      1 = note-off event, [2, self.num_melody_events) = note-on event for
      that pitch relative to the [self.min_note, self.max_note) range.
    """
    if melody_event < 0:
      return melody_event + NUM_SPECIAL_MELODY_EVENTS
    return melody_event - self.min_note + NUM_SPECIAL_MELODY_EVENTS

  def index_to_melody_event(self, index):
    """Expands a zero-based index value to its equivalent melody event value.

    Args:
      index: An int in the range [0, self.num_melody_events).
          0 = no event, 1 = note-off event,
          [2, self.num_melody_events) = note-on event for that pitch relative
          to the [self.min_note, self.max_note) range.

    Returns:
      A Melody event value. -2 = no event, -1 = note-off event,
      [0, 127] = note-on event for that midi pitch.
    """
    if index < NUM_SPECIAL_MELODY_EVENTS:
      return index - NUM_SPECIAL_MELODY_EVENTS
    return index - NUM_SPECIAL_MELODY_EVENTS + self.min_note

  @abc.abstractmethod
  def events_to_input(self, events, position):
    """Returns the input vector for the melody event at the given position.

    Args:
      events: A Melody object.
      position: An integer event position in the melody.

    Returns:
      An input vector, a self.input_size length list of floats.
    """
    pass

  @abc.abstractmethod
  def events_to_label(self, events, position):
    """Returns the label for the melody event at the given position.

    Args:
      events: A Melody object.
      position: An integer event position in the melody.

    Returns:
      A label, an integer in the range [0, self.num_classes).
    """
    pass

  @abc.abstractmethod
  def class_index_to_event(self, class_index, events):
    """Returns the melody event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A Melody object.

    Returns:
      An integer melody event value.
    """
    pass

  def squash_and_encode(self, melody):
    """Returns a SequenceExample for the given melody after squashing.

    Args:
      melody: A Melody object.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    melody.squash(self._min_note, self._max_note, self._transpose_to_key)
    return self._encode(melody)


class OneHotMelodyEncoderDecoder(MelodyEncoderDecoder):
  """A MelodyEncoderDecoder that produces a one-hot encoding for the input."""

  @property
  def input_size(self):
    return self.num_melody_events

  @property
  def num_classes(self):
    return self.num_melody_events

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the melody.

    Returns a one-hot vector for the given position in the melody mapped to the
    model's event range. 0 = no event, 1 = note-off event,
    [2, self.num_classes) = note-on event for that pitch relative to the
    [self.min_note, self.max_note) range.

    Args:
      events: A Melody object.
      position: An integer event position in the melody.

    Returns:
      An input vector, a list of floats.
    """
    input_ = [0.0] * self.input_size
    input_[self.melody_event_to_index(events[position])] = 1.0
    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the melody.

    Returns the zero-based index value for the given position in the melody
    mapped to the model's event range. 0 = no event, 1 = note-off event,
    [2, self.num_classes) = note-on event for that pitch relative to the
    [self.min_note, self.max_note) range.

    Args:
      events: A Melody object.
      position: An integer event position in the melody.

    Returns:
      A label, an integer.
    """
    return self.melody_event_to_index(events[position])

  def class_index_to_event(self, class_index, events):
    """Returns the melody event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A Melody object. This object is not used in this implementation.

    Returns:
      A Melody event value.
    """
    return self.index_to_melody_event(class_index)


class LookbackMelodyEncoderDecoder(MelodyEncoderDecoder):
  """A MelodyEncoderDecoder that encodes repeated events and keeps time.

  Args:
    lookback_distances: A list of step intervals to look back in history to
       encode both the following event and whether the current step is a repeat.
       Uses default values if None.
    binary_counter_bits: The number of input bits to use as a counter for the
       metric position of the next note.
  """

  def __init__(self, lookback_distances=None, binary_counter_bits=5,
               min_note=48, max_note=84, transpose_to_key=0):
    """Initializes the MelodyEncoderDecoder."""
    super(LookbackMelodyEncoderDecoder, self).__init__(
        min_note, max_note, transpose_to_key)
    self._lookback_distances = (DEFAULT_LOOKBACK_DISTANCES
                                if lookback_distances is None
                                else lookback_distances)
    self._binary_counter_bits = binary_counter_bits

  @property
  def input_size(self):
    num_lookbacks = len(self._lookback_distances)
    return ((num_lookbacks + 1) * self.num_melody_events +
            self._binary_counter_bits + num_lookbacks)

  @property
  def num_classes(self):
    return self.num_melody_events + len(self._lookback_distances)

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the melody.

    Returns a self.input_size length list of floats. Assuming self.min_note = 48
    and self.max_note = 84, self.input_size will = 121. Each index represents a
    different input signal to the model.

    Indices [0, 120]:
    [0, 37]: Event of current step.
    [38, 75]: Event of next step if repeating 1 bar ago.
    [76, 113]: Event of next step if repeating 2 bars ago.
    114: 16th note binary counter.
    115: 8th note binary counter.
    116: 4th note binary counter.
    117: Half note binary counter.
    118: Whole note binary counter.
    119: The current step is repeating 1 bar ago.
    120: The current step is repeating 2 bars ago.

    Args:
      events: A magenta.music.Melody object.
      position: An integer position in the melody.

    Returns:
      An input vector, an self.input_size length list of floats.
    """
    input_ = [0.0] * self.input_size

    # Last event.
    index = self.melody_event_to_index(events[position])
    input_[index] = 1.0

    # Next event if repeating N positions ago.
    for i, lookback_distance in enumerate(self._lookback_distances):
      lookback_position = position - lookback_distance + 1
      if lookback_position < 0:
        melody_event = MELODY_NO_EVENT
      else:
        melody_event = events[lookback_position]
      index = self.melody_event_to_index(melody_event)
      input_[i * self.num_melody_events + index] = 1.0

    # Binary time counter giving the metric location of the *next* note.
    n = position + 1
    for i in range(self._binary_counter_bits):
      input_[3 * self.num_melody_events + i] = 1.0 if (n / 2 ** i) % 2 else -1.0

    # Last event is repeating N bars ago.
    for i, lookback_distance in enumerate(self._lookback_distances):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        input_[3 * self.num_melody_events + 5 + i] = 1.0

    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the melody.

    Returns an integer in the range [0, self.num_classes). Indices in the range
    [0, self.num_melody_events) map to standard midi events. Indices
    self.num_melody_events and self.num_melody_events + 1 are signals to repeat
    events from earlier in the melody. More distant repeats are selected first
    and standard midi events are selected last.

    Assuming self.min_note = 48 and self.max_note = 84, then
    self.num_classes = 40, self.num_melody_events = 38, and the values will be
    as follows.
    Values [0, 39]:
      [0, 37]: Event of the last step in the melody, if not repeating 1 or 2
               bars ago.
      38: If the last event in the melody is repeating 1 bar ago, if not
          repeating 2 bars ago.
      39: If the last event in the melody is repeating 2 bars ago.

    Args:
      events: A magenta.music.Melody object.
      position: An integer position in the melody.

    Returns:
      A label, an integer.
    """
    if (position < self._lookback_distances[-1] and
        events[position] == MELODY_NO_EVENT):
      return self.num_melody_events + len(self._lookback_distances) - 1

    # If last step repeated N bars ago.
    for i, lookback_distance in reversed(
        list(enumerate(self._lookback_distances))):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        return self.num_melody_events + i

    # If last step didn't repeat at one of the lookback positions, use the
    # specific event.
    return self.melody_event_to_index(events[position])

  def class_index_to_event(self, class_index, events):
    """Returns the melody event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An int in the range [0, self.num_classes).
      events: The magenta.music.Melody events list of the current melody.

    Returns:
      A magenta.music.Melody event value.
    """
    # Repeat N bar ago.
    for i, lookback_distance in reversed(
        list(enumerate(self._lookback_distances))):
      if class_index == self.num_melody_events + i:
        if len(events) < lookback_distance:
          return MELODY_NO_EVENT
        return events[-lookback_distance]

    # Return the melody event for that class index.
    return self.index_to_melody_event(class_index)


class KeyMelodyEncoderDecoder(MelodyEncoderDecoder):
  """A MelodyEncoderDecoder that encodes repeated events, time, and key.

  Args:
    lookback_distances: A list of step intervals to look back in history to
       encode both the following event and whether the current step is a repeat.
       Uses default values if None.
    binary_counter_bits: The number of input bits to use as a counter for the
       metric position of the next note.
  """

  def __init__(self, lookback_distances=None, binary_counter_bits=7,
               min_note=48, max_note=84, transpose_to_key=0):
    """Initializes the MelodyEncoderDecoder."""
    super(KeyMelodyEncoderDecoder, self).__init__(
        min_note, max_note, transpose_to_key)
    self._lookback_distances = (DEFAULT_LOOKBACK_DISTANCES
                                if lookback_distances is None
                                else lookback_distances)
    self._binary_counter_bits = binary_counter_bits
    self._note_range = max_note - min_note

  @property
  def input_size(self):
    return (self._note_range + self._binary_counter_bits +
            len(self._lookback_distances) + NOTES_PER_OCTAVE * 2 + 5)

  @property
  def num_classes(self):
    return self.num_melody_events + len(self._lookback_distances)

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the melody.

    Returns a self.input_size length list of floats. Assuming self.min_note = 48
    and self.max_note = 84, then self.input_size = 74. Each index represents a
    different input signal to the model.
    Indices [0, 73]:
    [0, 35]: A note is playing at that pitch [48, 84).
    36: Any note is playing.
    37: Silence is playing.
    38: The current event is the note-on event of the currently playing note.
    39: Whether the melody is currently ascending or descending.
    40: The last event is repeating 1 bar ago.
    41: The last event is repeating 2 bars ago.
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
    if current_note:
      # The pitch of current note if a note is playing.
      input_[current_note - self.min_note] = 1.0
      # A note is playing.
      input_[self._note_range] = 1.0
    else:
      # Silence is playing.
      input_[self._note_range + 1] = 1.0

    # The current event is the note-on event of the currently playing note.
    if is_attack:
      input_[self._note_range + 2] = 1.0

    # Whether the melody is currently ascending or descending.
    if is_ascending is not None:
      input_[self._note_range + 3] = 1.0 if is_ascending else -1.0

    # Last event is repeating N bars ago.
    for i, lookback_distance in enumerate(self._lookback_distances):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        input_[self._note_range + 4 + i] = 1.0

    # Binary time counter giving the metric location of the *next* note.
    n = len(sub_melody)
    for i in range(self._binary_counter_bits):
      input_[self._note_range + 6 + i] = 1.0 if (n / 2 ** i) % 2 else -1.0

    # The next event is the start of a bar.
    if len(sub_melody) % DEFAULT_STEPS_PER_BAR == 0:
      input_[self._note_range + 13] = 1.0

    # The keys the current melody is in.
    key_histogram = sub_melody.get_major_key_histogram()
    max_val = max(key_histogram)
    for i, key_val in enumerate(key_histogram):
      if key_val == max_val:
        input_[self._note_range + 14 + i] = 1.0

    # The keys the last 3 notes are in.
    last_3_note_melody = melodies_lib.Melody(list(last_3_notes))
    key_histogram = last_3_note_melody.get_major_key_histogram()
    max_val = max(key_histogram)
    for i, key_val in enumerate(key_histogram):
      if key_val == max_val:
        input_[self._note_range + 14 + NOTES_PER_OCTAVE + i] = 1.0

    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the melody.

    Returns an int the range [0, self.num_classes). Assuming self.min_note = 48
    and self.max_note = 84, then self.num_classes = 40.
    Values [0, 39]:
    [0, 35]: Note-on event for midi pitch [48, 84).
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
    return self.min_note + class_index
