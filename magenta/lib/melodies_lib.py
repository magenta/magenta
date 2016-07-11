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
"""Utility functions for creating melody training datasets.

Use extract_melodies to extract monophonic melodies from a NoteSequence proto.
"""

import logging
from six.moves import range  # pylint: disable=redefined-builtin

# internal imports
import numpy as np

from magenta.protobuf import music_pb2


# Special events.
NUM_SPECIAL_EVENTS = 2
NOTE_OFF = -1
NO_EVENT = -2

# Other constants.
MIN_MIDI_PITCH = 0  # Inclusive.
MAX_MIDI_PITCH = 127  # Inclusive.
NOTES_PER_OCTAVE = 12
DEFAULT_BEATS_PER_MINUTE = 120.0
BEATS_PER_BAR = 4  # This code assumes 4 beats per measure of music.

# Standard pulses per quarter.
# https://en.wikipedia.org/wiki/Pulses_per_quarter_note
STANDARD_PPQ = 96

# NOTE_KEYS[note] = The major keys that note belongs to.
# ex. NOTE_KEYS[0] lists all the major keys that contain the note C,
# which are:
# [0, 1, 3, 5, 7, 8, 10]
# [C, C#, D#, F, G, G#, A#]
#
# 0 = C
# 1 = C#
# 2 = D
# 3 = D#
# 4 = E
# 5 = F
# 6 = F#
# 7 = G
# 8 = G#
# 9 = A
# 10 = A#
# 11 = B
#
# NOTE_KEYS can be generated using the code below, but is explicitly declared
# for readability:
# scale = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
# NOTE_KEYS = [[j for j in xrange(12) if scale[(i - j) % 12]]
#              for i in xrange(12)]
NOTE_KEYS = [
    [0, 1, 3, 5, 7, 8, 10],
    [1, 2, 4, 6, 8, 9, 11],
    [0, 2, 3, 5, 7, 9, 10],
    [1, 3, 4, 6, 8, 10, 11],
    [0, 2, 4, 5, 7, 9, 11],
    [0, 1, 3, 5, 6, 8, 10],
    [1, 2, 4, 6, 7, 9, 11],
    [0, 2, 3, 5, 7, 8, 10],
    [1, 3, 4, 6, 8, 9, 11],
    [0, 2, 4, 5, 7, 9, 10],
    [1, 3, 5, 6, 8, 10, 11],
    [0, 2, 4, 6, 7, 9, 11]]


class NonIntegerStepsPerBarException(Exception):
  pass


class PolyphonicMelodyException(Exception):
  pass


class MonophonicMelody(object):
  """Stores a quantized stream of monophonic melody events.

  MonophonicMelody is an intermediate representation that all melody models
  can use. NoteSequence proto to melody code will do work to align notes
  and extract monophonic melodies. Model specific code just needs to
  convert MonophonicMelody to SequenceExample protos for TensorFlow.

  MonophonicMelody implements an iterable object. Simply iterate to retrieve
  the melody events.

  MonophonicMelody events are integers in range [-2, 127] (inclusive),
  where negative values are the special event events: NOTE_OFF, and NO_EVENT.
  Non-negative values [0, 127] are note-on events for that midi pitch. A note
  starts at a non-negative value (that is the pitch), and is held through
  subsequent NO_EVENT events until either another non-negative value is reached
  (even if the pitch is the same as the previous note), or a NOTE_OFF event is
  reached. A NOTE_OFF starts at least one step of silence, which continues
  through NO_EVENT events until the next non-negative value.

  NO_EVENT values are treated as default filler. Notes must be inserted
  in ascending order by start time. Note end times will be truncated if the next
  note overlaps.

  Melodies can start at any non-zero time, and are shifted left so that the bar
  containing the first note-on event is the first bar.

  Attributes:
    events: A python list of melody events which are integers. MonophonicMelody events are
        described above.
    offset: When quantizing notes, this is the offset between indices in
        `events` and time steps of incoming melody events. An offset is chosen
        such that the first melody event is close to the beginning of `events`.
    steps_per_bar: Number of steps in a bar (measure) of music.
    last_on: Index of last note-on event added. This index will be within
        the range of `events`.
    last_off: Index of the NOTE_OFF event that belongs to the note-on event
        at `last_on`. This index is likely not in the range of `events` unless
        _write_all_notes was called.
  """

  def __init__(self):
    """Construct an empty MonophonicMelody.

    Args:
      steps_per_bar: How many time steps per bar of music. MonophonicMelody needs to know
          about bars to skip empty bars before the first note.
    """
    self._reset()

  def _reset(self):
    """Clear `events` and last note-on/off information."""
    self.events = []
    self.steps_per_bar = 16
    self.start_step = 0
    self.end_step = 0

  def __iter__(self):
    """Return an iterator over the events in this MonophonicMelody.

    Returns:
      Python iterator over events.
    """
    return iter(self.events)

  def __len__(self):
    """How many events are in this MonophonicMelody.

    Returns:
      Number of events as an int.
    """
    return len(self.events)

  def _add_note(self, pitch, start_step, end_step):
    """Adds the given note to the stream.

    The previous note's end step will be changed to end before this note if
    there is overlap.

    The note is not added if `start_step` is before the start step of the
    previously added note, or if `start_step` equals `end_step`.

    Args:
      pitch: Midi pitch. An integer between 0 and 127 inclusive.
      start_step: A non-zero integer step that the note begins on.
      end_step: An integer step that the note ends on. The note is considered to
          end at the onset of the end step. `end_step` must be greater than
          `start_step`.
    """
    if start_step >= end_step:
      raise BadNoteException('Start step is on or after end step: start=%d, end=%d'
                             % (start_step, end_step))

    if len(self.events) < end_step + 1:
      self.events += [NO_EVENT] * (end_step + 1 - len(self.events))
    elif len(self.events) > end_step + 1:
      del self.events[end_step + 1:]

    self.events[start_step] = pitch
    self.events[end_step] = NOTE_OFF
    for i in range(start_step + 1, end_step):
      self.events[i] = NO_EVENT

  def _get_last_on_off_events(self):
    last_off = len(self.events)
    for i in range(len(self.events) - 1, -1, -1):
      if self.events[i] == NOTE_OFF:
        last_off = i
      if self.events[i] >= MIN_MIDI_PITCH:
        return (i, last_off)
    raise ValueError('No events in the stream')


  def get_note_histogram(self):
    """Gets a histogram of the note occurrences in a melody.

    Returns:
      A list of 12 ints, one for each note value (C at index 0 through B at
      index 11). Each int is the total number of times that note occurred in
      the melody.
    """
    np_melody = np.array(self.events, dtype=int)
    return np.bincount(
        np_melody[np_melody >= MIN_MIDI_PITCH] % NOTES_PER_OCTAVE,
        minlength=NOTES_PER_OCTAVE)

  def get_major_key(self):
    """Finds the major key that this melody most likely belong to.

    Each key is matched against the pitches in the melody. The key that
    matches the most pitches is returned. If multiple keys match equally, the
    key with the lowest index is returned (where the indexes of the keys are
    C = 0 through B = 11).

    Returns:
      An int for the most likely key (C = 0 through B = 11)
    """
    note_histogram = self.get_note_histogram()
    key_histogram = np.zeros(NOTES_PER_OCTAVE)
    for note, count in enumerate(note_histogram):
      key_histogram[NOTE_KEYS[note]] += count
    return key_histogram.argmax()

  def from_quantized_sequence(self, quantized_sequence, start_step=0, track=0, gap_bars=1, ignore_polyphonic_notes=False):
    """Populate self with an iterable of music_pb2.NoteSequence.Note.

    BEATS_PER_BAR/4 time signature is assumed.

    The given list of notes is quantized according to the given beats per minute
    and populated into self. Any existing notes in the instance are cleared.

    0 velocity notes are ignored. The melody is ended when there is a gap of
    `gap` steps or more after a note.

    If note-on events occur at the same step, this melody is cleared and an
    exception is thrown.

    Args:
      notes: Iterable of music_pb2.NoteSequence.Note
      bpm: Beats per minute. This determines the quantization step size in
          seconds. Beats are subdivided according to `steps_per_bar` given to
          the constructor.
      gap: If this many steps or more follow a note, the melody is ended.
      ignore_polyphonic_notes: If true, any notes that come before or land on
          an already added note's start step will be ignored. If false,
          PolyphonicMelodyException will be raised.

    Raises:
      PolyphonicMelodyException: If any of the notes start on the same step when
      quantized and ignore_polyphonic_notes is False.
    """
    self._reset()

    offset = None
    steps_per_bar_float = quantized_sequence.steps_per_beat * quantized_sequence.time_signature.numerator * 4.0 / quantized_sequence.time_signature.denominator
    if steps_per_bar_float % 1 != 0:
      raise NonIntegerStepsPerBarException('There are %f timesteps per bar. Time signature: %d/%d' % (steps_per_bar_float, quantized_sequence.time_signature.numerator, quantized_sequence.time_signature.denominator))
    steps_per_bar = int(steps_per_bar_float)

    # Sort track by note start times.
    notes = sorted(quantized_sequence.tracks[track], key=lambda note: note.start)

    for note in notes:
      if note.start < start_step:
        continue

      # Ignore 0 velocity notes.
      if not note.velocity:
        continue

      if offset is None:
        offset = note.start - note.start % steps_per_bar

      start_index = note.start - offset
      end_index = note.end - offset

      if not self.events:
        self._add_note(note.pitch, start_index, end_index)
        continue

      # If start_step comes before or lands on an already added note's start
      # step, we cannot add it. Discard the melody because it is not monophonic.
      last_on, last_off = self._get_last_on_off_events()
      on_distance = start_index - last_on
      off_distance = start_index - last_off
      if on_distance == 0:
        if ignore_polyphonic_notes:
          # Keep highest note.
          if note.pitch > self.events[start_index]:
            del self.events[start_index:]
          else:
            continue
        else:
          self._reset()
          raise PolyphonicMelodyException()
      elif on_distance < 0:
        raise PolyphonicMelodyException('Unexpected note. Not in ascending order.')

      # If a gap of `gap` or more steps is found, end the melody.
      if len(self) and off_distance >= gap_bars * steps_per_bar:
        break

      # Add the note-on and off events to the melody.
      self._add_note(note.pitch, start_index, end_index)

    if offset is None:
      return

    self.start_step = offset

    # Round up end_step to a multiple of steps_per_bar
    self.end_step = len(self.events) + offset + (-len(self.events) % steps_per_bar)

  def from_event_list(self, events):
    self.events = events

  def to_sequence(self, velocity=100, instrument=0, sequence_start_time=0.0,
                  bpm=120.0):
    """Converts the MonophonicMelody to Sequence proto.

    Args:
      velocity: Midi velocity to give each note. Between 1 and 127 (inclusive).
      instrument: Midi instrument to give each note.
      sequence_start_time: A time in seconds (float) that the first note in the
        sequence will land on.
      bpm: Beats per minute (float).

    Returns:
      A NoteSequence proto encoding the given melody.
    """
    seconds_per_step = 60.0 / bpm * BEATS_PER_BAR / self.steps_per_bar

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().bpm = bpm
    sequence.ticks_per_beat = STANDARD_PPQ

    current_sequence_note = None
    for step, note in enumerate(self):
      if MIN_MIDI_PITCH <= note <= MAX_MIDI_PITCH:
        # End any sustained notes.
        if current_sequence_note is not None:
          current_sequence_note.end_time = (
              step * seconds_per_step + sequence_start_time)

        # Add a note.
        current_sequence_note = sequence.notes.add()
        current_sequence_note.start_time = (
            step * seconds_per_step + sequence_start_time)
        # Give the note an end time now just to be sure it gets closed.
        current_sequence_note.end_time = (
            (step + 1) * seconds_per_step + sequence_start_time)
        current_sequence_note.pitch = note
        current_sequence_note.velocity = velocity
        current_sequence_note.instrument = instrument

      elif note == NOTE_OFF:
        # End any sustained notes.
        if current_sequence_note is not None:
          current_sequence_note.end_time = (
              step * seconds_per_step + sequence_start_time)
          current_sequence_note = None

    return sequence


  def squash(self, min_note, max_note, transpose_to_key):
    """Transpose and octave shift the notes in this MonophonicMelody.

    The key center of this melody is computed with a heuristic, and the notes
    are transposed to be in the given key. The melody is also octave shifted
    to be centered in the given range. Additionally, all notes are octave
    shifted to lie within a given range.

    Args:
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
      transpose_to_key: The melody is transposed to be in this key. 0 = C Major.

    Returns:
      How much notes are transposed by.
    """
    melody_key = self.get_major_key()
    key_diff = transpose_to_key - melody_key
    midi_notes = [note for note in self.events
                  if MIN_MIDI_PITCH <= note <= MAX_MIDI_PITCH]
    if not midi_notes:
      return
    melody_min_note = min(midi_notes)
    melody_max_note = max(midi_notes)
    melody_center = (melody_min_note + melody_max_note) / 2
    target_center = (min_note + max_note - 1) / 2
    center_diff = target_center - (melody_center + key_diff)
    transpose_amount = (
        key_diff +
        NOTES_PER_OCTAVE * int(round(center_diff / float(NOTES_PER_OCTAVE))))
    for i in xrange(len(self.events)):
      # Transpose MIDI pitches. Special events below MIN_MIDI_PITCH are not changed.
      if self.events[i] >= MIN_MIDI_PITCH:
        self.events[i] += transpose_amount
        if self.events[i] < min_note:
          self.events[i] = (
              min_note + (self.events[i] - min_note) % NOTES_PER_OCTAVE)
        elif self.events[i] >= max_note:
          self.events[i] = (max_note - NOTES_PER_OCTAVE +
                            (self.events[i] - max_note) % NOTES_PER_OCTAVE)

    return transpose_amount


def extract_melodies(quantized_sequence, steps_per_beat=4, min_bars=7, gap_bars=1.0,
                     min_unique_pitches=5):
  """Extracts a list of melodies from the given NoteSequence proto.

  A time signature of BEATS_PER_BAR is assumed for each sequence. If the
  sequence has an incompatable time signature, like 3/4, 5/4, etc, then
  the time signature is ignored and BEATS_PER_BAR/4 time is assumed.

  Once a note-on event in a track is encountered, a melody begins. Once a
  gap of silence since the last note-off event of a bar length or more is
  encountered, or the end of the track is reached, that melody is ended. Only
  the first melody of each track is used (this reduces the number of repeated
  melodies that may come from repeated choruses or verses, but may also cause
  unique non-first melodies, such as bridges and outros, to be missed, so maybe
  this should be changed).

  The melody is then checked for validity. The melody is only used if it is
  at least `min_bars` bars long, and has at least `min_unique_pitches` unique
  notes (preventing melodies that only repeat a few notes, such as those found
  in some accompaniment tracks, from being used).

  After scanning each instrument track in the NoteSequence, a list of all the valid
  melodies is returned.

  Args:
    sequence: A NoteSequence proto containing notes.
    steps_per_beat: How many subdivisions of each beat. BEATS_PER_BAR/4 time is
        assumed, so steps per bar is equal to
        `BEATS_PER_BAR` * `steps_per_beat`.
    min_bars: Minimum length of melodies in number of bars. Shorter melodies are
        discarded.
    min_unique_pitches: Minimum number of unique notes with octave equivalence.
        Melodies with too few unique notes are discarded.

  Returns:
    A python list of MonophonicMelody instances.
  """

  melodies = []
  for track in quantized_sequence.tracks:
    start = 0

    # Quantize the track into a MonophonicMelody object.
    # If any notes start at the same time, only one is kept.
    while 1:
      melody = MonophonicMelody()
      melody.from_quantized_sequence(quantized_sequence, track=track, start_step=start, gap_bars=gap_bars, ignore_polyphonic_notes=True)  # TODO: setting to control what level of polyphony is acceptable 
      start = melody.end_step
      if not len(melody):
        break

      # Require a certain melody length.
      if len(melody) - 1 < melody.steps_per_bar * min_bars:
        logging.debug('melody too short')
        continue

      # Require a certain number of unique pitches.
      note_histogram = melody.get_note_histogram()
      unique_pitches = np.count_nonzero(note_histogram)
      if unique_pitches < min_unique_pitches:
        logging.debug('melody too simple')
        continue

      # Require rhythmic diversity.
      # TODO

      melodies.append(melody)

  return melodies

