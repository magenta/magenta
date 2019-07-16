# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for working with lead sheets."""

import copy
import itertools

from magenta.music import chords_lib
from magenta.music import constants
from magenta.music import events_lib
from magenta.music import melodies_lib
from magenta.music.protobuf import music_pb2

# Constants.
DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER

DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class MelodyChordsMismatchError(Exception):
  pass


class LeadSheet(events_lib.EventSequence):
  """A wrapper around Melody and ChordProgression.

  Attributes:
    melody: A Melody object, the lead sheet melody.
    chords: A ChordProgression object, the underlying chords.
  """

  def __init__(self, melody=None, chords=None):
    """Construct a LeadSheet.

    If `melody` and `chords` are specified, instantiate with the provided
    melody and chords.  Otherwise, create an empty LeadSheet.

    Args:
      melody: A Melody object.
      chords: A ChordProgression object.

    Raises:
      MelodyChordsMismatchError: If the melody and chord progression differ
          in temporal resolution or position in the source sequence, or if only
          one of melody or chords is specified.
    """
    if (melody is None) != (chords is None):
      raise MelodyChordsMismatchError(
          'melody and chords must be both specified or both unspecified')
    if melody is not None:
      self._from_melody_and_chords(melody, chords)
    else:
      self._reset()

  def _reset(self):
    """Clear events and reset object state."""
    self._melody = melodies_lib.Melody()
    self._chords = chords_lib.ChordProgression()

  def _from_melody_and_chords(self, melody, chords):
    """Initializes a LeadSheet with a given melody and chords.

    Args:
      melody: A Melody object.
      chords: A ChordProgression object.

    Raises:
      MelodyChordsMismatchError: If the melody and chord progression differ
          in temporal resolution or position in the source sequence.
    """
    if (len(melody) != len(chords) or
        melody.steps_per_bar != chords.steps_per_bar or
        melody.steps_per_quarter != chords.steps_per_quarter or
        melody.start_step != chords.start_step or
        melody.end_step != chords.end_step):
      raise MelodyChordsMismatchError()
    self._melody = melody
    self._chords = chords

  def __iter__(self):
    """Return an iterator over (melody, chord) tuples in this LeadSheet.

    Returns:
      Python iterator over (melody, chord) event tuples.
    """
    return itertools.izip(self._melody, self._chords)

  def __getitem__(self, i):
    """Returns the melody-chord tuple at the given index."""
    return self._melody[i], self._chords[i]

  def __getslice__(self, i, j):
    """Returns a LeadSheet object for the given slice range."""
    return LeadSheet(self._melody[i:j], self._chords[i:j])

  def __len__(self):
    """How many events (melody-chord tuples) are in this LeadSheet.

    Returns:
      Number of events as an integer.
    """
    return len(self._melody)

  def __deepcopy__(self, memo=None):
    return LeadSheet(copy.deepcopy(self._melody, memo),
                     copy.deepcopy(self._chords, memo))

  def __eq__(self, other):
    if not isinstance(other, LeadSheet):
      return False
    return (self._melody == other.melody and
            self._chords == other.chords)

  @property
  def start_step(self):
    return self._melody.start_step

  @property
  def end_step(self):
    return self._melody.end_step

  @property
  def steps(self):
    return self._melody.steps

  @property
  def steps_per_bar(self):
    return self._melody.steps_per_bar

  @property
  def steps_per_quarter(self):
    return self._melody.steps_per_quarter

  @property
  def melody(self):
    """Return the melody of the lead sheet.

    Returns:
        The lead sheet melody, a Melody object.
    """
    return self._melody

  @property
  def chords(self):
    """Return the chord progression of the lead sheet.

    Returns:
        The lead sheet chords, a ChordProgression object.
    """
    return self._chords

  def append(self, event):
    """Appends event to the end of the sequence and increments the end step.

    Args:
      event: The event (a melody-chord tuple) to append to the end.
    """
    melody_event, chord_event = event
    self._melody.append(melody_event)
    self._chords.append(chord_event)

  def to_sequence(self,
                  velocity=100,
                  instrument=0,
                  sequence_start_time=0.0,
                  qpm=120.0):
    """Converts the LeadSheet to NoteSequence proto.

    Args:
      velocity: Midi velocity to give each melody note. Between 1 and 127
          (inclusive).
      instrument: Midi instrument to give each melody note.
      sequence_start_time: A time in seconds (float) that the first note (and
          chord) in the sequence will land on.
      qpm: Quarter notes per minute (float).

    Returns:
      A NoteSequence proto encoding the melody and chords from the lead sheet.
    """
    sequence = self._melody.to_sequence(
        velocity=velocity, instrument=instrument,
        sequence_start_time=sequence_start_time, qpm=qpm)
    chord_sequence = self._chords.to_sequence(
        sequence_start_time=sequence_start_time, qpm=qpm)
    # A little ugly, but just add the chord annotations to the melody sequence.
    for text_annotation in chord_sequence.text_annotations:
      if text_annotation.annotation_type == CHORD_SYMBOL:
        chord = sequence.text_annotations.add()
        chord.CopyFrom(text_annotation)
    return sequence

  def transpose(self, transpose_amount, min_note=0, max_note=128):
    """Transpose notes and chords in this LeadSheet.

    All notes and chords are transposed the specified amount. Additionally,
    all notes are octave shifted to lie within the [min_note, max_note) range.

    Args:
      transpose_amount: The number of half steps to transpose this
          LeadSheet. Positive values transpose up. Negative values
          transpose down.
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
    """
    self._melody.transpose(transpose_amount, min_note, max_note)
    self._chords.transpose(transpose_amount)

  def squash(self, min_note, max_note, transpose_to_key):
    """Transpose and octave shift the notes and chords in this LeadSheet.

    Args:
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
      transpose_to_key: The lead sheet is transposed to be in this key.

    Returns:
      The transpose amount, in half steps.
    """
    transpose_amount = self._melody.squash(min_note, max_note,
                                           transpose_to_key)
    self._chords.transpose(transpose_amount)
    return transpose_amount

  def set_length(self, steps):
    """Sets the length of the lead sheet to the specified number of steps.

    Args:
      steps: How many steps long the lead sheet should be.
    """
    self._melody.set_length(steps)
    self._chords.set_length(steps)

  def increase_resolution(self, k):
    """Increase the resolution of a LeadSheet.

    Increases the resolution of a LeadSheet object by a factor of `k`. This
    increases the resolution of the melody and chords separately, which uses
    MELODY_NO_EVENT to extend each event in the melody, and simply repeats each
    chord event `k` times.

    Args:
      k: An integer, the factor by which to increase the resolution of the lead
          sheet.
    """
    self._melody.increase_resolution(k)
    self._chords.increase_resolution(k)
