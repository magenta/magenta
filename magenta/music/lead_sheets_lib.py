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
"""Utility functions for working with lead sheets."""

import copy
import itertools

# internal imports
from magenta.music import chords_lib
from magenta.music import constants
from magenta.music import events_lib
from magenta.music import melodies_lib
from magenta.music import sequences_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2

# Constants.
DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER

DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class MelodyChordsMismatchException(Exception):
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
      MelodyChordsMismatchException: If the melody and chord progression differ
          in temporal resolution or position in the source sequence, or if only
          one of melody or chords is specified.
    """
    if (melody is None) != (chords is None):
      raise MelodyChordsMismatchException(
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
      MelodyChordsMismatchException: If the melody and chord progression differ
          in temporal resolution or position in the source sequence.
    """
    if (len(melody) != len(chords) or
        melody.steps_per_bar != chords.steps_per_bar or
        melody.steps_per_quarter != chords.steps_per_quarter or
        melody.start_step != chords.start_step or
        melody.end_step != chords.end_step):
      raise MelodyChordsMismatchException()
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


def extract_lead_sheet_fragments(quantized_sequence,
                                 search_start_step=0,
                                 min_bars=7,
                                 max_steps_truncate=None,
                                 max_steps_discard=None,
                                 gap_bars=1.0,
                                 min_unique_pitches=5,
                                 ignore_polyphonic_notes=True,
                                 pad_end=False,
                                 filter_drums=True,
                                 require_chords=False,
                                 all_transpositions=False):
  """Extracts a list of lead sheet fragments from a quantized NoteSequence.

  This function first extracts melodies using melodies_lib.extract_melodies,
  then extracts the chords underlying each melody using
  chords_lib.extract_chords_for_melodies.

  Args:
    quantized_sequence: A quantized NoteSequence object.
    search_start_step: Start searching for a melody at this time step. Assumed
        to be the first step of a bar.
    min_bars: Minimum length of melodies in number of bars. Shorter melodies are
        discarded.
    max_steps_truncate: Maximum number of steps in extracted melodies. If
        defined, longer melodies are truncated to this threshold. If pad_end is
        also True, melodies will be truncated to the end of the last bar below
        this threshold.
    max_steps_discard: Maximum number of steps in extracted melodies. If
        defined, longer melodies are discarded.
    gap_bars: A melody comes to an end when this number of bars (measures) of
        silence is encountered.
    min_unique_pitches: Minimum number of unique notes with octave equivalence.
        Melodies with too few unique notes are discarded.
    ignore_polyphonic_notes: If True, melodies will be extracted from
        `quantized_sequence` tracks that contain polyphony (notes start at the
        same time). If False, tracks with polyphony will be ignored.
    pad_end: If True, the end of the melody will be padded with NO_EVENTs so
        that it will end at a bar boundary.
    filter_drums: If True, notes for which `is_drum` is True will be ignored.
    require_chords: If True, only return lead sheets that have at least one
        chord other than NO_CHORD. If False, lead sheets with only melody will
        also be returned.
    all_transpositions: If True, also transpose each lead sheet fragment into
        all 12 keys.

  Returns:
    A python list of LeadSheet instances.

  Raises:
    NonIntegerStepsPerBarException: If `quantized_sequence`'s bar length
        (derived from its time signature) is not an integer number of time
        steps.
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
  stats = dict([('empty_chord_progressions',
                 statistics.Counter('empty_chord_progressions'))])
  melodies, melody_stats = melodies_lib.extract_melodies(
      quantized_sequence, search_start_step=search_start_step,
      min_bars=min_bars, max_steps_truncate=max_steps_truncate,
      max_steps_discard=max_steps_discard, gap_bars=gap_bars,
      min_unique_pitches=min_unique_pitches,
      ignore_polyphonic_notes=ignore_polyphonic_notes, pad_end=pad_end,
      filter_drums=filter_drums)
  chord_progressions, chord_stats = chords_lib.extract_chords_for_melodies(
      quantized_sequence, melodies)
  lead_sheets = []
  for melody, chords in zip(melodies, chord_progressions):
    # If `chords` is None, it's because a chord progression could not be
    # extracted for this particular melody.
    if chords is not None:
      if require_chords and all(chord == chords_lib.NO_CHORD
                                for chord in chords):
        stats['empty_chord_progressions'].increment()
      else:
        lead_sheet = LeadSheet(melody, chords)
        if all_transpositions:
          for amount in range(-6, 6):
            transposed_lead_sheet = copy.deepcopy(lead_sheet)
            transposed_lead_sheet.transpose(amount)
            lead_sheets.append(transposed_lead_sheet)
        else:
          lead_sheets.append(lead_sheet)
  return lead_sheets, stats.values() + melody_stats + chord_stats
