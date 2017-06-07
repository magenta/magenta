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
"""Utility functions for working with chord progressions.

Use extract_chords_for_melodies to extract chord progressions from a
quantized NoteSequence object, aligned with already-extracted melodies.

Use ChordProgression.to_sequence to write a chord progression to a
NoteSequence proto, encoding the chords as text annotations.
"""

import abc
import copy

from six.moves import range  # pylint: disable=redefined-builtin

from magenta.music import chord_symbols_lib
from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


STANDARD_PPQ = constants.STANDARD_PPQ
NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
NO_CHORD = constants.NO_CHORD

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class CoincidentChordsException(Exception):
  pass


class BadChordException(Exception):
  pass


class ChordProgression(events_lib.SimpleEventSequence):
  """Stores a quantized stream of chord events.

  ChordProgression is an intermediate representation that all chord or lead
  sheet models can use. Chords are represented here by a chord symbol string;
  model-specific code is responsible for converting this representation to
  SequenceExample protos for TensorFlow.

  ChordProgression implements an iterable object. Simply iterate to retrieve
  the chord events.

  ChordProgression events are chord symbol strings like "Cm7", with special
  event NO_CHORD to indicate no chordal harmony. When a chord lasts for longer
  than a single step, the chord symbol event is repeated multiple times. Note
  that this is different from Melody, where the special MELODY_NO_EVENT is used
  for subsequent steps of sustained notes; in the case of harmony, there's no
  distinction between a repeated chord and a sustained chord.

  Chords must be inserted in ascending order by start time.

  Attributes:
    start_step: The offset of the first step of the progression relative to the
        beginning of the source sequence.
    end_step: The offset to the beginning of the bar following the last step
       of the progression relative to the beginning of the source sequence.
    steps_per_quarter: Number of steps in in a quarter note.
    steps_per_bar: Number of steps in a bar (measure) of music.
  """

  def __init__(self, events=None, **kwargs):
    """Construct a ChordProgression."""
    if 'pad_event' in kwargs:
      del kwargs['pad_event']
    super(ChordProgression, self).__init__(pad_event=NO_CHORD,
                                           events=events, **kwargs)

  def _add_chord(self, figure, start_step, end_step):
    """Adds the given chord to the `events` list.

    `start_step` is set to the given chord. Everything after `start_step` in
    `events` is deleted before the chord is added. `events`'s length will be
     changed so that the last event has index `end_step` - 1.

    Args:
      figure: Chord symbol figure. A string like "Cm9" representing the chord.
      start_step: A non-negative integer step that the chord begins on.
      end_step: An integer step that the chord ends on. The chord is considered
          to end at the onset of the end step. `end_step` must be greater than
          `start_step`.

    Raises:
      BadChordException: If `start_step` does not precede `end_step`.
    """
    if start_step >= end_step:
      raise BadChordException(
          'Start step does not precede end step: start=%d, end=%d' %
          (start_step, end_step))

    self.set_length(end_step)

    for i in range(start_step, end_step):
      self._events[i] = figure

  def from_quantized_sequence(self, quantized_sequence, start_step, end_step):
    """Populate self with the chords from the given quantized NoteSequence.

    A chord progression is extracted from the given sequence starting at time
    step `start_step` and ending at time step `end_step`.

    The number of time steps per bar is computed from the time signature in
    `quantized_sequence`.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start populating chords at this time step.
      end_step: Stop populating chords at this time step.

    Raises:
      NonIntegerStepsPerBarException: If `quantized_sequence`'s bar length
          (derived from its time signature) is not an integer number of time
          steps.
      CoincidentChordsException: If any of the chords start on the same step.
    """
    sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
    self._reset()

    steps_per_bar_float = sequences_lib.steps_per_bar_in_quantized_sequence(
        quantized_sequence)
    if steps_per_bar_float % 1 != 0:
      raise events_lib.NonIntegerStepsPerBarException(
          'There are %f timesteps per bar. Time signature: %d/%d' %
          (steps_per_bar_float, quantized_sequence.time_signature.numerator,
           quantized_sequence.time_signature.denominator))
    self._steps_per_bar = int(steps_per_bar_float)
    self._steps_per_quarter = (
        quantized_sequence.quantization_info.steps_per_quarter)

    # Sort track by chord times.
    chords = sorted([a for a in quantized_sequence.text_annotations
                     if a.annotation_type == CHORD_SYMBOL],
                    key=lambda chord: chord.quantized_step)

    prev_step = None
    prev_figure = NO_CHORD

    for chord in chords:
      if chord.quantized_step >= end_step:
        # No more chords within range.
        break

      elif chord.quantized_step < start_step:
        # Chord is before start of range.
        prev_step = chord.quantized_step
        prev_figure = chord.text
        continue

      if chord.quantized_step == prev_step:
        if chord.text == prev_figure:
          # Identical coincident chords, just skip.
          continue
        else:
          # Two different chords start at the same time step.
          self._reset()
          raise CoincidentChordsException('chords %s and %s are coincident' %
                                          (prev_figure, chord.text))

      if chord.quantized_step > start_step:
        # Add the previous chord.
        start_index = max(prev_step, start_step) - start_step
        end_index = chord.quantized_step - start_step
        self._add_chord(prev_figure, start_index, end_index)

      prev_step = chord.quantized_step
      prev_figure = chord.text

    if prev_step < end_step:
      # Add the last chord active before end_step.
      start_index = max(prev_step, start_step) - start_step
      end_index = end_step - start_step
      self._add_chord(prev_figure, start_index, end_index)

    self._start_step = start_step
    self._end_step = end_step

  def to_sequence(self,
                  sequence_start_time=0.0,
                  qpm=120.0):
    """Converts the ChordProgression to NoteSequence proto.

    This doesn't generate actual notes, but text annotations specifying the
    chord changes when they occur.

    Args:
      sequence_start_time: A time in seconds (float) that the first chord in
          the sequence will land on.
      qpm: Quarter notes per minute (float).

    Returns:
      A NoteSequence proto encoding the given chords as text annotations.
    """
    seconds_per_step = 60.0 / qpm / self.steps_per_quarter

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().qpm = qpm
    sequence.ticks_per_quarter = STANDARD_PPQ

    current_figure = NO_CHORD
    for step, figure in enumerate(self):
      if figure != current_figure:
        current_figure = figure
        chord = sequence.text_annotations.add()
        chord.time = step * seconds_per_step + sequence_start_time
        chord.text = figure
        chord.annotation_type = CHORD_SYMBOL

    return sequence

  def transpose(self, transpose_amount):
    """Transpose chords in this ChordProgression.

    Args:
      transpose_amount: The number of half steps to transpose this
          ChordProgression. Positive values transpose up. Negative values
          transpose down.

    Raises:
      ChordSymbolException: If a chord (other than "no chord") fails to be
          interpreted by the `chord_symbols_lib` module.
    """
    for i in xrange(len(self._events)):
      if self._events[i] != NO_CHORD:
        self._events[i] = chord_symbols_lib.transpose_chord_symbol(
            self._events[i], transpose_amount % NOTES_PER_OCTAVE)


def extract_chords(quantized_sequence, max_steps=None,
                   all_transpositions=False):
  """Extracts a single chord progression from a quantized NoteSequence.

  This function will extract the underlying chord progression (encoded as text
  annotations) from `quantized_sequence`.

  Args:
    quantized_sequence: A quantized NoteSequence.
    max_steps: An integer, maximum length of a chord progression. Chord
        progressions will be trimmed to this length. If None, chord
        progressions will not be trimmed.
    all_transpositions: If True, also transpose the chord progression into all
        12 keys.

  Returns:
    chord_progressions: If `all_transpositions` is False, a python list
        containing a single ChordProgression instance. If `all_transpositions`
        is True, a python list containing 12 ChordProgression instances, one
        for each transposition.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

  stats = dict([('chords_truncated', statistics.Counter('chords_truncated'))])
  chords = ChordProgression()
  chords.from_quantized_sequence(
      quantized_sequence, 0, quantized_sequence.total_quantized_steps)
  if max_steps is not None:
    if len(chords) > max_steps:
      chords.set_length(max_steps)
      stats['chords_truncated'].increment()
  if all_transpositions:
    chord_progressions = []
    for amount in range(-6, 6):
      transposed_chords = copy.deepcopy(chords)
      transposed_chords.transpose(amount)
      chord_progressions.append(transposed_chords)
    return chord_progressions, stats.values()
  else:
    return [chords], stats.values()


def extract_chords_for_melodies(quantized_sequence, melodies):
  """Extracts a chord progression from the quantized NoteSequence for melodies.

  This function will extract the underlying chord progression (encoded as text
  annotations) from `quantized_sequence` for each monophonic melody in
  `melodies`.  Each chord progression will be the same length as its
  corresponding melody.

  Args:
    quantized_sequence: A quantized NoteSequence object.
    melodies: A python list of Melody instances.

  Returns:
    chord_progressions: A python list of ChordProgression instances, the same
        length as `melodies`. If a progression fails to be extracted for a
        melody, the corresponding list entry will be None.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  """
  chord_progressions = []
  stats = dict([('coincident_chords', statistics.Counter('coincident_chords'))])
  for melody in melodies:
    try:
      chords = ChordProgression()
      chords.from_quantized_sequence(
          quantized_sequence, melody.start_step, melody.end_step)
    except CoincidentChordsException:
      stats['coincident_chords'].increment()
      chords = None
    chord_progressions.append(chords)

  return chord_progressions, stats.values()


class ChordRenderer(object):
  """An abstract class for rendering NoteSequence chord symbols as notes."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def render(self, sequence):
    """Renders the chord symbols of a NoteSequence.

    This function renders chord symbol annotations in a NoteSequence as actual
    notes. Notes are added to the NoteSequence object, and the chord symbols
    remain also.

    Args:
      sequence: The NoteSequence for which to render chord symbols.
    """
    pass


class BasicChordRenderer(ChordRenderer):
  """A chord renderer that holds each note for the duration of the chord."""

  def __init__(self,
               velocity=100,
               instrument=1,
               program=88,
               octave=4,
               bass_octave=3):
    """Initialize a BasicChordRenderer object.

    Args:
      velocity: The MIDI note velocity to use.
      instrument: The MIDI instrument to use.
      program: The MIDI program to use.
      octave: The octave in which to render chord notes. If the bass note is not
          otherwise part of the chord, it will not be rendered in this octave.
      bass_octave: The octave in which to render chord bass notes.
    """
    self._velocity = velocity
    self._instrument = instrument
    self._program = program
    self._octave = octave
    self._bass_octave = bass_octave

  def _render_notes(self, sequence, pitches, bass_pitch, start_time, end_time):
    all_pitches = []
    for pitch in pitches:
      all_pitches.append(12 * self._octave + pitch % 12)
    all_pitches.append(12 * self._bass_octave + bass_pitch % 12)

    for pitch in all_pitches:
      # Add a note.
      note = sequence.notes.add()
      note.start_time = start_time
      note.end_time = end_time
      note.pitch = pitch
      note.velocity = self._velocity
      note.instrument = self._instrument
      note.program = self._program

  def render(self, sequence):
    # Sort text annotations by time.
    annotations = sorted(sequence.text_annotations, key=lambda a: a.time)

    prev_time = 0.0
    prev_figure = NO_CHORD

    for annotation in annotations:
      if annotation.time >= sequence.total_time:
        break

      if annotation.annotation_type == CHORD_SYMBOL:
        if prev_figure != NO_CHORD:
          # Render the previous chord.
          pitches = chord_symbols_lib.chord_symbol_pitches(prev_figure)
          bass_pitch = chord_symbols_lib.chord_symbol_bass(prev_figure)
          self._render_notes(sequence=sequence,
                             pitches=pitches,
                             bass_pitch=bass_pitch,
                             start_time=prev_time,
                             end_time=annotation.time)

        prev_time = annotation.time
        prev_figure = annotation.text

    if (prev_time < sequence.total_time and
        prev_figure != NO_CHORD):
      # Render the last chord.
      pitches = chord_symbols_lib.chord_symbol_pitches(prev_figure)
      bass_pitch = chord_symbols_lib.chord_symbol_bass(prev_figure)
      self._render_notes(sequence=sequence,
                         pitches=pitches,
                         bass_pitch=bass_pitch,
                         start_time=prev_time,
                         end_time=sequence.total_time)
