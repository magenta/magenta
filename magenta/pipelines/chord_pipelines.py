# Copyright 2020 The Magenta Authors.
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

"""Data processing pipelines for chord progressions."""
import copy

from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from note_seq import chord_symbols_lib
from note_seq import ChordProgression
from note_seq import chords_lib
from note_seq import events_lib
from note_seq import sequences_lib
from note_seq.chords_lib import CoincidentChordsError
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf


class ChordsExtractor(pipeline.Pipeline):
  """Extracts a chord progression from a quantized NoteSequence."""

  def __init__(self, max_steps=512, all_transpositions=False, name=None):
    super(ChordsExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=chords_lib.ChordProgression,
        name=name)
    self._max_steps = max_steps
    self._all_transpositions = all_transpositions

  def transform(self, quantized_sequence):
    try:
      chord_progressions, stats = extract_chords(
          quantized_sequence, max_steps=self._max_steps,
          all_transpositions=self._all_transpositions)
    except events_lib.NonIntegerStepsPerBarError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      chord_progressions = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    except chords_lib.CoincidentChordsError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      chord_progressions = []
      stats = [statistics.Counter('coincident_chords', 1)]
    except chord_symbols_lib.ChordSymbolError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      chord_progressions = []
      stats = [statistics.Counter('chord_symbol_exception', 1)]
    self._set_stats(stats)
    return chord_progressions


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
    except CoincidentChordsError:
      stats['coincident_chords'].increment()
      chords = None
    chord_progressions.append(chords)

  return chord_progressions, list(stats.values())
