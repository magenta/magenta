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

"""Data processing pipelines for lead sheets."""
import copy

from magenta.pipelines import chord_pipelines
from magenta.pipelines import melody_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from note_seq import chord_symbols_lib
from note_seq import chords_lib
from note_seq import events_lib
from note_seq import lead_sheets_lib
from note_seq import LeadSheet
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf


class LeadSheetExtractor(pipeline.Pipeline):
  """Extracts lead sheet fragments from a quantized NoteSequence."""

  def __init__(self, min_bars=7, max_steps=512, min_unique_pitches=5,
               gap_bars=1.0, ignore_polyphonic_notes=False, filter_drums=True,
               require_chords=True, all_transpositions=True, name=None):
    super(LeadSheetExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=lead_sheets_lib.LeadSheet,
        name=name)
    self._min_bars = min_bars
    self._max_steps = max_steps
    self._min_unique_pitches = min_unique_pitches
    self._gap_bars = gap_bars
    self._ignore_polyphonic_notes = ignore_polyphonic_notes
    self._filter_drums = filter_drums
    self._require_chords = require_chords
    self._all_transpositions = all_transpositions

  def transform(self, quantized_sequence):
    try:
      lead_sheets, stats = extract_lead_sheet_fragments(
          quantized_sequence,
          min_bars=self._min_bars,
          max_steps_truncate=self._max_steps,
          min_unique_pitches=self._min_unique_pitches,
          gap_bars=self._gap_bars,
          ignore_polyphonic_notes=self._ignore_polyphonic_notes,
          filter_drums=self._filter_drums,
          require_chords=self._require_chords,
          all_transpositions=self._all_transpositions)
    except events_lib.NonIntegerStepsPerBarError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      lead_sheets = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    except chord_symbols_lib.ChordSymbolError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      lead_sheets = []
      stats = [statistics.Counter('chord_symbol_exception', 1)]
    self._set_stats(stats)
    return lead_sheets


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
    NonIntegerStepsPerBarError: If `quantized_sequence`'s bar length
        (derived from its time signature) is not an integer number of time
        steps.
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
  stats = dict([('empty_chord_progressions',
                 statistics.Counter('empty_chord_progressions'))])
  melodies, melody_stats = melody_pipelines.extract_melodies(
      quantized_sequence, search_start_step=search_start_step,
      min_bars=min_bars, max_steps_truncate=max_steps_truncate,
      max_steps_discard=max_steps_discard, gap_bars=gap_bars,
      min_unique_pitches=min_unique_pitches,
      ignore_polyphonic_notes=ignore_polyphonic_notes, pad_end=pad_end,
      filter_drums=filter_drums)
  chord_progressions, chord_stats = chord_pipelines.extract_chords_for_melodies(
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
  return lead_sheets, list(stats.values()) + melody_stats + chord_stats
