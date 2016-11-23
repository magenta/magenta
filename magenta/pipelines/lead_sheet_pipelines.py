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
"""Data processing pipelines for lead sheets."""

# internal imports
import tensorflow as tf

from magenta.music import chord_symbols_lib
from magenta.music import events_lib
from magenta.music import lead_sheets_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


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
      lead_sheets, stats = lead_sheets_lib.extract_lead_sheet_fragments(
          quantized_sequence,
          min_bars=self._min_bars,
          max_steps_truncate=self._max_steps,
          min_unique_pitches=self._min_unique_pitches,
          gap_bars=self._gap_bars,
          ignore_polyphonic_notes=self._ignore_polyphonic_notes,
          filter_drums=self._filter_drums,
          require_chords=self._require_chords,
          all_transpositions=self._all_transpositions)
    except events_lib.NonIntegerStepsPerBarException as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      lead_sheets = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    except chord_symbols_lib.ChordSymbolException as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      lead_sheets = []
      stats = [statistics.Counter('chord_symbol_exception', 1)]
    self._set_stats(stats)
    return lead_sheets
