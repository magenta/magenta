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
"""Data processing pipelines for chord progressions."""

# internal imports
import tensorflow as tf

from magenta.music import chord_symbols_lib
from magenta.music import chords_lib
from magenta.music import events_lib
from magenta.music import sequences_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics


class ChordsExtractor(pipeline.Pipeline):
  """Extracts a chord progression from a QuantizedSequence."""

  def __init__(self, max_steps=512, all_transpositions=False):
    super(ChordsExtractor, self).__init__(
        input_type=sequences_lib.QuantizedSequence,
        output_type=chords_lib.ChordProgression)
    self._max_steps = max_steps
    self._all_transpositions = all_transpositions

  def transform(self, quantized_sequence):
    try:
      chord_progressions, stats = chords_lib.extract_chords(
          quantized_sequence, max_steps=self._max_steps,
          all_transpositions=self._all_transpositions)
    except events_lib.NonIntegerStepsPerBarException as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      chord_progressions = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    except chords_lib.CoincidentChordsException as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      chord_progressions = []
      stats = [statistics.Counter('coincident_chords', 1)]
    except chord_symbols_lib.ChordSymbolException as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      chord_progressions = []
      stats = [statistics.Counter('chord_symbol_exception', 1)]
    self._set_stats(stats)
    return chord_progressions
