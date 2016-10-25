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
"""Data processing pipelines for melodies."""

# internal imports
import tensorflow as tf

from magenta.music import events_lib
from magenta.music import melodies_lib
from magenta.music import sequences_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics


class MelodyExtractor(pipeline.Pipeline):
  """Extracts monophonic melodies from a QuantizedSequence."""

  def __init__(self, min_bars=7, min_unique_pitches=5, gap_bars=1.0,
               ignore_polyphonic_notes=False):
    super(MelodyExtractor, self).__init__(
        input_type=sequences_lib.QuantizedSequence,
        output_type=melodies_lib.Melody)
    self.min_bars = min_bars
    self.min_unique_pitches = min_unique_pitches
    self.gap_bars = gap_bars
    self.ignore_polyphonic_notes = False

  def transform(self, quantized_sequence):
    try:
      melodies, stats = melodies_lib.extract_melodies(
          quantized_sequence,
          min_bars=self.min_bars,
          min_unique_pitches=self.min_unique_pitches,
          gap_bars=self.gap_bars,
          ignore_polyphonic_notes=self.ignore_polyphonic_notes)
    except events_lib.NonIntegerStepsPerBarException as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      melodies = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    self._set_stats(stats)
    return melodies
