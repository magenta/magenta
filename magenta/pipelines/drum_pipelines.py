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
"""Data processing pipelines for drum tracks."""

# internal imports
import tensorflow as tf

from magenta.music import drums_lib
from magenta.music import events_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


class DrumsExtractor(pipeline.Pipeline):
  """Extracts drum tracks from a quantized NoteSequence."""

  def __init__(self, min_bars=7, max_steps=512, gap_bars=1.0, name=None):
    super(DrumsExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=drums_lib.DrumTrack,
        name=name)
    self._min_bars = min_bars
    self._max_steps = max_steps
    self._gap_bars = gap_bars

  def transform(self, quantized_sequence):
    try:
      drum_tracks, stats = drums_lib.extract_drum_tracks(
          quantized_sequence,
          min_bars=self._min_bars,
          max_steps_truncate=self._max_steps,
          gap_bars=self._gap_bars)
    except events_lib.NonIntegerStepsPerBarException as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      drum_tracks = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    self._set_stats(stats)
    return drum_tracks
