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

"""Data processing pipelines for drum tracks."""

from magenta.music import drums_lib
from magenta.music import events_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2
import tensorflow as tf


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
    stats = [statistics.Counter('drum_tracks_discarded_too_short'),
             statistics.Counter('drum_tracks_discarded_too_long'),
             statistics.Counter('drum_tracks_truncated'),
             # Create a histogram measuring drum track lengths (in bars not
             # steps). Capture drum tracks that are very small, in the range of
             # the filter lower bound `min_bars`, and large. The bucket
             # intervals grow approximately exponentially.
             statistics.Histogram(
                 'drum_track_lengths_in_bars',
                 [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, self._min_bars // 2,
                  self._min_bars, self._min_bars + 1, self._min_bars - 1])]
    try:
      drum_tracks = drums_lib.extract_drum_tracks(
          quantized_sequence,
          min_bars=self._min_bars,
          max_steps_truncate=self._max_steps,
          gap_bars=self._gap_bars,
          callbacks={stat.name: stat.increment for stat in stats})
    except events_lib.NonIntegerStepsPerBarError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      drum_tracks = []
      stats.append(statistics.Counter('non_integer_steps_per_bar', 1))
    self._set_stats(stats)
    return drum_tracks
