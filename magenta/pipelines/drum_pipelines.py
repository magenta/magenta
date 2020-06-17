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

"""Data processing pipelines for drum tracks."""

from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from note_seq import drums_lib
from note_seq import DrumTrack
from note_seq import events_lib
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf


def extract_drum_tracks(quantized_sequence,
                        search_start_step=0,
                        min_bars=7,
                        max_steps_truncate=None,
                        max_steps_discard=None,
                        gap_bars=1.0,
                        pad_end=False,
                        ignore_is_drum=False):
  """Extracts a list of drum tracks from the given quantized NoteSequence.

  This function will search through `quantized_sequence` for drum tracks. A drum
  track can span multiple "tracks" in the sequence. Only one drum track can be
  active at a given time, but multiple drum tracks can be extracted from the
  sequence if gaps are present.

  Once a note-on drum event is encountered, a drum track begins. Gaps of silence
  will be splitting points that divide the sequence into separate drum tracks.
  The minimum size of these gaps are given in `gap_bars`. The size of a bar
  (measure) of music in time steps is computed form the time signature stored in
  `quantized_sequence`.

  A drum track is only used if it is at least `min_bars` bars long.

  After scanning the quantized NoteSequence, a list of all extracted DrumTrack
  objects is returned.

  Args:
    quantized_sequence: A quantized NoteSequence.
    search_start_step: Start searching for drums at this time step. Assumed to
        be the beginning of a bar.
    min_bars: Minimum length of drum tracks in number of bars. Shorter drum
        tracks are discarded.
    max_steps_truncate: Maximum number of steps in extracted drum tracks. If
        defined, longer drum tracks are truncated to this threshold. If pad_end
        is also True, drum tracks will be truncated to the end of the last bar
        below this threshold.
    max_steps_discard: Maximum number of steps in extracted drum tracks. If
        defined, longer drum tracks are discarded.
    gap_bars: A drum track comes to an end when this number of bars (measures)
        of no drums is encountered.
    pad_end: If True, the end of the drum track will be padded with empty events
        so that it will end at a bar boundary.
    ignore_is_drum: Whether accept notes where `is_drum` is False.

  Returns:
    drum_tracks: A python list of DrumTrack instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.

  Raises:
    NonIntegerStepsPerBarError: If `quantized_sequence`'s bar length
        (derived from its time signature) is not an integer number of time
        steps.
  """
  drum_tracks = []
  stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
               ['drum_tracks_discarded_too_short',
                'drum_tracks_discarded_too_long', 'drum_tracks_truncated'])
  # Create a histogram measuring drum track lengths (in bars not steps).
  # Capture drum tracks that are very small, in the range of the filter lower
  # bound `min_bars`, and large. The bucket intervals grow approximately
  # exponentially.
  stats['drum_track_lengths_in_bars'] = statistics.Histogram(
      'drum_track_lengths_in_bars',
      [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, min_bars // 2, min_bars,
       min_bars + 1, min_bars - 1])

  steps_per_bar = int(
      sequences_lib.steps_per_bar_in_quantized_sequence(quantized_sequence))

  # Quantize the track into a DrumTrack object.
  # If any notes start at the same time, only one is kept.
  while 1:
    drum_track = DrumTrack()
    drum_track.from_quantized_sequence(
        quantized_sequence,
        search_start_step=search_start_step,
        gap_bars=gap_bars,
        pad_end=pad_end,
        ignore_is_drum=ignore_is_drum)
    search_start_step = (
        drum_track.end_step +
        (search_start_step - drum_track.end_step) % steps_per_bar)
    if not drum_track:
      break

    # Require a certain drum track length.
    if len(drum_track) < drum_track.steps_per_bar * min_bars:
      stats['drum_tracks_discarded_too_short'].increment()
      continue

    # Discard drum tracks that are too long.
    if max_steps_discard is not None and len(drum_track) > max_steps_discard:
      stats['drum_tracks_discarded_too_long'].increment()
      continue

    # Truncate drum tracks that are too long.
    if max_steps_truncate is not None and len(drum_track) > max_steps_truncate:
      truncated_length = max_steps_truncate
      if pad_end:
        truncated_length -= max_steps_truncate % drum_track.steps_per_bar
      drum_track.set_length(truncated_length)
      stats['drum_tracks_truncated'].increment()

    stats['drum_track_lengths_in_bars'].increment(
        len(drum_track) // drum_track.steps_per_bar)

    drum_tracks.append(drum_track)

  return drum_tracks, stats.values()


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
      # pylint has a false positive error on this method call for some reason.
      # pylint:disable=redundant-keyword-arg
      drum_tracks, stats = extract_drum_tracks(
          quantized_sequence,
          min_bars=self._min_bars,
          max_steps_truncate=self._max_steps,
          gap_bars=self._gap_bars)
      # pylint:enable=redundant-keyword-arg
    except events_lib.NonIntegerStepsPerBarError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      drum_tracks = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    self._set_stats(stats)
    return drum_tracks
