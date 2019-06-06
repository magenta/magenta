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

r"""Utilities for managing wav files and labels for transcription."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import math

import librosa

from magenta.music import audio_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

import numpy as np
import tensorflow as tf


def find_inactive_ranges(note_sequence):
  """Returns ranges where no notes are active in the note_sequence."""
  start_sequence = sorted(
      note_sequence.notes, key=lambda note: note.start_time, reverse=True)
  end_sequence = sorted(
      note_sequence.notes, key=lambda note: note.end_time, reverse=True)

  notes_active = 0

  time = start_sequence[-1].start_time
  inactive_ranges = []
  if time > 0:
    inactive_ranges.append(0.)
    inactive_ranges.append(time)
  start_sequence.pop()
  notes_active += 1
  # Iterate through all note on events
  while start_sequence or end_sequence:
    if start_sequence and (start_sequence[-1].start_time <
                           end_sequence[-1].end_time):
      if notes_active == 0:
        time = start_sequence[-1].start_time
        inactive_ranges.append(time)
      notes_active += 1
      start_sequence.pop()
    else:
      notes_active -= 1
      if notes_active == 0:
        time = end_sequence[-1].end_time
        inactive_ranges.append(time)
      end_sequence.pop()

  # if the last note is the same time as the end, don't add it
  # remove the start instead of creating a sequence with 0 length
  if inactive_ranges[-1] < note_sequence.total_time:
    inactive_ranges.append(note_sequence.total_time)
  else:
    inactive_ranges.pop()

  assert len(inactive_ranges) % 2 == 0

  inactive_ranges = [(inactive_ranges[2 * i], inactive_ranges[2 * i + 1])
                     for i in range(len(inactive_ranges) // 2)]
  return inactive_ranges


def _last_zero_crossing(samples, start, end):
  """Returns the last zero crossing in the window [start, end)."""
  samples_greater_than_zero = samples[start:end] > 0
  samples_less_than_zero = samples[start:end] < 0
  samples_greater_than_equal_zero = samples[start:end] >= 0
  samples_less_than_equal_zero = samples[start:end] <= 0

  # use np instead of python for loop for speed
  xings = np.logical_or(
      np.logical_and(samples_greater_than_zero[:-1],
                     samples_less_than_equal_zero[1:]),
      np.logical_and(samples_less_than_zero[:-1],
                     samples_greater_than_equal_zero[1:])).nonzero()[0]

  return xings[-1] + start if xings.size > 0 else None


def find_split_points(note_sequence, samples, sample_rate, min_length,
                      max_length):
  """Returns times at which there are no notes.

  The general strategy employed is to first check if there are places in the
  sustained pianoroll where no notes are active within the max_length window;
  if so the middle of the last gap is chosen as the split point.

  If not, then it checks if there are places in the pianoroll without sustain
  where no notes are active and then finds last zero crossing of the wav file
  and chooses that as the split point.

  If neither of those is true, then it chooses the last zero crossing within
  the max_length window as the split point.

  If there are no zero crossings in the entire window, then it basically gives
  up and advances time forward by max_length.

  Args:
      note_sequence: The NoteSequence to split.
      samples: The audio file as samples.
      sample_rate: The sample rate (samples/second) of the audio file.
      min_length: Minimum number of seconds in a split.
      max_length: Maximum number of seconds in a split.

  Returns:
      A list of split points in seconds from the beginning of the file.
  """

  if not note_sequence.notes:
    return []

  end_time = note_sequence.total_time

  note_sequence_sustain = sequences_lib.apply_sustain_control_changes(
      note_sequence)

  ranges_nosustain = find_inactive_ranges(note_sequence)
  ranges_sustain = find_inactive_ranges(note_sequence_sustain)

  nosustain_starts = [x[0] for x in ranges_nosustain]
  sustain_starts = [x[0] for x in ranges_sustain]

  nosustain_ends = [x[1] for x in ranges_nosustain]
  sustain_ends = [x[1] for x in ranges_sustain]

  split_points = [0.]

  while end_time - split_points[-1] > max_length:
    max_advance = split_points[-1] + max_length

    # check for interval in sustained sequence
    pos = bisect.bisect_right(sustain_ends, max_advance)
    if pos < len(sustain_starts) and max_advance > sustain_starts[pos]:
      split_points.append(max_advance)

    # if no interval, or we didn't fit, try the unmodified sequence
    elif pos == 0 or sustain_starts[pos - 1] <= split_points[-1] + min_length:
      # no splits available, use non sustain notes and find close zero crossing
      pos = bisect.bisect_right(nosustain_ends, max_advance)

      if pos < len(nosustain_starts) and max_advance > nosustain_starts[pos]:
        # we fit, great, try to split at a zero crossing
        zxc_start = nosustain_starts[pos]
        zxc_end = max_advance
        last_zero_xing = _last_zero_crossing(
            samples, int(math.floor(zxc_start * sample_rate)),
            int(math.ceil(zxc_end * sample_rate)))
        if last_zero_xing:
          last_zero_xing = float(last_zero_xing) / sample_rate
          split_points.append(last_zero_xing)
        else:
          # give up and just return where there are at least no notes
          split_points.append(max_advance)

      else:
        # there are no good places to cut, so just pick the last zero crossing
        # check the entire valid range for zero crossings
        start_sample = int(
            math.ceil((split_points[-1] + min_length) * sample_rate)) + 1
        end_sample = start_sample + (max_length - min_length) * sample_rate
        last_zero_xing = _last_zero_crossing(samples, start_sample, end_sample)

        if last_zero_xing:
          last_zero_xing = float(last_zero_xing) / sample_rate
          split_points.append(last_zero_xing)
        else:
          # give up and advance by max amount
          split_points.append(max_advance)
    else:
      # only advance as far as max_length
      new_time = min(np.mean(ranges_sustain[pos - 1]), max_advance)
      split_points.append(new_time)

  if split_points[-1] != end_time:
    split_points.append(end_time)

  # ensure that we've generated a valid sequence of splits
  for prev, curr in zip(split_points[:-1], split_points[1:]):
    assert curr > prev
    assert curr - prev <= max_length + 1e-8
    if curr < end_time:
      assert curr - prev >= min_length - 1e-8
  assert end_time - split_points[-1] < max_length

  return split_points


def create_example(example_id, ns, wav_data, velocity_range=None):
  """Creates a tf.train.Example proto for training or testing."""
  if velocity_range is None:
    velocities = [note.velocity for note in ns.notes]
    velocity_max = np.max(velocities)
    velocity_min = np.min(velocities)
    velocity_range = music_pb2.VelocityRange(min=velocity_min, max=velocity_max)

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'id':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[example_id.encode('utf-8')])),
              'sequence':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[ns.SerializeToString()])),
              'audio':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(value=[wav_data])),
              'velocity_range':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[velocity_range.SerializeToString()])),
          }))
  return example


def process_record(wav_data,
                   ns,
                   example_id,
                   min_length=5,
                   max_length=20,
                   sample_rate=16000,
                   allow_empty_notesequence=False,
                   load_audio_with_librosa=False):
  """Split a record into chunks and create an example proto.

  To use the full length audio and notesequence, set min_length=0 and
  max_length=-1.

  Args:
    wav_data: audio data in WAV format.
    ns: corresponding NoteSequence.
    example_id: id for the example proto
    min_length: minimum length in seconds for audio chunks.
    max_length: maximum length in seconds for audio chunks.
    sample_rate: desired audio sample rate.
    allow_empty_notesequence: whether an empty NoteSequence is allowed.
    load_audio_with_librosa: Use librosa for sampling. Works with 24-bit wavs.

  Yields:
    Example protos.
  """
  try:
    if load_audio_with_librosa:
      samples = audio_io.wav_data_to_samples_librosa(wav_data, sample_rate)
    else:
      samples = audio_io.wav_data_to_samples(wav_data, sample_rate)
  except audio_io.AudioIOReadError as e:
    print('Exception %s', e)
    return
  samples = librosa.util.normalize(samples, norm=np.inf)
  if max_length == min_length:
    splits = np.arange(0, ns.total_time, max_length)
  elif max_length > 0:
    splits = find_split_points(ns, samples, sample_rate, min_length, max_length)
  else:
    splits = [0, ns.total_time]
  velocities = [note.velocity for note in ns.notes]
  velocity_max = np.max(velocities) if velocities else 0
  velocity_min = np.min(velocities) if velocities else 0
  velocity_range = music_pb2.VelocityRange(min=velocity_min, max=velocity_max)

  for start, end in zip(splits[:-1], splits[1:]):
    if end - start < min_length:
      continue

    if start == 0 and end == ns.total_time:
      new_ns = ns
    else:
      new_ns = sequences_lib.extract_subsequence(ns, start, end)

    if not new_ns.notes and not allow_empty_notesequence:
      tf.logging.warning('skipping empty sequence')
      continue

    if start == 0 and end == ns.total_time:
      new_samples = samples
    else:
      # the resampling that happen in crop_wav_data is really slow
      # and we've already done it once, avoid doing it twice
      new_samples = audio_io.crop_samples(samples, sample_rate, start,
                                          end - start)
    new_wav_data = audio_io.samples_to_wav_data(new_samples, sample_rate)
    yield create_example(
        example_id, new_ns, new_wav_data, velocity_range=velocity_range)
