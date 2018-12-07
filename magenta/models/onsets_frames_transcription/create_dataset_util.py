# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Create the tfrecord files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import math
import numpy as np

from magenta.music import sequences_lib


def _find_inactive_ranges(note_sequence):
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

  ranges_nosustain = _find_inactive_ranges(note_sequence)
  ranges_sustain = _find_inactive_ranges(note_sequence_sustain)

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
            samples,
            int(math.floor(zxc_start * sample_rate)),
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
