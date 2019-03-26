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

"""Utilities for fine alignment.

CQT calculations and NoteSequence manipulations are done in Python. For speed,
DTW calculations are done in C++ by calling the 'align' program, which is
specifically intended to be used with this library. Communication between
Python and C++ is done with a protobuf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import tempfile

from absl import logging
import alignment_pb2
import librosa
from magenta.music import midi_synth
from magenta.music import sequences_lib
import numpy as np


# Constants based on craffel's example alignment script:
# https://github.com/craffel/pretty-midi/blob/master/examples/align_midi.py

SAMPLE_RATE = 22050
CQT_HOP_LENGTH_FINE = 64  # ~3ms
CQT_N_BINS = 48
CQT_BINS_PER_OCTAVE = 12
CQT_FMIN = librosa.midi_to_hz(36)

ALIGN_BINARY = './align'


def extract_cqt(samples, sample_rate, cqt_hop_length):
  """Transforms the contents of a wav/mp3 file into a series of CQT frames."""
  cqt = np.abs(librosa.core.cqt(
      samples,
      sample_rate,
      hop_length=cqt_hop_length,
      fmin=CQT_FMIN,
      n_bins=CQT_N_BINS,
      bins_per_octave=CQT_BINS_PER_OCTAVE), dtype=np.float32)

  # Compute log-amplitude
  cqt = librosa.power_to_db(cqt)
  return cqt


def align_cpp(samples,
              sample_rate,
              ns,
              cqt_hop_length,
              sf2_path,
              penalty_mul=1.0,
              band_radius_seconds=.5):
  """Aligns the notesequence to the wav file using C++ DTW.

  Args:
    samples: Samples to align.
    sample_rate: Sample rate for samples.
    ns: The source notesequence to align.
    cqt_hop_length: Hop length to use for CQT calculations.
    sf2_path: Path to SF2 file for synthesis.
    penalty_mul: Penalty multiplier to use for non-diagonal moves.
    band_radius_seconds: What size of band radius to use for restricting DTW.

  Raises:
    RuntimeError: If notes are skipped during alignment.

  Returns:
    samples: The samples used from the wav file.
    aligned_ns: The aligned version of the notesequence.
    remaining_ns: Any remaining notesequence that extended beyond the length
        of the wav file.
  """
  logging.info('Synthesizing')
  ns_samples = midi_synth.fluidsynth(
      ns, sf2_path=sf2_path, sample_rate=sample_rate).astype(np.float32)

  # It is critical that ns_samples and samples are the same length because the
  # alignment code does not do subsequence alignment.
  ns_samples = np.pad(ns_samples,
                      (0, max(0, samples.shape[0] - ns_samples.shape[0])),
                      'constant')

  # Pad samples too, if needed, because there are some cases where the
  # synthesized NoteSequence is actually longer.
  samples = np.pad(samples,
                   (0, max(0, ns_samples.shape[0] - samples.shape[0])),
                   'constant')

  # Note that we skip normalization here becasue it happens in C++.
  logging.info('source_cqt')
  source_cqt = extract_cqt(ns_samples, sample_rate, cqt_hop_length)

  logging.info('dest_cqt')
  dest_cqt = extract_cqt(samples, sample_rate, cqt_hop_length)

  alignment_task = alignment_pb2.AlignmentTask()
  alignment_task.sequence_1.x = source_cqt.shape[0]
  alignment_task.sequence_1.y = source_cqt.shape[1]
  for c in source_cqt.reshape([-1]):
    alignment_task.sequence_1.content.append(c)

  alignment_task.sequence_2.x = dest_cqt.shape[0]
  alignment_task.sequence_2.y = dest_cqt.shape[1]
  for c in dest_cqt.reshape([-1]):
    alignment_task.sequence_2.content.append(c)

  seconds_per_frame = cqt_hop_length / sample_rate

  alignment_task.band_radius = int(band_radius_seconds / seconds_per_frame)
  alignment_task.penalty = 0
  alignment_task.penalty_mul = penalty_mul

  # Write to file.
  fh, temp_path = tempfile.mkstemp(suffix='.proto')
  os.close(fh)
  with open(temp_path, 'w') as f:
    f.write(alignment_task.SerializeToString())

  # Align with C++ program.
  subprocess.check_call([ALIGN_BINARY, temp_path])

  # Read file.
  with open(temp_path + '.result') as f:
    result = alignment_pb2.AlignmentResult.FromString(f.read())

  # Clean up.
  os.remove(temp_path)
  os.remove(temp_path + '.result')

  logging.info('Aligning NoteSequence with warp path.')

  warp_seconds_i = np.array([i * seconds_per_frame for i in result.i])
  warp_seconds_j = np.array([j * seconds_per_frame for j in result.j])

  time_diffs = np.abs(warp_seconds_i - warp_seconds_j)
  warps = np.abs(time_diffs[1:] - time_diffs[:-1])

  stats = {
      'alignment_score': result.score,
      'warp_mean_s': np.mean(warps),
      'warp_median_s': np.median(warps),
      'warp_max_s': np.max(warps),
      'warp_min_s': np.min(warps),
      'time_diff_mean_s': np.mean(time_diffs),
      'time_diff_median_s': np.median(time_diffs),
      'time_diff_max_s': np.max(time_diffs),
      'time_diff_min_s': np.min(time_diffs),
  }

  for name, value in sorted(stats.iteritems()):
    logging.info('%s: %f', name, value)

  aligned_ns, skipped_notes = sequences_lib.adjust_notesequence_times(
      ns,
      lambda t: np.interp(t, warp_seconds_i, warp_seconds_j),
      minimum_duration=seconds_per_frame)
  if skipped_notes > 0:
    raise RuntimeError('Skipped {} notes'.format(skipped_notes))

  logging.debug('done')

  return aligned_ns, stats
