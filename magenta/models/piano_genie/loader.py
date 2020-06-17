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

"""Loads music data from TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from note_seq.protobuf import music_pb2
import numpy as np
import tensorflow.compat.v1 as tf


def load_noteseqs(fp,
                  batch_size,
                  seq_len,
                  max_discrete_times=None,
                  max_discrete_velocities=None,
                  augment_stretch_bounds=None,
                  augment_transpose_bounds=None,
                  randomize_chord_order=False,
                  repeat=False,
                  buffer_size=512):
  """Loads random subsequences from NoteSequences in TFRecords.

  Args:
    fp: List of shard fps.
    batch_size: Number of sequences in batch.
    seq_len: Length of subsequences.
    max_discrete_times: Maximum number of time buckets at 31.25Hz.
    max_discrete_velocities: Maximum number of velocity buckets.
    augment_stretch_bounds: Tuple containing speed ratio range.
    augment_transpose_bounds: Tuple containing semitone augmentation range.
    randomize_chord_order: If True, list notes of chord in random order.
    repeat: If True, continuously loop through records.
    buffer_size: Size of random queue.

  Returns:
    A dict containing the loaded tensor subsequences.

  Raises:
    ValueError: Invalid file format for shard filepaths.
  """

  # Deserializes NoteSequences and extracts numeric tensors
  def _str_to_tensor(note_sequence_str,
                     augment_stretch_bounds=None,
                     augment_transpose_bounds=None):
    """Converts a NoteSequence serialized proto to arrays."""
    note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)

    note_sequence_ordered = list(note_sequence.notes)

    if randomize_chord_order:
      random.shuffle(note_sequence_ordered)
      note_sequence_ordered = sorted(
          note_sequence_ordered, key=lambda n: n.start_time)
    else:
      note_sequence_ordered = sorted(
          note_sequence_ordered, key=lambda n: (n.start_time, n.pitch))

    # Transposition data augmentation
    if augment_transpose_bounds is not None:
      transpose_factor = np.random.randint(*augment_transpose_bounds)

      for note in note_sequence_ordered:
        note.pitch += transpose_factor

    note_sequence_ordered = [
        n for n in note_sequence_ordered if (n.pitch >= 21) and (n.pitch <= 108)
    ]

    pitches = np.array([note.pitch for note in note_sequence_ordered])
    velocities = np.array([note.velocity for note in note_sequence_ordered])
    start_times = np.array([note.start_time for note in note_sequence_ordered])
    end_times = np.array([note.end_time for note in note_sequence_ordered])

    # Tempo data augmentation
    if augment_stretch_bounds is not None:
      stretch_factor = np.random.uniform(*augment_stretch_bounds)
      start_times *= stretch_factor
      end_times *= stretch_factor

    if note_sequence_ordered:
      # Delta time start high to indicate free decision
      delta_times = np.concatenate([[100000.],
                                    start_times[1:] - start_times[:-1]])
    else:
      delta_times = np.zeros_like(start_times)

    return note_sequence_str, np.stack(
        [pitches, velocities, delta_times, start_times, end_times],
        axis=1).astype(np.float32)

  # Filter out excessively short examples
  def _filter_short(note_sequence_tensor, seq_len):
    note_sequence_len = tf.shape(note_sequence_tensor)[0]
    return tf.greater_equal(note_sequence_len, seq_len)

  # Take a random crop of a note sequence
  def _random_crop(note_sequence_tensor, seq_len):
    note_sequence_len = tf.shape(note_sequence_tensor)[0]
    start_max = note_sequence_len - seq_len
    start_max = tf.maximum(start_max, 0)

    start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)
    seq = note_sequence_tensor[start:start + seq_len]

    return seq

  # Find sharded filenames
  filenames = tf.gfile.Glob(fp)

  # Create dataset
  dataset = tf.data.TFRecordDataset(filenames)

  # Deserialize protos
  # pylint: disable=g-long-lambda
  dataset = dataset.map(
      lambda data: tf.py_func(
          lambda x: _str_to_tensor(
              x, augment_stretch_bounds, augment_transpose_bounds),
          [data], (tf.string, tf.float32), stateful=False))
  # pylint: enable=g-long-lambda

  # Filter sequences that are too short
  dataset = dataset.filter(lambda s, n: _filter_short(n, seq_len))

  # Get random crops
  dataset = dataset.map(lambda s, n: (s, _random_crop(n, seq_len)))

  # Shuffle
  if repeat:
    dataset = dataset.shuffle(buffer_size=buffer_size)

  # Make batches
  dataset = dataset.batch(batch_size, drop_remainder=True)

  # Repeat
  if repeat:
    dataset = dataset.repeat()

  # Get tensors
  iterator = dataset.make_one_shot_iterator()
  note_sequence_strs, note_sequence_tensors = iterator.get_next()

  # Set shapes
  note_sequence_strs.set_shape([batch_size])
  note_sequence_tensors.set_shape([batch_size, seq_len, 5])

  # Retrieve tensors
  note_pitches = tf.cast(note_sequence_tensors[:, :, 0] + 1e-4, tf.int32)
  note_velocities = tf.cast(note_sequence_tensors[:, :, 1] + 1e-4, tf.int32)
  note_delta_times = note_sequence_tensors[:, :, 2]
  note_start_times = note_sequence_tensors[:, :, 3]
  note_end_times = note_sequence_tensors[:, :, 4]

  # Onsets and frames model samples at 31.25Hz
  note_delta_times_int = tf.cast(
      tf.round(note_delta_times * 31.25) + 1e-4, tf.int32)

  # Reduce time discretizations to a fixed number of buckets
  if max_discrete_times is not None:
    note_delta_times_int = tf.minimum(note_delta_times_int, max_discrete_times)

  # Quantize velocities
  if max_discrete_velocities is not None:
    note_velocities = tf.minimum(
        note_velocities / (128 // max_discrete_velocities),
        max_discrete_velocities)

  # Build return dict
  note_tensors = {
      "pb_strs": note_sequence_strs,
      "midi_pitches": note_pitches,
      "velocities": note_velocities,
      "delta_times": note_delta_times,
      "delta_times_int": note_delta_times_int,
      "start_times": note_start_times,
      "end_times": note_end_times
  }

  return note_tensors
