# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Dataset loading code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from magenta.models.gansynth.lib import spectral_ops


def _get_dataset_from_file_pattern(file_pattern):
  dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))
  dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=20, sloppy=True))
  return dataset


def provide_one_hot_labels(pitch_counts, batch_size):
  """Provides one hot labels."""
  pitches = sorted(pitch_counts.keys())
  counts = [pitch_counts[p] for p in pitches]
  indices = tf.reshape(
      tf.multinomial(tf.log([tf.to_float(counts)]), batch_size), [batch_size])
  one_hot_labels = tf.one_hot(indices, depth=len(pitches))
  return one_hot_labels


def provide_audio_dataset(file_pattern, pitches, length, channels):
  """Provides dataset of nsynth audios."""
  label_index_table = tf.contrib.lookup.index_table_from_tensor(
      sorted(pitches), dtype=tf.int64)

  def _parse_nsynth(record):
    """Parsing function for NSynth dataset."""
    features = {
        'pitch': tf.FixedLenFeature([1], dtype=tf.int64),
        'audio': tf.FixedLenFeature([64000], dtype=tf.float32),
        'qualities': tf.FixedLenFeature([10], dtype=tf.int64),
        'instrument_source': tf.FixedLenFeature([1], dtype=tf.int64),
        'instrument_family': tf.FixedLenFeature([1], dtype=tf.int64),
    }

    example = tf.parse_single_example(record, features)
    wave, label = example['audio'], example['pitch']
    wave = spectral_ops.crop_or_pad(wave[tf.newaxis, :, tf.newaxis],
                                    length,
                                    channels)[0]
    one_hot_label = tf.one_hot(
        label_index_table.lookup(label), depth=len(pitches))[0]
    return wave, one_hot_label, label

  dataset = _get_dataset_from_file_pattern(file_pattern)
  dataset = dataset.map(_parse_nsynth, num_parallel_calls=4)

  # Filter just pitches 24-84
  dataset = dataset.filter(lambda w, l, p: tf.greater_equal(p, 24)[0])
  dataset = dataset.filter(lambda w, l, p: tf.less_equal(p, 84)[0])
  dataset = dataset.map(lambda w, l, p: (w, l))
  return dataset

