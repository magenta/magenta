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

"""Module contains a registry of dataset classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from magenta.models.gansynth.lib import spectral_ops
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

Counter = collections.Counter


class BaseDataset(object):
  """A base class for reading data from disk."""

  def __init__(self, config):
    self._config = config

  def provide_one_hot_labels(self, batch_size):
    """Provides one-hot labels."""
    raise NotImplementedError

  def provide_dataset(self):
    """Provides audio dataset."""
    raise NotImplementedError

  def get_pitch_counts(self):
    """Returns a dictionary {pitch value (int): count (int)}."""
    raise NotImplementedError

  def get_pitches(self, num_samples):
    """Returns pitch_counter for num_samples for given dataset."""
    all_pitches = []
    pitch_counts = self.get_pitch_counts()
    for k, v in pitch_counts.items():
      all_pitches.extend([k]*v)
    sample_pitches = np.random.choice(all_pitches, num_samples)
    pitch_counter = Counter(sample_pitches)
    return pitch_counter


class NSynthTfdsDataset(BaseDataset):
  """A dataset for reading NSynth from Tensorflow Datasets (TFDS)."""

  def _get_dataset_from_tfds(self):
    """Loads GANsynth subset of NSynth dataset from TFDS."""
    try:
      dataset = tfds.load(
          'nsynth/gansynth_subset:2.3.*',
          data_dir=self._config['tfds_data_dir'],
          split=tfds.Split.TRAIN,
          download=False)
    except AssertionError as e:
      tf.logging.warning(
          'To train with the NSynth dataset, you must either set the '
          '\tfds_data_dir\' hparam to \'gs://tfds-data/datasets\' (recommended '
          'if running on GCP) or to a local directory where the dataset has '
          'been downloaded. You can download the dataset with the command '
          '`python -m tensorflow_datasets.scripts.download_and_prepare '
          '--datasets=nsynth/gansynth_subset --data_dir=/path/to/local/dir`.')
      raise e
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    return dataset

  def provide_one_hot_labels(self, batch_size):
    """Provides one hot labels."""
    pitch_counts = self.get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    counts = [pitch_counts[p] for p in pitches]
    indices = tf.reshape(
        tf.multinomial(tf.log([tf.to_float(counts)]), batch_size), [batch_size])
    one_hot_labels = tf.one_hot(indices, depth=len(pitches))
    return one_hot_labels

  def provide_dataset(self):
    """Provides dataset (audio, labels) of nsynth."""
    length = 64000
    channels = 1

    pitch_counts = self.get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    label_index_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=pitches,
            values=np.arange(len(pitches)),
            key_dtype=tf.int64,
            value_dtype=tf.int64),
        num_oov_buckets=1)

    def _parse_nsynth(example):
      """Parsing function for NSynth dataset."""
      wave, label = example['audio'], example['pitch']
      wave = spectral_ops.crop_or_pad(wave[tf.newaxis, :, tf.newaxis],
                                      length,
                                      channels)[0]
      one_hot_label = tf.one_hot(
          label_index_table.lookup(label), depth=len(pitches))
      return wave, one_hot_label

    dataset = self._get_dataset_from_tfds()
    dataset = dataset.map(
        _parse_nsynth, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def get_pitch_counts(self):
    pitch_counts = {
        24: 551,
        25: 545,
        26: 561,
        27: 550,
        28: 549,
        29: 578,
        30: 547,
        31: 628,
        32: 626,
        33: 630,
        34: 670,
        35: 668,
        36: 751,
        37: 768,
        38: 788,
        39: 801,
        40: 821,
        41: 871,
        42: 873,
        43: 860,
        44: 873,
        45: 844,
        46: 883,
        47: 865,
        48: 1058,
        49: 1036,
        50: 1094,
        51: 1065,
        52: 1186,
        53: 1200,
        54: 1187,
        55: 1346,
        56: 1273,
        57: 1320,
        58: 1263,
        59: 1283,
        60: 1349,
        61: 1285,
        62: 1281,
        63: 1332,
        64: 1286,
        65: 1301,
        66: 1220,
        67: 1300,
        68: 1209,
        69: 1257,
        70: 1288,
        71: 1223,
        72: 1258,
        73: 1183,
        74: 1178,
        75: 1125,
        76: 1171,
        77: 1089,
        78: 1024,
        79: 1081,
        80: 966,
        81: 1043,
        82: 991,
        83: 959,
        84: 977,
    }
    return pitch_counts


registry = {
    'nsynth_tfds': NSynthTfdsDataset,
}
