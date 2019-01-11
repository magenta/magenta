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
"""Providing training data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from magenta.models.gansynth.lib.datasets import dataset_nsynth_tfrecord


def _provide_test_one_hot_labels(batch_size):
  return tf.random_normal([batch_size, 10])


def _provide_test_audio_dataset(length=16000, channels=1):
  """Provides test dataset of audios."""
  waves = tf.random_normal([1000, length, channels])
  one_hot_labels = tf.random_normal([1000, 10])
  dataset = tf.data.Dataset.from_tensor_slices((waves, one_hot_labels))
  dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=100))
  return dataset


def provide_one_hot_labels(name, **kwargs):
  """Provides one hot labels."""
  fn_dict = {
      'nsynth_tfrecord':
          functools.partial(dataset_nsynth_tfrecord.provide_one_hot_labels,
                            **kwargs),
  }
  fn = fn_dict.get(name)
  if fn is None:
    raise ValueError('Unsupported dataset: {}'.format(name))
  return fn()


def provide_dataset(name, **kwargs):
  """Provides dataset."""
  fn_dict = {
      'nsynth_tfrecord':
          functools.partial(dataset_nsynth_tfrecord.provide_audio_dataset,
                            **kwargs),
  }
  fn = fn_dict.get(name)
  if fn is None:
    raise ValueError('Unsupported dataset: {}'.format(name))
  return fn()


def get_pitch_counts(name):
  """Gets dictionary of pitch counts."""
  registry = {
      'nsynth_tfrecord': dataset_nsynth_tfrecord.PITCH_COUNTS,
  }
  if name not in registry.keys():
    raise ValueError('Unsupported dataset: {}'.format(name))
  return registry[name]
