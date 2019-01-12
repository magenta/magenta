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

import collections
import functools
import numpy as np

from magenta.models.gansynth.lib.datasets import dataset_nsynth_tfrecord

Counter = collections.Counter


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


def get_pitch_counts(dataset_name):
  """Gets dictionary of pitch counts."""
  registry = {
      'nsynth_tfrecord': dataset_nsynth_tfrecord.PITCH_COUNTS,
  }
  if dataset_name not in registry.keys():
    raise ValueError('Unsupported dataset: {}'.format(dataset_name))
  return registry[dataset_name]


def get_pitches(num_samples, dataset_name):
  """Returns pitch_counter for num_samples for given dataset."""
  all_pitches = []
  pitch_counts = get_pitch_counts(dataset_name=dataset_name)
  for k, v in pitch_counts.items():
    all_pitches.extend([k]*v)
  sample_pitches = np.random.choice(all_pitches, num_samples)
  pitch_counter = Counter(sample_pitches)
  return pitch_counter
