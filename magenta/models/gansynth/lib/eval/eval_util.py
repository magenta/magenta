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
"""Evaluation utililty functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import scipy.io.wavfile
import tensorflow as tf

from magenta.models.gansynth.lib import datasets

Counter = collections.Counter


def generate_samples_from_file(samples_path,
                               dataset_name='full',
                               with_replacement=False):
  """Wrapper for generate_samples from samples.

  Args:
    samples_path: (string) directory for samples from model.
    dataset_name: (string) name of dataset to get pitch distribution.
    with_replacement: (boolean) whether generate samples with replacement.

  Returns:
    function for generating samples give pitch, num_samples.

  Raises:
    ValueError: if samples_path not specified
  """

  if samples_path is None:
    raise ValueError('Needs directory with samples.')
  pitch_counts = datasets.get_pitch_counts(dataset_name=dataset_name)
  pitch_to_files = {}
  for pitch in pitch_counts:
    samples_dir = os.path.join(samples_path, 'pitch_{}/*'.format(pitch))
    pitch_to_files[pitch] = tf.gfile.Glob(samples_dir)
    np.random.shuffle(pitch_to_files[pitch])
    print('Read samples for pitch %s' % pitch)

  def generate_samples_without_replacement(num_samples, pitch=None):
    """Wrap generate_samples for files."""
    if pitch is not None:
      if len(pitch_to_files[pitch]) < num_samples:
        raise Exception('not enough sample files')
      sample_files = pitch_to_files[pitch][:num_samples]
      pitch_to_files[pitch] = pitch_to_files[pitch][num_samples:]
    else:
      all_pitches = []
      pitch_counts = datasets.get_pitch_counts(dataset_name=dataset_name)
      for k, v in pitch_counts.items():
        all_pitches.extend([k]*v)
      sample_pitches = np.random.choice(all_pitches, num_samples)
      sample_files = []
      for pitch in sample_pitches:
        if not pitch_to_files[pitch]:
          raise Exception('not enough sample files')
        sample_files.append(pitch_to_files[pitch].pop())
    return read_samples_from_files(sample_files)

  def generate_samples_with_replacement(num_samples, pitch=None):
    """Wrap generate_samples for files."""
    if pitch is not None:
      sample_files = np.random.choice(pitch_to_files[pitch], num_samples)
    else:
      all_pitches = []
      pitch_counts = datasets.get_pitch_counts(dataset_name=dataset_name)
      for k, v in pitch_counts.items():
        all_pitches.extend([k]*v)
      sample_pitches = np.random.choice(all_pitches, num_samples)
      pitch_counter = Counter(sample_pitches)
      sample_files = []
      for pitch_key, pitch_value in pitch_counter.items():
        sample_files.append(np.random.choice(pitch_to_files[pitch_key],
                                             pitch_value))
    return read_samples_from_files(sample_files)

  if with_replacement:
    return generate_samples_with_replacement
  return generate_samples_without_replacement


def read_samples_from_files(sample_files):
  samples = []
  for sample_file in sample_files:
    with tf.gfile.GFile(sample_file, 'r') as f:
      _, sample = scipy.io.wavfile.read(f)
    samples.append(sample)
  return np.stack(samples, axis=0)


def write_samples_to_file(pitch, samples, samples_path):
  samples_dir = os.path.join(samples_path, 'pitch_{}'.format(pitch))
  if not tf.gfile.Exists(samples_dir):
    tf.gfile.MakeDirs(samples_dir)
  for idx, sample in enumerate(samples):
    sample_file = os.path.join(samples_dir, 'sample_{}.wav'.format(idx))
    with tf.gfile.GFile(sample_file, 'w') as f:
      scipy.io.wavfile.write(f, 16000, sample.astype('float32'))


