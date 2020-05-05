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

# Lint as: python3
"""Data normalizer."""

import io
import os

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf


def _range_normalizer(x, margin):
  x = x.flatten()
  min_x = np.min(x)
  max_x = np.max(x)
  a = margin * (2.0 / (max_x - min_x))
  b = margin * (-2.0 * min_x / (max_x - min_x) - 1.0)
  return a, b


class DataNormalizer(object):
  """A class to normalize data."""

  def __init__(self, config, file_name):
    self._work_dir = os.path.join(config['train_root_dir'], 'assets')
    self._margin = config['normalizer_margin']
    self._path = os.path.join(self._work_dir, file_name)
    self._done_path = os.path.join(self._work_dir, 'DONE_' + file_name)
    self._num_examples = config['normalizer_num_examples']

  def _run_data(self, data):
    """Runs data in session to get data_np."""
    if data is None:
      return None
    data_np = []
    count_examples = 0
    with tf.MonitoredTrainingSession() as sess:
      while count_examples < self._num_examples:
        out = sess.run(data)
        data_np.append(out)
        count_examples += out.shape[0]
    data_np = np.concatenate(data_np, axis=0)
    return data_np

  def compute(self, data_np):
    """Computes normalizer."""
    raise NotImplementedError

  def exists(self):
    return tf.gfile.Exists(self._done_path)

  def save(self, data):
    """Computes and saves normalizer."""
    if self.exists():
      logging.info('Skip save() as %s already exists', self._done_path)
      return
    data_np = self._run_data(data)
    normalizer = self.compute(data_np)
    logging.info('Save normalizer to %s', self._path)
    bytes_io = io.BytesIO()
    np.savez(bytes_io, normalizer=normalizer)
    if not tf.gfile.Exists(self._work_dir):
      tf.gfile.MakeDirs(self._work_dir)
    with tf.gfile.Open(self._path, 'wb') as f:
      f.write(bytes_io.getvalue())
    with tf.gfile.Open(self._done_path, 'w') as f:
      f.write('')
    return normalizer

  def load(self):
    """Loads normalizer."""
    logging.info('Load data from %s', self._path)
    with tf.gfile.Open(self._path, 'rb') as f:
      result = np.load(f)
      return result['normalizer']

  def normalize_op(self, x):
    raise NotImplementedError

  def denormalize_op(self, x):
    raise NotImplementedError


class NoneNormalizer(object):
  """A dummy class that does not normalize data."""

  def __init__(self, unused_config=None):
    pass

  def save(self, data):
    pass

  def load(self):
    pass

  def exists(self):
    return True

  def normalize_op(self, x):
    return x

  def denormalize_op(self, x):
    return x


class SpecgramsPrespecifiedNormalizer(object):
  """A class that uses prespecified normalization data."""

  def __init__(self, config):
    m_a = config['mag_normalizer_a']
    m_b = config['mag_normalizer_b']
    p_a = config['p_normalizer_a']
    p_b = config['p_normalizer_b']
    self._a = np.asarray([m_a, p_a])[None, None, None, :]
    self._b = np.asarray([m_b, p_b])[None, None, None, :]

  def exists(self):
    return True

  def save(self, data):
    pass

  def load(self):
    pass

  def normalize_op(self, x):
    return tf.clip_by_value(self._a * x + self._b, -1.0, 1.0)

  def denormalize_op(self, x):
    return (x - self._b) / self._a


class SpecgramsSimpleNormalizer(DataNormalizer):
  """A class to normalize specgrams for each channel."""

  def __init__(self, config):
    super(SpecgramsSimpleNormalizer, self).__init__(
        config, 'specgrams_simple_normalizer.npz')

  def compute(self, data_np):
    m_a, m_b = _range_normalizer(data_np[:, :, :, 0], self._margin)
    p_a, p_b = _range_normalizer(data_np[:, :, :, 1], self._margin)
    return np.asarray([m_a, m_b, p_a, p_b])

  def load_and_decode(self):
    m_a, m_b, p_a, p_b = self.load()
    a = np.asarray([m_a, p_a])[None, None, None, :]
    b = np.asarray([m_b, p_b])[None, None, None, :]
    return a, b

  def normalize_op(self, x):
    a, b = self.load_and_decode()
    a = tf.constant(a, dtype=x.dtype)
    b = tf.constant(b, dtype=x.dtype)
    return tf.clip_by_value(a * x + b, -1.0, 1.0)

  def denormalize_op(self, x):
    a, b = self.load_and_decode()
    a = tf.constant(a, dtype=x.dtype)
    b = tf.constant(b, dtype=x.dtype)
    return (x - b) / a


class SpecgramsFreqNormalizer(DataNormalizer):
  """A class to normalize specgrams for each freq bin, channel."""

  def __init__(self, config):
    super(SpecgramsFreqNormalizer, self).__init__(
        config, 'specgrams_freq_normalizer.npz')

  def compute(self, data_np):
    # data_np: [N, time, freq, channels]
    normalizer = []
    for f in range(data_np.shape[2]):
      m_a, m_b = _range_normalizer(data_np[:, :, f, 0], self._margin)
      p_a, p_b = _range_normalizer(data_np[:, :, f, 1], self._margin)
      normalizer.append([m_a, m_b, p_a, p_b])
    return np.asarray(normalizer)

  def load_and_decode(self):
    normalizer = self.load()
    m_a = normalizer[:, 0][None, None, :, None]
    m_b = normalizer[:, 1][None, None, :, None]
    p_a = normalizer[:, 2][None, None, :, None]
    p_b = normalizer[:, 3][None, None, :, None]
    a = np.concatenate([m_a, p_a], axis=-1)
    b = np.concatenate([m_b, p_b], axis=-1)
    return a, b

  def normalize_op(self, x):
    a, b = self.load_and_decode()
    a = tf.constant(a, dtype=x.dtype)
    b = tf.constant(b, dtype=x.dtype)
    return tf.clip_by_value(a * x + b, -1.0, 1.0)

  def denormalize_op(self, x):
    a, b = self.load_and_decode()
    a = tf.constant(a, dtype=x.dtype)
    b = tf.constant(b, dtype=x.dtype)
    return (x - b) / a


registry = {
    'none': NoneNormalizer,
    'specgrams_prespecified_normalizer': SpecgramsPrespecifiedNormalizer,
    'specgrams_simple_normalizer': SpecgramsSimpleNormalizer,
    'specgrams_freq_normalizer': SpecgramsFreqNormalizer
}
