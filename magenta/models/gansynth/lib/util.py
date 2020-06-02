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
"""Useful functions."""

import io
import os

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim


def get_default_embedding_size(num_features):
  return min(int(np.round(6 * (num_features**0.25))), num_features)


def one_hot_to_embedding(one_hot, embedding_size=None):
  """Gets a dense embedding vector from a one-hot encoding."""
  num_tokens = int(one_hot.shape[1])
  label_id = tf.argmax(one_hot, axis=1)
  if embedding_size is None:
    embedding_size = get_default_embedding_size(num_tokens)
  embedding = tf.get_variable(
      'one_hot_embedding', [num_tokens, embedding_size], dtype=tf.float32)
  return tf.nn.embedding_lookup(embedding, label_id, name='token_to_embedding')


def make_ordered_one_hot_vectors(num, num_tokens):
  """Makes one hot vectors of size [num, num_tokens]."""
  num_repeats = int(np.ceil(num / float(num_tokens)))
  indices = tf.stack([tf.range(num_tokens)] * num_repeats)
  indices = tf.reshape(tf.transpose(indices), [-1])[0:num]
  return tf.one_hot(indices, depth=num_tokens)


def make_random_one_hot_vectors(num, num_tokens):
  """Makes random one hot vectors of size [num, num_tokens]."""
  return tf.one_hot(
      tf.random_uniform(shape=(num,), maxval=num_tokens, dtype=tf.int32),
      depth=num_tokens)


def compute_or_load_data(path, compute_data_fn):
  """Computes or loads data."""
  if tf.gfile.Exists(path):
    logging.info('Load data from %s', path)
    with tf.gfile.Open(path, 'rb') as f:
      result = np.load(f)
      return result

  result = compute_data_fn()

  logging.info('Save data to %s', path)
  bytes_io = io.BytesIO()
  np.savez(bytes_io, **result)
  with tf.gfile.Open(path, 'wb') as f:
    f.write(bytes_io.getvalue())
  return result


def compute_data_mean_and_std(data, axis, num_samples):
  """Computes data mean and std."""
  with tf.Session() as sess:
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
        tf.tables_initializer()
    ])
    with tf_slim.queues.QueueRunners(sess):
      data_value = np.concatenate(
          [sess.run(data) for _ in range(num_samples)], axis=0)
  mean = np.mean(data_value, axis=tuple(axis), keepdims=True)
  std = np.std(data_value, axis=tuple(axis), keepdims=True)
  return mean, std


def parse_config_str(config_str):
  """Parses config string.

  For example: config_str = "a=1 b='hello'", the function returns
  {'a': 1, 'b': 'hello'}.

  Args:
    config_str: A config string.
  Returns:
    A dictionary.
  """
  ans = {}
  for line in config_str.split('\n'):
    k, v = line.partition('=')[::2]
    k = k.strip()
    v = v.strip()
    if k and not k.startswith('//'):
      try:
        v = int(v)
      except ValueError:
        try:
          v = float(v)
        except ValueError:
          v = v[1:-1]  # remove quotes for string argument.
      ans[k] = v
  return ans


def expand_path(path):
  return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
