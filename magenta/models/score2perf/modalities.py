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

"""Modality transformations used by Magenta and not in core Tensor2Tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf


def _get_weights(model_hparams, vocab_size, hidden_dim=None):
  """Copied from tensor2tensor/layers/modalities.py but uses total vocab."""
  if hidden_dim is None:
    hidden_dim = model_hparams.hidden_size
  num_shards = model_hparams.symbol_modality_num_shards
  shards = []
  for i in range(num_shards):
    shard_size = (sum(vocab_size) // num_shards) + (
        1 if i < sum(vocab_size) % num_shards else 0)
    var_name = 'weights_%d' % i
    shards.append(
        tf.get_variable(
            var_name, [shard_size, hidden_dim],
            initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5)))
  if num_shards == 1:
    ret = shards[0]
  else:
    ret = tf.concat(shards, 0)
  # Convert ret to tensor.
  if not tf.executing_eagerly():
    ret = common_layers.convert_gradient_to_tensor(ret)
  return ret


def bottom_simple(x, model_hparams, vocab_size, name, reuse):
  """Internal bottom transformation."""
  with tf.variable_scope(name, reuse=reuse):
    var = _get_weights(model_hparams, vocab_size)
    x = common_layers.dropout_no_scaling(
        x, 1.0 - model_hparams.symbol_dropout)
    # Add together the embeddings for each tuple position.
    ret = tf.add_n([
        tf.gather(var, x[:, :, :, i] + sum(vocab_size[:i])) *
        tf.expand_dims(tf.to_float(tf.not_equal(x[:, :, :, i], 0)), -1)
        for i in range(len(vocab_size))
    ])
    if model_hparams.multiply_embedding_mode == 'sqrt_depth':
      ret *= model_hparams.hidden_size**0.5
    return ret


def bottom(x, model_hparams, vocab_size):
  """Bottom transformation for tuples of symbols.

  Like tensor2tensor.modalities.symbol_bottom but operates on tuples of
  symbols. Each tuple position uses its own vocabulary.

  Args:
    x: Tensor with shape [batch, ...].
    model_hparams: tf.contrib.training.HParams, model hyperparmeters.
    vocab_size: list of int, vocabulary sizes.

  Returns:
    Tensor.
  """
  if model_hparams.shared_embedding_and_softmax_weights:
    return bottom_simple(x, model_hparams, vocab_size, 'shared', reuse=None)
  return bottom_simple(x, model_hparams, vocab_size, 'input_emb', reuse=None)
