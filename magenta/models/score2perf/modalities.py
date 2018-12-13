# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Modalities used by Magenta and not in core Tensor2Tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import modality

import tensorflow as tf


class SymbolTupleModality(modality.Modality):
  """Modality for tuples of discrete symbols.

  Like tensor2tensor.modalities.SymbolModality but operates on tuples of
  symbols. Each tuple position uses its own vocabulary. Currently only inputs
  ("bottom") are supported.
  """

  def __init__(self, model_hparams, vocab_size=None):
    """Override to support tuple vocabulary."""
    # pylint: disable=super-init-not-called
    self._model_hparams = model_hparams
    if vocab_size is not None and hasattr(model_hparams, 'vocab_divisor'):
      # Extend the vocabulary of the last tuple element.
      vocab_size[-1] += (0 - vocab_size[-1]) % model_hparams.vocab_divisor
    self._vocab_size = vocab_size

  @property
  def top_dimensionality(self):
    """Override to use total vocabulary size."""
    return sum(self._vocab_size)

  @property
  def top_is_pointwise(self):
    return True

  @property
  def name(self):
    return 'symbol_tuple_modality_%s_%d' % (
        '_'.join('%d' % s for s in self._vocab_size), self._body_input_depth)

  def _get_weights(self, hidden_dim=None):
    """Copied from tensor2tensor/layers/modalities.py but uses total vocab."""
    if hidden_dim is None:
      hidden_dim = self._body_input_depth
    num_shards = self._model_hparams.symbol_modality_num_shards
    shards = []
    for i in range(num_shards):
      shard_size = (sum(self._vocab_size) // num_shards) + (
          1 if i < sum(self._vocab_size) % num_shards else 0)
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
    if not tf.contrib.eager.in_eager_mode():
      ret = common_layers.convert_gradient_to_tensor(ret)
    return ret

  def bottom_simple(self, x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
      var = self._get_weights()
      x = common_layers.dropout_no_scaling(
          x, 1.0 - self._model_hparams.symbol_dropout)
      # Add together the embeddings for each tuple position.
      ret = tf.add_n([
          tf.gather(var, x[:, :, :, i] + sum(self._vocab_size[:i])) *
          tf.expand_dims(tf.to_float(tf.not_equal(x[:, :, :, i], 0)), -1)
          for i in range(len(self._vocab_size))
      ])
      if self._model_hparams.multiply_embedding_mode == 'sqrt_depth':
        ret *= self._body_input_depth**0.5
      return ret

  def bottom(self, x):
    if self._model_hparams.shared_embedding_and_softmax_weights:
      return self.bottom_simple(x, 'shared', reuse=None)
    return self.bottom_simple(x, 'input_emb', reuse=None)
