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

"""Implementation of a NADE (Neural Autoreressive Distribution Estimator)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def _safe_log(tensor):
  """Lower bounded log function."""
  return tf.log(1e-6 + tensor)


class Nade(object):
  """Neural Autoregressive Distribution Estimator [1].

  [1]: https://arxiv.org/abs/1605.02226

  Args:
    num_dims: The number of binary dimensions for each observation.
    num_hidden: The number of hidden units in the NADE.
    internal_bias: Whether the model should maintain its own bias varaibles.
        Otherwise, external values must be passed to `log_prob` and `sample`.
  """

  def __init__(self, num_dims, num_hidden, internal_bias=False, name='nade'):
    self._num_dims = num_dims
    self._num_hidden = num_hidden

    std = 1.0 / math.sqrt(self._num_dims)
    initializer = tf.truncated_normal_initializer(stddev=std)

    with tf.variable_scope(name):
      # Encoder weights (`V` in [1]).
      self.w_enc = tf.get_variable(
          'w_enc',
          shape=[self._num_dims, 1, self._num_hidden],
          initializer=initializer)
      # Transposed decoder weights (`W'` in [1]).
      self.w_dec_t = tf.get_variable(
          'w_dec_t',
          shape=[self._num_dims, self._num_hidden, 1],
          initializer=initializer)
      # Internal encoder bias term (`b` in [1]). Will be used if external biases
      # are not provided.
      if internal_bias:
        self.b_enc = tf.get_variable(
            'b_enc',
            shape=[1, self._num_hidden],
            initializer=initializer)
      else:
        self.b_enc = None
      # Internal decoder bias term (`c` in [1]). Will be used if external biases
      # are not provided.
      if internal_bias:
        self.b_dec = tf.get_variable(
            'b_dec',
            shape=[1, self._num_dims],
            initializer=initializer)
      else:
        self.b_dec = None

  @property
  def num_hidden(self):
    """The number of hidden units for each input/output of the NADE."""
    return self._num_hidden

  @property
  def num_dims(self):
    """The number of input/output dimensions of the NADE."""
    return self._num_dims

  def log_prob(self, x, b_enc=None, b_dec=None):
    """Gets the log probability and conditionals for observations.

    Args:
      x: A batch of observations to compute the log probability of, sized
          `[batch_size, num_dims]`.
      b_enc: External encoder bias terms (`b` in [1]), sized
          `[batch_size, num_hidden]`, or None if the internal bias term should
          be used.
      b_dec: External decoder bias terms (`c` in [1]), sized
         `[batch_size, num_dims]`, or None if the internal bias term should be
         used.

    Returns:
       log_prob: The log probabilities of each observation in the batch, sized
           `[batch_size]`.
       cond_probs: The conditional probabilities at each index for every batch,
           sized `[batch_size, num_dims]`.
    """
    batch_size = tf.shape(x)[0]

    b_enc = b_enc if b_enc is not None else self.b_enc
    b_dec = b_dec if b_dec is not None else self.b_dec

    # Broadcast if needed.
    if b_enc.shape[0] == 1 != batch_size:
      b_enc = tf.tile(b_enc, [batch_size, 1])
    if b_dec.shape[0] == 1 != batch_size:
      b_dec = tf.tile(b_dec, [batch_size, 1])

    # Initial condition before the loop.
    a_0 = b_enc
    log_p_0 = tf.zeros([batch_size, 1])
    cond_p_0 = []

    x_arr = tf.unstack(
        tf.reshape(tf.transpose(x), [self.num_dims, batch_size, 1]))
    w_enc_arr = tf.unstack(self.w_enc)
    w_dec_arr = tf.unstack(self.w_dec_t)
    b_dec_arr = tf.unstack(
        tf.reshape(tf.transpose(b_dec), [self.num_dims, batch_size, 1]))

    def loop_body(i, a, log_p, cond_p):
      """Accumulate hidden state, log_p, and cond_p for index i."""
      # Get variables for time step.
      w_enc_i = w_enc_arr[i]
      w_dec_i = w_dec_arr[i]
      b_dec_i = b_dec_arr[i]
      v_i = x_arr[i]

      cond_p_i, _ = self._cond_prob(a, w_dec_i, b_dec_i)

      # Get log probability for this value. Log space avoids numerical issues.
      log_p_i = v_i * _safe_log(cond_p_i) + (1 - v_i) * _safe_log(1 - cond_p_i)

      # Accumulate log probability.
      log_p_new = log_p + log_p_i

      # Save conditional probabilities.
      cond_p_new = cond_p + [cond_p_i]

      # Encode value and add to hidden units.
      a_new = a + tf.matmul(v_i, w_enc_i)

      return a_new, log_p_new, cond_p_new

    # Build the actual loop
    a, log_p, cond_p = a_0, log_p_0, cond_p_0
    for i in range(self.num_dims):
      a, log_p, cond_p = loop_body(i, a, log_p, cond_p)

    return (tf.squeeze(log_p, squeeze_dims=[1]),
            tf.transpose(tf.squeeze(tf.stack(cond_p), [2])))

  def sample(self, b_enc=None, b_dec=None, n=None, temperature=None):
    """Generate samples for the batch from the NADE.

    Args:
      b_enc: External encoder bias terms (`b` in [1]), sized
          `[batch_size, num_hidden]`, or None if the internal bias term should
          be used.
      b_dec: External decoder bias terms (`c` in [1]), sized
          `[batch_size, num_dims]`, or None if the internal bias term should
          be used.
      n: The number of samples to generate, or None, if the batch size of
          `b_enc` should be used.
      temperature: The amount to divide the logits by before sampling
          each Bernoulli, or None if a threshold of 0.5 should be used instead
          of sampling.

    Returns:
      sample: The generated samples, sized `[batch_size, num_dims]`.
      log_prob: The log probabilities of each observation in the batch, sized
          `[batch_size]`.
    """
    b_enc = b_enc if b_enc is not None else self.b_enc
    b_dec = b_dec if b_dec is not None else self.b_dec

    batch_size = n or tf.shape(b_enc)[0]

    # Broadcast if needed.
    if b_enc.shape[0] == 1 != batch_size:
      b_enc = tf.tile(b_enc, [batch_size, 1])
    if b_dec.shape[0] == 1 != batch_size:
      b_dec = tf.tile(b_dec, [batch_size, 1])

    a_0 = b_enc
    sample_0 = []
    log_p_0 = tf.zeros([batch_size, 1])

    w_enc_arr = tf.unstack(self.w_enc)
    w_dec_arr = tf.unstack(self.w_dec_t)
    b_dec_arr = tf.unstack(
        tf.reshape(tf.transpose(b_dec), [self.num_dims, batch_size, 1]))

    def loop_body(i, a, sample, log_p):
      """Accumulate hidden state, sample, and log probability for index i."""
      # Get weights and bias for time step.
      w_enc_i = w_enc_arr[i]
      w_dec_i = w_dec_arr[i]
      b_dec_i = b_dec_arr[i]

      cond_p_i, cond_l_i = self._cond_prob(a, w_dec_i, b_dec_i)

      if temperature is None:
        v_i = tf.to_float(tf.greater_equal(cond_p_i, 0.5))
      else:
        bernoulli = tfp.distributions.Bernoulli(
            logits=cond_l_i / temperature, dtype=tf.float32)
        v_i = bernoulli.sample()

      # Accumulate sampled values.
      sample_new = sample + [v_i]

      # Get log probability for this value. Log space avoids numerical issues.
      log_p_i = v_i * _safe_log(cond_p_i) + (1 - v_i) * _safe_log(1 - cond_p_i)

      # Accumulate log probability.
      log_p_new = log_p + log_p_i

      # Encode value and add to hidden units.
      a_new = a + tf.matmul(v_i, w_enc_i)

      return a_new, sample_new, log_p_new

    a, sample, log_p = a_0, sample_0, log_p_0
    for i in range(self.num_dims):
      a, sample, log_p = loop_body(i, a, sample, log_p)

    return (tf.transpose(tf.squeeze(tf.stack(sample), [2])),
            tf.squeeze(log_p, squeeze_dims=[1]))

  def _cond_prob(self, a, w_dec_i, b_dec_i):
    """Gets the conditional probability for a single dimension.

    Args:
      a: Model's hidden state, sized `[batch_size, num_hidden]`.
      w_dec_i: The decoder weight terms for the dimension, sized
          `[num_hidden, 1]`.
      b_dec_i: The decoder bias terms, sized `[batch_size, 1]`.

    Returns:
      cond_p_i: The conditional probability of the dimension, sized
        `[batch_size, 1]`.
      cond_l_i: The conditional logits of the dimension, sized
        `[batch_size, 1]`.
    """
    # Decode hidden units to get conditional probability.
    h = tf.sigmoid(a)
    cond_l_i = b_dec_i + tf.matmul(h, w_dec_i)
    cond_p_i = tf.sigmoid(cond_l_i)
    return cond_p_i, cond_l_i
