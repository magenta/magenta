# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Provides function to build an RNN-NADE model's graph."""

import collections
import math

# internal imports

import tensorflow as tf

import magenta
from magenta.models.shared import events_rnn_graph
from tensorflow.python.layers import base as tf_layers_base
from tensorflow.python.layers import core as tf_layers_core
from tensorflow.python.util import nest as tf_nest

# TODO(adarob): Switch to public Layer once available in TF pip package.
TF_BASE_LAYER = (
    tf_layers_base.Layer if hasattr(tf_layers_base, 'Layer') else
    tf_layers_base._Layer)  # pylint:disable=protected-access


class Nade(object):
  """Neural Autoregressive Distribution Estimator [1], with external bias.

  [1]: https://arxiv.org/abs/1605.02226

  Args:
    num_dims: The number of binary dimensions for each observation.
    num_hidden: The number of hidden units in the NADE.
  """

  def __init__(self, num_dims, num_hidden, name='nade'):
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
      # Transposed decoder weigths (`W'` in [1]).
      self.w_dec_t = tf.get_variable(
          'w_dec_t',
          shape=[self._num_dims, self._num_hidden, 1],
          initializer=initializer)

  @property
  def num_hidden(self):
    """The number of hidden units for each input/output of the NADE."""
    return self._num_hidden

  @property
  def num_dims(self):
    """The number of input/output dimensions of the NADE."""
    return self._num_dims

  def log_prob(self, x, b_enc, b_dec):
    """Computes the log probability and conditionals for observations.

    Args:
      x: A batch of observations to compute the log probability of, sized
          `[batch_size, num_dims]`.
      b_enc: External encoder bias terms (`b` in [1]), sized
          `[batch_size, num_hidden]`.
      b_dec: External decoder bias terms (`c` in [1]), sized
         `[batch_size, num_dims]`.

    Returns:
       log_prob: The log probabilities of each observation in the batch, sized
           `[batch_size, 1]`.
       cond_probs: The conditional probabilities at each index for every batch,
           sized `[batch_size, num_dims]`.
    """
    batch_size = tf.shape(x)[0]

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

      cond_p_i = self._cond_prob(a, w_dec_i, b_dec_i)

      # Get log probability for this value. Log space avoids numerical issues.
      log_p_i = v_i * safe_log(cond_p_i) + (1 - v_i) * safe_log(1 - cond_p_i)

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

  def sample(self, b_enc, b_dec):
    """Generate samples for the batch from the NADE.

    Args:
      b_enc: External encoder bias terms (`b` in [1]), sized
          `[batch_size, num_hidden]`.
      b_dec: External decoder bias terms (`c` in [1]), sized
          `[batch_size, num_dims]`.

    Returns:
      sample: The generated samples, sized `[batch_size, num_dims]`.
      log_prob: The log probabilities of each observation in the batch, sized
          `[batch_size, 1]`.
    """
    batch_size = tf.shape(b_enc)[0]

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

      cond_p_i = self._cond_prob(a, w_dec_i, b_dec_i)

      bernoulli = tf.contrib.distributions.Bernoulli(probs=cond_p_i,
                                                     dtype=tf.float32)
      v_i = bernoulli.sample()

      # Accumulate sampled values.
      sample_new = sample + [v_i]

      # Get log probability for this value. Log space avoids numerical issues.
      log_p_i = v_i * safe_log(cond_p_i) + (1 - v_i) * safe_log(1 - cond_p_i)

      # Accumulate log probability.
      log_p_new = log_p + log_p_i

      # Encode value and add to hidden units.
      a_new = a + tf.matmul(v_i, w_enc_i)

      return a_new, sample_new, log_p_new

    # Build the actual loop.
    a, sample, log_p = a_0, sample_0, log_p_0
    for i in range(self.num_dims):
      a, sample, log_p = loop_body(i, a, sample, log_p)

    return tf.transpose(tf.squeeze(tf.stack(sample), [2])), log_p

  def _cond_prob(self, a, w_dec_i, b_dec_i):
    """Gets the conditional probability for a single dimension.

    Args:
      a: Model's hidden state, sized `[batch_size, num_hidden]`.
      w_dec_i: The decoder weight terms for the dimension, sized
          `[num_hidden, 1]`.
      b_dec_i: The decoder bias terms, sized `[batch_size, 1]`.

    Returns:
      The conditional probability of the dimension, sized `[batch_size, 1]`.
    """
    # Decode hidden units to get conditional probability.
    h = tf.sigmoid(a)
    p_cond_i = tf.sigmoid(b_dec_i + tf.matmul(h, w_dec_i))
    return p_cond_i


def safe_log(tensor):
  """Lower bounded log function."""
  return tf.log(1e-6 + tensor)


_RnnNadeStateTuple = collections.namedtuple(
    'RnnNadeStateTuple', ('b_enc', 'b_dec', 'rnn_state'))


class RnnNadeStateTuple(_RnnNadeStateTuple):
  """Tuple used by RnnNade to store state.

  Stores three elements `(b_enc, b_dec, rnn_state)`, in that order:
    b_enc: NADE encoder bias terms (`b` in [1]), sized
        `[batch_size, num_hidden]`.
    b_dec: NADE decoder bias terms (`c` in [1]), sized `[batch_size, num_dims]`.
    rnn_state: The RNN cell's state.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (b_enc, b_dec, rnn_state) = self
    if not b_enc.dtype == b_dec.dtype == rnn_state.dtype:
      raise TypeError(
          'Inconsistent internal state: %s vs %s vs %s' %
          (str(b_enc.dtype), str(b_dec.dtype), str(rnn_state.dtype)))
    return b_enc.dtype


class RnnNade(object):
  """RNN-NADE [2], a NADE parameterized by an RNN.

  The NADE's bias parameters are given by the output of the RNN.

  [2]: https://arxiv.org/abs/1206.6392

  Args:
    rnn_cell: The tf.contrib.rnn.RnnCell to use.
    num_dims: The number of binary dimensions for each observation.
    num_hidden: The number of hidden units in the NADE.
  """

  def __init__(self, rnn_cell, num_dims, num_hidden):
    self._num_dims = num_dims
    self._rnn_cell = rnn_cell
    self._fc_layer = tf_layers_core.Dense(units=num_dims + num_hidden)
    self._nade = Nade(num_dims, num_hidden)

  def _get_rnn_zero_state(self, batch_size):
    """Return a tensor or tuple of tensors for an initial rnn state."""
    return self._rnn_cell.zero_state(batch_size, tf.float32)

  class SampleNadeLayer(TF_BASE_LAYER):
    """Layer that computes samples from a NADE."""

    def __init__(self, nade, name=None, **kwargs):
      super(RnnNade.SampleNadeLayer, self).__init__(name=name, **kwargs)
      self._nade = nade
      self._empty_result = tf.zeros([0, nade.num_dims])

    def call(self, inputs):
      b_enc, b_dec = tf.split(
          inputs, [self._nade.num_hidden, self._nade.num_dims], axis=1)
      return self._nade.sample(b_enc, b_dec)[0]

  def _get_state(self,
                 inputs,
                 lengths=None,
                 initial_state=None):
    """Computes the state of the RNN-NADE (NADE bias parameters and RNN state).

    Args:
      inputs: A batch of sequences to compute the state from, sized
          `[batch_size, max(lengths), num_dims]` or `[batch_size, num_dims]`.
      lengths: The length of each sequence, sized `[batch_size]`.
      initial_state: An RnnNadeStateTuple, the initial state of the RNN-NADE, or
          None if the zero state should be used.

    Returns:
      final_state: An RnnNadeStateTuple, the final state of the RNN-NADE.
    """
    batch_size = inputs.shape[0].value

    lengths = (
        tf.tile(tf.shape(inputs)[1:2], [batch_size]) if lengths is None else
        lengths)
    initial_rnn_state = (
        self._get_rnn_zero_state(batch_size) if initial_state is None else
        initial_state.rnn_state)

    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=inputs,
        sequence_length=lengths)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=self._rnn_cell,
        helper=helper,
        initial_state=initial_rnn_state,
        output_layer=self._fc_layer)

    final_outputs, final_rnn_state = tf.contrib.seq2seq.dynamic_decode(
        decoder)[0:2]

    # Flatten time dimension.
    final_outputs_flat = magenta.common.flatten_maybe_padded_sequences(
        final_outputs.rnn_output, lengths)

    b_enc, b_dec = tf.split(
        final_outputs_flat, [self._nade.num_hidden, self._nade.num_dims],
        axis=1)

    return RnnNadeStateTuple(b_enc, b_dec, final_rnn_state)

  def log_prob(self, sequences, lengths=None):
    """Computes the log probability of a sequence of values.

    Flattens the time dimension.

    Args:
      sequences: A batch of sequences to compute the log probabilities of,
          sized `[batch_size, max(lengths), num_dims]`.
      lengths: The length of each sequence, sized `[batch_size]` or None if
          all are equal.

    Returns:
      log_prob: The log probability of each sequence value, sized
          `[sum(lengths), 1]`.
      cond_prob: The conditional probabilities at each non-padded value for
          every batch, sized `[sum(lengths), num_dims]`.
    """
    assert self._num_dims == sequences.shape[2].value

    # Remove last value from input sequences.
    inputs = sequences[:, 0:-1, :]

    # Add initial padding value to input sequences.
    inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])

    state = self._get_state(inputs, lengths=lengths)

    # Flatten time dimension.
    labels_flat = magenta.common.flatten_maybe_padded_sequences(
        sequences, lengths)

    return self._nade.log_prob(labels_flat, state.b_enc, state.b_dec)

  def steps(self, inputs, state):
    """Computes the new RNN-NADE state from a batch of inputs.

    Args:
      inputs: A batch of values to compute the log probabilities of,
          sized `[batch_size, length, num_dims]`.
      state: An RnnNadeStateTuple containing the RNN-NADE for each value, sized
          `([batch_size, self._nade.num_hidden], [batch_size, num_dims],
            [batch_size, self._rnn_cell.state_size]`).

    Returns:
      new_state: The updated RNN-NADE state tuple given the new inputs.
    """
    return self._get_state(inputs, initial_state=state)

  def sample_single(self, state):
    """Computes a sample and its probability from each of a batch of states.

    Args:
      state: An RnnNadeStateTuple containing the state of the RNN-NADE for each
          sample, sized
          `([batch_size, self._nade.num_hidden], [batch_size, num_dims],
            [batch_size, self._rnn_cell.state_size]`).

    Returns:
      sample: A sample for each input state, sized `[batch_size, num_dims]`.
      log_prob: The log probability of each sample, sized `[batch_size, 1]`.
    """
    sample, log_prob = self._nade.sample(state.b_enc, state.b_dec)

    return sample, log_prob

  def zero_state(self, batch_size):
    """Create an RnnNadeStateTuple of zeros.

    Args:
      batch_size: batch size.

    Returns:
      An RnnNadeStateTuple of zeros.
    """
    with tf.name_scope('RnnNadeZeroState', values=[batch_size]):
      zero_state = self._get_rnn_zero_state(batch_size)
      return RnnNadeStateTuple(
          tf.zeros((batch_size, self._nade.num_hidden), name='b_enc'),
          tf.zeros((batch_size, self._num_dims), name='b_dec'),
          zero_state)


def build_graph(mode, config, sequence_example_file_paths=None):
  """Builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
        to use.
    sequence_example_file_paths: A list of paths to TFRecord files containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation. May be a sharded file of the form.

  Returns:
    A tf.Graph instance which contains the TF ops.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError("The mode parameter must be 'train', 'eval', "
                     "or 'generate'. The mode parameter was: %s" % mode)

  hparams = config.hparams
  encoder_decoder = config.encoder_decoder

  tf.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size

  with tf.Graph().as_default() as graph:
    inputs, lengths = None, None

    if mode == 'train' or mode == 'eval':
      inputs, _, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams.batch_size, input_size,
          shuffle=mode == 'train')

    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32,
                              [hparams.batch_size, None, input_size])

    cell = events_rnn_graph.make_rnn_cell(
        hparams.rnn_layer_sizes,
        dropout_keep_prob=hparams.dropout_keep_prob if mode == 'train' else 1.0,
        attn_length=(
            hparams.attn_length if hasattr(hparams, 'attn_length') else 0))

    rnn_nade = RnnNade(
        cell,
        num_dims=input_size,
        num_hidden=hparams.nade_hidden_units)

    if mode == 'train' or mode == 'eval':
      log_probs, cond_probs = rnn_nade.log_prob(inputs, lengths)

      inputs_flat = tf.to_float(
          magenta.common.flatten_maybe_padded_sequences(inputs, lengths))
      predictions_flat = tf.to_float(tf.greater_equal(cond_probs, .5))

      if mode == 'train':
        loss = tf.reduce_mean(-log_probs)
        perplexity = tf.reduce_mean(tf.exp(log_probs))
        correct_predictions = tf.to_float(
            tf.equal(inputs_flat, predictions_flat))
        accuracy = tf.reduce_mean(correct_predictions)
        precision = (tf.reduce_sum(inputs_flat * predictions_flat) /
                     tf.reduce_sum(predictions_flat))
        recall = (tf.reduce_sum(inputs_flat * predictions_flat) /
                  tf.reduce_sum(inputs_flat))

        optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparams.clip_norm)
        tf.add_to_collection('train_op', train_op)

        vars_to_summarize = {
            'loss': loss,
            'metrics/perplexity': perplexity,
            'metrics/accuracy': accuracy,
            'metrics/precision': precision,
            'metrics/recall': recall,
        }
      elif mode == 'eval':
        vars_to_summarize, update_ops = tf.contrib.metrics.aggregate_metric_map(
            {
                'loss': tf.metrics.mean(-log_probs),
                'metrics/perplexity': tf.metrics.mean(tf.exp(log_probs)),
                'metrics/accuracy': tf.metrics.accuracy(
                    inputs_flat, predictions_flat),
                'metrics/precision': tf.metrics.precision(
                    inputs_flat, predictions_flat),
                'metrics/recall': tf.metrics.recall(
                    inputs_flat, predictions_flat),
            })
        for updates_op in update_ops.values():
          tf.add_to_collection('eval_ops', updates_op)

      precision = vars_to_summarize['metrics/precision']
      recall = vars_to_summarize['metrics/precision']
      f1_score = tf.where(
          tf.greater(precision + recall, 0), 2 * (
              (precision * recall) / (precision + recall)), 0)
      vars_to_summarize['metrics/f1_score'] = f1_score
      for var_name, var_value in vars_to_summarize.iteritems():
        tf.summary.scalar(var_name, var_value)
        tf.add_to_collection(var_name, var_value)

    elif mode == 'generate':
      initial_state = rnn_nade.zero_state(hparams.batch_size)

      final_state = rnn_nade.steps(inputs, initial_state)
      samples, log_prob = rnn_nade.sample_single(initial_state)

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('sample', samples)
      tf.add_to_collection('log_prob', log_prob)

      # Flatten state tuples for metagraph compatibility.
      for state in tf_nest.flatten(initial_state):
        tf.add_to_collection('initial_state', state)
      for state in tf_nest.flatten(final_state):
        tf.add_to_collection('final_state', state)

  return graph
