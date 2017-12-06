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
"""LSTM-based encoders and decoders for MusicVAE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# internal imports
import tensorflow as tf

from magenta.common import flatten_maybe_padded_sequences
from magenta.models.music_vae import base_model
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest as tf_nest


def rnn_cell(rnn_cell_size, dropout_keep_prob, attn_len=0):
  """Builds an LSTMBlockCell based on the given parameters."""
  cells = []
  for layer_size in rnn_cell_size:
    cell = tf.contrib.rnn.LSTMBlockCell(layer_size)
    if attn_len and not cells:
      # Add attention wrapper to first layer.
      cell = tf.contrib.rnn.AttentionCellWrapper(
          cell, attn_len, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell,
        input_keep_prob=dropout_keep_prob,
        output_keep_prob=dropout_keep_prob)
    cells.append(cell)
  return tf.contrib.rnn.MultiRNNCell(cells)


def initial_cell_state_from_embedding(cell, z, batch_size, name=None):
  """Computes an initial RNN `cell` state from an embedding, `z`."""
  flat_state_sizes = tf_nest.flatten(cell.state_size)
  return tf_nest.pack_sequence_as(
      cell.zero_state(batch_size=batch_size, dtype=tf.float32),
      tf.split(
          tf.layers.dense(
              z,
              sum(flat_state_sizes),
              activation=tf.tanh,
              kernel_initializer=tf.random_normal_initializer(stddev=0.001),
              name=name),
          flat_state_sizes,
          axis=1))


def _get_sampling_probability(hparams, is_training):
  """Returns `sampling_probabiliy` if `sampling schedule` given or 0."""
  if (not hasattr(hparams, 'sampling_schedule') or
      not hparams.sampling_schedule):
    return tf.convert_to_tensor(0.0, tf.float32)

  if not is_training:
    # This is likely an eval/test job associated with a training job using
    # scheduled sampling.
    tf.logging.warning(
        'Setting non-training sampling schedule from %s:%f to constant:1.0.',
        hparams.sampling_schedule, hparams.sampling_rate)
    hparams.sampling_schedule = 'constant'
    hparams.sampling_rate = 1.0
  if hparams.sampling_schedule == 'constant':
    sampling_probability = tf.constant(hparams.sampling_rate)
  elif hparams.sampling_schedule == 'inverse_sigmoid':
    k = tf.constant(hparams.sampling_rate)
    sampling_probability = 1.0 - (
        k / (k + tf.exp(tf.to_float(tf.train.get_or_create_global_step()) / k)))
  elif hparams.sampling_schedule == 'exponential':
    if not 0 < hparams.sampling_rate < 1:
      raise ValueError(
          'Exponential sampling rate must be in the interval (0, 1). Got %f.'
          % hparams.sampling_rate)
    k = tf.constant(hparams.sampling_rate)
    sampling_probability = (
        1.0 - tf.pow(k, tf.to_float(tf.train.get_or_create_global_step())))
  else:
    tf.logging.fatal('Invalid sampling_schedule: %s',
                     hparams.sampling_schedule)
  tf.summary.scalar('sampling_probability', sampling_probability)
  return tf.convert_to_tensor(sampling_probability, tf.float32)


class BidirectionalLstmEncoder(base_model.BaseEncoder):
  """Bidirectional LSTM Encoder."""

  def build(self, hparams, is_training=True):
    dropout_keep_prob = hparams.dropout_keep_prob if is_training else 1.0

    tf.logging.info('\nEncoder Cells (bidirectional):\n'
                    '  units: %s\n'
                    '  input dropout keep prob: %4.4f\n'
                    '  output dropout keep prob: %4.4f\n',
                    hparams.enc_rnn_size,
                    dropout_keep_prob,
                    dropout_keep_prob)
    self._enc_cells_fw = []
    self._enc_cells_bw = []
    for layer_size in hparams.enc_rnn_size:
      self._enc_cells_fw.append(rnn_cell([layer_size], dropout_keep_prob))
      self._enc_cells_bw.append(rnn_cell([layer_size], dropout_keep_prob))

  def encode(self, sequence, sequence_length):
    res = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        self._enc_cells_fw,
        self._enc_cells_bw,
        sequence,
        sequence_length=sequence_length,
        time_major=False,
        dtype=tf.float32,
        scope='encoder')

    unused_outputs, last_states_fw, last_states_bw = res
    last_state_fw = last_states_fw[-1][-1]
    last_state_bw = last_states_bw[-1][-1]

    last_h = tf.concat([last_state_fw.h, last_state_bw.h], 1)
    return last_h


class BaseLstmDecoder(base_model.BaseDecoder):
  """Abstract LSTM Decoder class.

  Implementations must define the following abstract methods:
      -`_sample`
      -`_flat_reconstruction_loss`
  """

  def build(self, hparams, output_depth, is_training=False):
    dropout_keep_prob = hparams.dropout_keep_prob if is_training else 1.0
    tf.logging.info('\nDecoder Cells:\n'
                    '  units: %s\n'
                    '  input dropout keep prob: %4.4f\n'
                    '  output dropout keep prob: %4.4f\n',
                    hparams.dec_rnn_size,
                    dropout_keep_prob,
                    dropout_keep_prob)

    self._sampling_probability = _get_sampling_probability(
        hparams, is_training)
    self._output_depth = output_depth
    self._output_layer = layers_core.Dense(
        output_depth, name='output_projection')
    self._dec_cell = rnn_cell(
        hparams.dec_rnn_size,
        hparams.dropout_keep_prob,
        hparams.dec_rnn_attn_len)

  @abc.abstractmethod
  def _sample(self, rnn_output, temperature):
    """Core sampling method for a single time step.

    Args:
      rnn_output: The output from a single timestep of the RNN, sized
          [batch_size, rnn_output_size].
      temperature: A scalar float specifying a sampling temperature.
    Returns:
      A batch of samples from the model.
    """
    pass

  @abc.abstractmethod
  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    """Core loss calculation method for flattened outputs.

    Args:
      flat_x_target: The flattened ground truth vectors, sized
        [sum(x_length), self._output_depth].
      flat_rnn_output: The flattened output from all timeputs of the RNN,
        sized [sum(x_length), rnn_output_size].
    Returns:
      r_loss: The unreduced reconstruction losses, sized [sum(x_length)].
      metric_map: A map of metric names to tuples, each of which contain the
        pair of (value_tensor, update_op) from a tf.metrics streaming metric.
      truths: Ground truth labels.
      predictions: Predicted labels.
    """
    pass

  def _decode(self, batch_size, helper, z, max_length=None):
    initial_state = initial_cell_state_from_embedding(
        self._dec_cell, z, batch_size, name='decoder/z_to_initial_state')

    decoder = tf.contrib.seq2seq.BasicDecoder(
        self._dec_cell,
        helper,
        initial_state=initial_state,
        output_layer=self._output_layer)
    final_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=max_length,
        swap_memory=True,
        scope='decoder')
    return final_output

  def reconstruction_loss(self, x_input, x_target, x_length, z=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
          `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
          sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
          `[n, z_size]`.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      truths: Ground truth labels, sized
    """
    batch_size = x_input.shape[0].value

    has_z = z is not None
    z = tf.zeros([batch_size, 0]) if z is None else z
    repeated_z = tf.tile(
        tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])

    sampling_probability_static = tensor_util.constant_value(
        self._sampling_probability)
    if sampling_probability_static == 0.0:
      # Use teacher forcing.
      x_input = tf.concat([x_input, repeated_z], axis=2)
      helper = tf.contrib.seq2seq.TrainingHelper(x_input, x_length)
    else:
      # Use scheduled sampling.
      helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
          inputs=x_input,
          sequence_length=x_length,
          auxiliary_inputs=repeated_z if has_z else None,
          sampling_probability=self._sampling_probability,
          next_inputs_fn=self._sample)

    decoder_outputs = self._decode(batch_size, helper=helper, z=z)
    flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
    flat_rnn_output = flatten_maybe_padded_sequences(
        decoder_outputs.rnn_output, x_length)
    r_loss, metric_map, truths, predictions = self._flat_reconstruction_loss(
        flat_x_target, flat_rnn_output)

    # Sum loss over sequences.
    cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
    r_losses = []
    for i in range(batch_size):
      b, e = cum_x_len[i], cum_x_len[i + 1]
      r_losses.append(tf.reduce_sum(r_loss[b:e]))
    r_loss = tf.stack(r_losses)

    return r_loss, metric_map, truths, predictions

  def sample(self, n, max_length=None, z=None, temperature=1.0,
             start_inputs=None, end_fn=None):
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    # Use a dummy Z in unconditional case.
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    # If not given, start with zeros.
    start_inputs = start_inputs if start_inputs is not None else tf.zeros(
        [n, self._output_depth], dtype=tf.float32)
    # In the conditional case, also concatenate the Z.
    start_inputs = tf.concat([start_inputs, z], axis=-1)

    sample_fn = lambda x: self._sample(x, temperature)
    end_fn = end_fn or (lambda x: False)
    # In the conditional case, concatenate Z to the sampled value.
    next_inputs_fn = lambda x: tf.concat([x, z], axis=-1)

    sampler = tf.contrib.seq2seq.InferenceHelper(
        sample_fn, sample_shape=[self._output_depth], sample_dtype=tf.float32,
        start_inputs=start_inputs, end_fn=end_fn, next_inputs_fn=next_inputs_fn)

    decoder_outputs = self._decode(
        n, helper=sampler, z=z, max_length=max_length)

    return decoder_outputs.sample_id


class CategoricalLstmDecoder(BaseLstmDecoder):
  """LSTM decoder with single categorical output."""

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    flat_logits = flat_rnn_output
    flat_truth = tf.argmax(flat_x_target, axis=1)
    flat_predictions = tf.argmax(flat_logits, axis=1)
    r_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=flat_x_target, logits=flat_logits)

    metric_map = {
        'metrics/accuracy':
            tf.metrics.accuracy(flat_truth, flat_predictions),
        'metrics/mean_per_class_accuracy':
            tf.metrics.mean_per_class_accuracy(
                flat_truth, flat_predictions, flat_x_target.shape[-1].value),
    }
    return r_loss, metric_map, flat_truth, flat_predictions

  def _sample(self, rnn_output, temperature=1.0):
    sampler = tf.contrib.distributions.Categorical(
        logits=rnn_output / temperature)
    sample_labels = sampler.sample()
    return tf.one_hot(sample_labels, self._output_depth, dtype=tf.float32)


class MultiOutCategoricalLstmDecoder(CategoricalLstmDecoder):
  """LSTM decoder with multiple categorical outputs."""

  def __init__(self, output_depths):
    self._output_depths = output_depths

  def build(self, hparams, output_depth, is_training):
    if sum(self._output_depths) != output_depth:
      raise ValueError(
          'Decoder output depth does not match sum of sub-decoders: %s vs %d',
          self._output_depths, output_depth)
    super(MultiOutCategoricalLstmDecoder, self).build(
        hparams, output_depth, is_training)

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    split_x_target = tf.split(flat_x_target, self._output_depths, axis=-1)
    split_rnn_output = tf.split(
        flat_rnn_output, self._output_depths, axis=-1)

    losses = []
    truths = []
    predictions = []
    metric_map = {}
    for i in range(len(self._output_depths)):
      l, m, t, p = (
          super(MultiOutCategoricalLstmDecoder, self)._flat_reconstruction_loss(
              split_x_target[i], split_rnn_output[i]))
      losses.append(l)
      truths.append(t)
      predictions.append(p)
      for k, v in m.items():
        metric_map['%s_%d' % (k, i)] = v

    return (tf.reduce_sum(losses, axis=0),
            metric_map,
            tf.stack(truths),
            tf.stack(predictions))

  def _sample(self, rnn_output, temperature=1.0):
    split_logits = tf.split(rnn_output, self._output_depths, axis=-1)
    samples = []
    for logits, output_depth in zip(split_logits, self._output_depths):
      sampler = tf.contrib.distributions.Categorical(
          logits=logits / temperature)
      sample_label = sampler.sample()
      samples.append(tf.one_hot(sample_label, output_depth, dtype=tf.float32))
    return tf.concat(samples, axis=-1)


class HierarchicalMultiOutLstmDecoder(base_model.BaseDecoder):
  """Hierarchical LSTM decoder with (optional) multiple categorical outputs."""

  def __init__(self, core_decoders, output_depths):
    """Initializer for a HierarchicalMultiOutLstmDecoder.

    Args:
      core_decoders: The BaseDecoder implementation class(es) to use at the
          output layer.
      output_depths: A list of output depths for the core decoders.
    Raises:
      ValueError: If the number of core decoders and output depths are not
          equal.
    """
    if len(core_decoders) != len(output_depths):
      raise ValueError(
          'The number of `core_decoders` and `output_depths` provided to a '
          'HierarchicalMultiOutLstmDecoder must be equal. Got: %d != %d',
          len(core_decoders), len(output_depths))
    self._core_decoders = core_decoders
    self._output_depths = output_depths

  def build(self, hparams, output_depth, is_training):
    if sum(self._output_depths) != output_depth:
      raise ValueError(
          'Decoder output depth does not match sum of sub-decoders: %s vs %d',
          self._output_depths, output_depth)
    self.hparams = hparams
    for cd, od in zip(self._core_decoders, self._output_depths):
      cd.build(hparams, od, is_training)

  def _hierarchical_decode(self, z=None):
    hparams = self.hparams
    batch_size = hparams.batch_size

    if z is None:
      learned_initial_embedding = tf.get_variable(
          'learned_initial_embedding',
          shape=hparams.z_size,
          initializer=tf.random_normal_initializer(stddev=0.001))
      embeddings = [tf.stack([learned_initial_embedding] * batch_size)]
    else:
      embeddings = [z]

    for i, h_size in enumerate(hparams.hierarchical_output_sizes):
      if h_size % len(embeddings) != 0:
        raise ValueError(
            'Each size in `hierarchical_output_sizes` must be evenly divisible '
            'by the previous. Got: %d !/ %d', h_size, len(embeddings))
      all_outputs = []
      with tf.variable_scope('hierarchical_layer_%d' % i) as scope:
        cell = rnn_cell(hparams.dec_rnn_size, dropout_keep_prob=1.0, attn_len=0)
        for e in embeddings:
          initial_state = initial_cell_state_from_embedding(
              cell, e, batch_size, name='e_to_initial_state')
          input_ = tf.zeros((batch_size, 1))
          outputs, _ = tf.nn.static_rnn(
              cell, [input_] * (h_size // len(embeddings)),
              initial_state=initial_state)
          all_outputs.extend(outputs)
          # Reuse layer next time.
          scope.reuse_variables()
      embeddings = all_outputs
    return embeddings

  def reconstruction_loss(self, x_input, x_target, x_length, z=None):
    embeddings = self._hierarchical_decode(z)
    n = len(embeddings)

    # TODO(adarob): Support variable length outputs.
    with tf.control_dependencies([
        tf.assert_equal(
            x_length, (x_length[0] // n) * n,
            message='HierarchicalMultiOutLstmDecoder requires `x_length` to '
            'all be equal and divisible by the final number of embeddings.')]):
      x_input = tf.identity(x_input)

    # Split sequences into n x M subsequences where M is the number of core
    # models.
    split_x_input = [
        tf.split(x, self._output_depths, axis=-1)
        for x in tf.split(x_input, n, axis=1)]
    split_x_target = [
        tf.split(x, self._output_depths, axis=-1)
        for x in tf.split(x_target, n, axis=1)]
    loss_outputs = [[] for _ in self._core_decoders]

    # Compute reconstruction loss for the n x M split sequences.
    for i, e in enumerate(embeddings):
      for j, cd in enumerate(self._core_decoders):
        with tf.variable_scope('core_decoder_%d' % j, reuse=i > 0):
          # TODO(adarob): Sample initial inputs when using scheduled sampling.
          loss_outputs[j].append(
              cd.reconstruction_loss(
                  split_x_input[i][j], split_x_target[i][j], x_length // n, e))

    # Accumulate the split sequence losses.
    all_r_losses = []
    all_truth = []
    all_predictions = []
    metric_map = {}
    for j, loss_outputs_j in enumerate(loss_outputs):
      r_losses, _, truth, predictions = zip(*loss_outputs_j)
      all_r_losses.append(tf.reduce_sum(r_losses, axis=0))
      all_truth.append(tf.concat(truth, axis=-1))
      all_predictions.append(tf.concat(predictions, axis=-1))
      metric_map['metrics/accuracy_%d' % j] = tf.metrics.accuracy(
          all_truth[-1], all_predictions[-1])
      metric_map['metrics/mean_per_class_accuracy_%d' % j] = (
          tf.metrics.mean_per_class_accuracy(
              all_truth[-1], all_predictions[-1], self._output_depths[j]))

    return (tf.reduce_sum(all_r_losses, axis=0),
            metric_map,
            tf.stack(all_truth, axis=-1),
            tf.stack(all_predictions, axis=-1))

  def sample(self, n, max_length=None, z=None, temperature=1.0):
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    if max_length is None:
      # TODO(adarob): Support variable length outputs.
      raise ValueError(
          'HierarchicalMultiOutLstmDecoder requires `max_length` be provided '
          'during sampling.')

    embeddings = self._hierarchical_decode(z)

    sample_ids = []
    for j, cd in enumerate(self._core_decoders):
      sample_ids_j = []
      with tf.variable_scope('core_decoder_%d' % j) as scope:
        for e in embeddings:
          sample_ids_j.append(
              cd.sample(
                  n,
                  max_length // len(embeddings),
                  z=e,
                  temperature=temperature,
                  start_inputs=(
                      sample_ids_j[-1][:, -1] if sample_ids_j else None)))
          scope.reuse_variables()
        sample_ids.append(tf.concat(sample_ids_j, axis=1))

    return tf.concat(sample_ids, axis=-1)


def get_default_hparams():
  """Returns copy of default HParams for LSTM models."""
  hparams_map = base_model.get_default_hparams().values()
  hparams_map.update({
      'conditional': True,
      'dec_rnn_size': [512],  # Decoder RNN: number of units per layer.
      'dec_rnn_attn_len': 0,  # Decoder RNN: length of attention vector.
      'enc_rnn_size': [256],  # Encoder RNN: number of units per layer per dir.
      'dropout_keep_prob': 1.0,  # Probability all dropout keep.
      'sampling_schedule': 'constant',  # constant, exponential, inverse_sigmoid
      'sampling_rate': 0.0,  # Interpretation is based on `sampling_schedule`.
  })
  return tf.contrib.training.HParams(**hparams_map)
