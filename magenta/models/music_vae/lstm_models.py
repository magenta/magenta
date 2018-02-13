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
from magenta.common import Nade
from magenta.models.music_vae import base_model
from magenta.models.music_vae import lstm_utils
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest


class LstmEncoder(base_model.BaseEncoder):
  """Unidirectional LSTM Encoder."""

  @property
  def output_depth(self):
    return self._cell.output_size

  def build(self, hparams, is_training=True, name_or_scope='encoder'):
    self._is_training = is_training
    self._name_or_scope = name_or_scope
    self._use_cudnn = hparams.use_cudnn

    tf.logging.info('\nEncoder Cells (unidirectional):\n'
                    '  units: %s\n',
                    hparams.enc_rnn_size)
    if self._use_cudnn:
      self._cudnn_lstm = lstm_utils.cudnn_lstm_layer(
          hparams.enc_rnn_size,
          hparams.dropout_keep_prob,
          is_training,
          name_or_scope=self._name_or_scope)
    else:
      self._cell = lstm_utils.rnn_cell(
          hparams.enc_rnn_size, hparams.dropout_keep_prob, is_training)

  def encode(self, sequence, sequence_length):
    # Convert to time-major.
    sequence = tf.transpose(sequence, [1, 0, 2])
    if self._use_cudnn:
      outputs, _ = self._cudnn_lstm(
          sequence, training=self._is_training)
      return lstm_utils.get_final(outputs, sequence_length)
    else:
      outputs, _ = tf.nn.dynamic_rnn(
          self._cell, sequence, sequence_length, dtype=tf.float32,
          time_major=True, scope=self._name_or_scope)
      return outputs[-1]


class BidirectionalLstmEncoder(base_model.BaseEncoder):
  """Bidirectional LSTM Encoder."""

  @property
  def output_depth(self):
    if self._use_cudnn:
      return self._cells[0][-1].num_units + self._cells[1][-1].num_units
    return self._cells[0][-1].output_size + self._cells[1][-1].output_size

  def build(self, hparams, is_training=True, name_or_scope='encoder'):
    self._is_training = is_training
    self._name_or_scope = name_or_scope
    self._use_cudnn = hparams.use_cudnn

    tf.logging.info('\nEncoder Cells (bidirectional):\n'
                    '  units: %s\n',
                    hparams.enc_rnn_size)

    if isinstance(name_or_scope, tf.VariableScope):
      name = name_or_scope.name
      reuse = name_or_scope.reuse
    else:
      name = name_or_scope
      reuse = None

    cells_fw = []
    cells_bw = []
    for i, layer_size in enumerate(hparams.enc_rnn_size):
      if self._use_cudnn:
        cells_fw.append(lstm_utils.cudnn_lstm_layer(
            [layer_size], hparams.dropout_keep_prob, is_training,
            name_or_scope=tf.VariableScope(
                reuse,
                name + '/cell_%d/bidirectional_rnn/fw' % i)))
        cells_bw.append(lstm_utils.cudnn_lstm_layer(
            [layer_size], hparams.dropout_keep_prob, is_training,
            name_or_scope=tf.VariableScope(
                reuse,
                name + '/cell_%d/bidirectional_rnn/bw' % i)))
      else:
        cells_fw.append(
            lstm_utils.rnn_cell(
                [layer_size], hparams.dropout_keep_prob, is_training))
        cells_bw.append(
            lstm_utils.rnn_cell(
                [layer_size], hparams.dropout_keep_prob, is_training))

    self._cells = (cells_fw, cells_bw)

  def encode(self, sequence, sequence_length):
    cells_fw, cells_bw = self._cells
    if self._use_cudnn:
      # Implements stacked bidirectional LSTM for variable-length sequences,
      # which are not supported by the CudnnLSTM layer.
      inputs_fw = tf.transpose(sequence, [1, 0, 2])
      for lstm_fw, lstm_bw in zip(cells_fw, cells_bw):
        outputs_fw, _ = lstm_fw(inputs_fw, training=self._is_training)
        inputs_bw = tf.reverse_sequence(
            inputs_fw, sequence_length, seq_axis=0, batch_axis=1)
        outputs_bw, _ = lstm_bw(inputs_bw, training=self._is_training)
        outputs_bw = tf.reverse_sequence(
            outputs_bw, sequence_length, seq_axis=0, batch_axis=1)

        inputs_fw = tf.concat([outputs_fw, outputs_bw], axis=2)

      last_h_fw = lstm_utils.get_final(outputs_fw, sequence_length)
      # outputs_bw has already been reversed, so we can take the first element.
      last_h_bw = outputs_bw[0]

    else:
      _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
          cells_fw,
          cells_bw,
          sequence,
          sequence_length=sequence_length,
          time_major=False,
          dtype=tf.float32,
          scope=self._name_or_scope)
      # Note we access the outputs (h) from the states since the backward
      # ouputs are reversed to the input order in the returned outputs.
      last_h_fw = states_fw[-1][-1].h
      last_h_bw = states_bw[-1][-1].h

    return tf.concat([last_h_fw, last_h_bw], 1)


class BaseLstmDecoder(base_model.BaseDecoder):
  """Abstract LSTM Decoder class.

  Implementations must define the following abstract methods:
      -`_sample`
      -`_flat_reconstruction_loss`
  """

  def build(self, hparams, output_depth, is_training=False):
    self._is_training = is_training

    tf.logging.info('\nDecoder Cells:\n'
                    '  units: %s\n',
                    hparams.dec_rnn_size)

    self._sampling_probability = lstm_utils.get_sampling_probability(
        hparams, is_training)
    self._output_depth = output_depth
    self._output_layer = layers_core.Dense(
        output_depth, name='output_projection')
    self._dec_cell = lstm_utils.rnn_cell(
        hparams.dec_rnn_size, hparams.dropout_keep_prob, is_training)
    self._cudnn_dec_lstm = lstm_utils.cudnn_lstm_layer(
        hparams.dec_rnn_size, hparams.dropout_keep_prob, is_training,
        name_or_scope='decoder') if hparams.use_cudnn else None

  @abc.abstractmethod
  def _sample(self, rnn_output, temperature):
    """Core sampling method for a single time step.

    Args:
      rnn_output: The output from a single timestep of the RNN, sized
          `[batch_size, rnn_output_size]`.
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
        `[sum(x_length), self._output_depth]`.
      flat_rnn_output: The flattened output from all timeputs of the RNN,
        sized `[sum(x_length), rnn_output_size]`.
    Returns:
      r_loss: The unreduced reconstruction losses, sized `[sum(x_length)]`.
      metric_map: A map of metric names to tuples, each of which contain the
        pair of (value_tensor, update_op) from a tf.metrics streaming metric.
      truths: Ground truth labels.
      predictions: Predicted labels.
    """
    pass

  def _decode(self, z, helper, input_shape, max_length=None):
    """Decodes the given batch of latent vectors vectors, which may be 0-length.

    Args:
      z: Batch of latent vectors, sized `[batch_size, z_size]`, where `z_size`
        may be 0 for unconditioned decoding.
      helper: A seq2seq.Helper to use. If a TrainingHelper is passed and a
        CudnnLSTM has previously been defined, it will be used instead.
      input_shape: The shape of each model input vector passed to the decoder.
      max_length: (Optional) The maximum iterations to decode.

    Returns:
      results: The LstmDecodeResults.
    """
    initial_state = lstm_utils.initial_cell_state_from_embedding(
        self._dec_cell, z, name='decoder/z_to_initial_state')

    # CudnnLSTM does not support sampling so it can only replace TrainingHelper.
    if  self._cudnn_dec_lstm and type(helper) is seq2seq.TrainingHelper:  # pylint:disable=unidiomatic-typecheck
      rnn_output, _ = self._cudnn_dec_lstm(
          tf.transpose(helper.inputs, [1, 0, 2]),
          initial_state=lstm_utils.cudnn_lstm_state(initial_state),
          training=self._is_training)
      with tf.variable_scope('decoder'):
        rnn_output = self._output_layer(rnn_output)

      results = lstm_utils.LstmDecodeResults(
          rnn_input=helper.inputs[:, :, :self._output_depth],
          rnn_output=tf.transpose(rnn_output, [1, 0, 2]),
          samples=None,
          final_state=None,
          final_sequence_lengths=helper.sequence_length)
    else:
      if self._cudnn_dec_lstm:
        tf.logging.warning(
            'CudnnLSTM does not support sampling. Using `dynamic_decode` '
            'instead.')
      decoder = lstm_utils.Seq2SeqLstmDecoder(
          self._dec_cell,
          helper,
          initial_state=initial_state,
          input_shape=input_shape,
          output_layer=self._output_layer)
      final_output, final_state, final_lengths = seq2seq.dynamic_decode(
          decoder,
          maximum_iterations=max_length,
          swap_memory=True,
          scope='decoder')
      results = lstm_utils.LstmDecodeResults(
          rnn_input=final_output.rnn_input[:, :, :self._output_depth],
          rnn_output=final_output.rnn_output,
          samples=final_output.sample_id,
          final_state=final_state,
          final_sequence_lengths=final_lengths)

    return results

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
        `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
        sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
        `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
          `[batch_size, max(x_length), control_depth]`. Required if conditioning
          on control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      truths: Ground truth labels.
      predictions: Predicted labels.
      decode_results: The LstmDecodeResults.
    """
    batch_size = x_input.shape[0].value

    has_z = z is not None
    z = tf.zeros([batch_size, 0]) if z is None else z
    repeated_z = tf.tile(
        tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])

    has_control = c_input is not None
    if c_input is None:
      c_input = tf.zeros([batch_size, tf.shape(x_input)[1], 0])

    sampling_probability_static = tensor_util.constant_value(
        self._sampling_probability)
    if sampling_probability_static == 0.0:
      # Use teacher forcing.
      x_input = tf.concat([x_input, repeated_z, c_input], axis=2)
      helper = seq2seq.TrainingHelper(x_input, x_length)
    else:
      # Use scheduled sampling.
      if has_z or has_control:
        auxiliary_inputs = tf.zeros([batch_size, tf.shape(x_input)[1], 0])
        if has_z:
          auxiliary_inputs = tf.concat([auxiliary_inputs, repeated_z], axis=2)
        if has_control:
          auxiliary_inputs = tf.concat([auxiliary_inputs, c_input], axis=2)
      else:
        auxiliary_inputs = None
      helper = seq2seq.ScheduledOutputTrainingHelper(
          inputs=x_input,
          sequence_length=x_length,
          auxiliary_inputs=auxiliary_inputs,
          sampling_probability=self._sampling_probability,
          next_inputs_fn=self._sample)

    decode_results = self._decode(
        z, helper=helper, input_shape=helper.inputs.shape[2:])
    flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
    flat_rnn_output = flatten_maybe_padded_sequences(
        decode_results.rnn_output, x_length)
    r_loss, metric_map, truths, predictions = self._flat_reconstruction_loss(
        flat_x_target, flat_rnn_output)

    # Sum loss over sequences.
    cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
    r_losses = []
    for i in range(batch_size):
      b, e = cum_x_len[i], cum_x_len[i + 1]
      r_losses.append(tf.reduce_sum(r_loss[b:e]))
    r_loss = tf.stack(r_losses)

    return r_loss, metric_map, truths, predictions, decode_results

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
             start_inputs=None, end_fn=None):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      temperature: (Optional) The softmax temperature to use when sampling, if
        applicable.
      start_inputs: (Optional) Initial inputs to use for batch.
        Sized `[n, output_depth]`.
      end_fn: (Optional) A callable that takes a batch of samples (sized
        `[n, output_depth]` and emits a `bool` vector
        shaped `[batch_size]` indicating whether each sample is an end token.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
      final_state: The final states of the decoder.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`.
    """
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    # Use a dummy Z in unconditional case.
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    if c_input is not None:
      # Tile control sequence across samples.
      c_input = tf.tile(tf.expand_dims(c_input, 1), [1, n, 1])

    # If not given, start with zeros.
    start_inputs = start_inputs if start_inputs is not None else tf.zeros(
        [n, self._output_depth], dtype=tf.float32)
    # In the conditional case, also concatenate the Z.
    start_inputs = tf.concat([start_inputs, z], axis=-1)
    if c_input is not None:
      start_inputs = tf.concat([start_inputs, c_input[0]], axis=-1)
    initialize_fn = lambda: (tf.zeros([n], tf.bool), start_inputs)

    sample_fn = lambda time, outputs, state: self._sample(outputs, temperature)
    end_fn = end_fn or (lambda x: False)

    def next_inputs_fn(time, outputs, state, sample_ids):
      del outputs
      finished = end_fn(sample_ids)
      next_inputs = tf.concat([sample_ids, z], axis=-1)
      if c_input is not None:
        next_inputs = tf.concat([next_inputs, c_input[time]], axis=-1)
      return (finished, next_inputs, state)

    sampler = seq2seq.CustomHelper(
        initialize_fn=initialize_fn, sample_fn=sample_fn,
        next_inputs_fn=next_inputs_fn, sample_ids_shape=[self._output_depth],
        sample_ids_dtype=tf.float32)

    decode_results = self._decode(
        z, helper=sampler, input_shape=start_inputs.shape[1:],
        max_length=max_length)

    return decode_results.samples, decode_results


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
    sampler = tf.contrib.distributions.OneHotCategorical(
        logits=rnn_output / temperature, dtype=tf.float32)
    return sampler.sample()

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=None,
             start_inputs=None, beam_width=None, end_token=None):
    """Overrides BaseLstmDecoder `sample` method to add optional beam search.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      temperature: (Optional) The softmax temperature to use when not doing beam
        search. Defaults to 1.0. Ignored when `beam_width` is provided.
      start_inputs: (Optional) Initial inputs to use for batch.
        Sized `[n, output_depth]`.
      beam_width: (Optional) Width of beam to use for beam search. Beam search
        is disabled if not provided.
      end_token: (Optional) Scalar token signaling the end of the sequence to
        use for early stopping.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
      final_state: The final states of the decoder.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`.
    """
    if beam_width is None:
      end_fn = (None if end_token is None else
                lambda x: tf.equal(tf.argmax(x, axis=-1), end_token))
      return super(CategoricalLstmDecoder, self).sample(
          n, max_length, z, c_input, temperature, start_inputs, end_fn)

    # If `end_token` is not given, use an impossible value.
    end_token = self._output_depth if end_token is None else end_token
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    if temperature is not None:
      tf.logging.warning('`temperature` is ignored when using beam search.')
    # Use a dummy Z in unconditional case.
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    # If not given, start with dummy `-1` token and replace with zero vectors in
    # `embedding_fn`.
    start_tokens = (
        tf.argmax(start_inputs, axis=-1, output_type=tf.int32)
        if start_inputs is not None else
        -1 * tf.ones([n], dtype=tf.int32))

    initial_state = lstm_utils.initial_cell_state_from_embedding(
        self._dec_cell, z, name='decoder/z_to_initial_state')
    beam_initial_state = seq2seq.tile_batch(
        initial_state, multiplier=beam_width)

    # Tile `z` across beams.
    beam_z = tf.tile(tf.expand_dims(z, 1), [1, beam_width, 1])

    def embedding_fn(tokens):
      # If tokens are the start_tokens (negative), replace with zero vectors.
      next_inputs = tf.cond(
          tf.less(tokens[0, 0], 0),
          lambda: tf.zeros([n, beam_width, self._output_depth]),
          lambda: tf.one_hot(tokens, self._output_depth))

      # Concatenate `z` to next inputs.
      next_inputs = tf.concat([next_inputs, beam_z], axis=-1)
      return next_inputs

    decoder = seq2seq.BeamSearchDecoder(
        self._dec_cell,
        embedding_fn,
        start_tokens,
        end_token,
        beam_initial_state,
        beam_width,
        output_layer=self._output_layer,
        length_penalty_weight=0.0)

    final_output, final_state, final_lengths = seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=max_length,
        swap_memory=True,
        scope='decoder')

    samples = tf.one_hot(final_output.predicted_ids[:, :, 0],
                         self._output_depth)
    # Rebuild the input by combining the inital input with the sampled output.
    initial_inputs = (
        tf.zeros([n, 1, self._output_depth]) if start_inputs is None else
        tf.expand_dims(start_inputs, axis=1))
    rnn_input = tf.concat([initial_inputs, samples[:, :-1]], axis=1)

    results = lstm_utils.LstmDecodeResults(
        rnn_input=rnn_input,
        rnn_output=None,
        samples=samples,
        final_state=nest.map_structure(
            lambda x: x[:, 0], final_state.cell_state),
        final_sequence_lengths=final_lengths[:, 0])
    return samples, results


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
    self._is_training = is_training

    for j, (cd, od) in enumerate(zip(self._core_decoders, self._output_depths)):
      with tf.variable_scope('core_decoder_%d' % j):
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
      num_steps = h_size // len(embeddings)
      all_outputs = []
      with tf.variable_scope('hierarchical_layer_%d' % i) as scope:
        cell = lstm_utils.rnn_cell(
            hparams.dec_rnn_size, hparams.dropout_keep_prob, self._is_training)
        cudnn_cell = lstm_utils.cudnn_lstm_layer(
            hparams.dec_rnn_size, hparams.dropout_keep_prob, self._is_training)
        for e in embeddings:
          e.set_shape([batch_size] + e.shape[1:].as_list())
          initial_state = lstm_utils.initial_cell_state_from_embedding(
              cell, e, name='e_to_initial_state')
          if hparams.use_cudnn:
            input_ = tf.zeros([num_steps, batch_size, 1])
            outputs, _ = cudnn_cell(
                input_,
                initial_state=lstm_utils.cudnn_lstm_state(initial_state),
                training=self._is_training)
            outputs = tf.unstack(outputs)
          else:
            input_ = [tf.zeros([batch_size, 1])] * num_steps
            outputs, _ = tf.nn.static_rnn(
                cell, input_, initial_state=initial_state)
          all_outputs.extend(outputs)
          # Reuse layer next time.
          scope.reuse_variables()
      embeddings = all_outputs
    return embeddings

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    if c_input is not None:
      raise ValueError('Control sequence unsupported in hierarchical decoder.')

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
    all_decode_results = []
    for j, loss_outputs_j in enumerate(loss_outputs):
      r_losses, _, truth, predictions, decode_results = zip(*loss_outputs_j)
      all_r_losses.append(tf.reduce_sum(r_losses, axis=0))
      all_truth.append(tf.concat(truth, axis=-1))
      all_predictions.append(tf.concat(predictions, axis=-1))
      metric_map['metrics/accuracy_%d' % j] = tf.metrics.accuracy(
          all_truth[-1], all_predictions[-1])
      metric_map['metrics/mean_per_class_accuracy_%d' % j] = (
          tf.metrics.mean_per_class_accuracy(
              all_truth[-1], all_predictions[-1], self._output_depths[j]))
      all_decode_results.append(decode_results)

    return (tf.reduce_sum(all_r_losses, axis=0),
            metric_map,
            tf.stack(all_truth, axis=-1),
            tf.stack(all_predictions, axis=-1),
            all_decode_results)

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
             **core_sampler_kwargs):
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    if max_length is None:
      # TODO(adarob): Support variable length outputs.
      raise ValueError(
          'HierarchicalMultiOutLstmDecoder requires `max_length` be provided '
          'during sampling.')

    if c_input is not None:
      raise ValueError('Control sequence unsupported in hierarchical decoder.')

    embeddings = self._hierarchical_decode(z)

    sample_ids = []
    decode_results = []
    for j, cd in enumerate(self._core_decoders):
      sample_ids_j = []
      decode_results_j = []
      with tf.variable_scope('core_decoder_%d' % j) as scope:
        for e in embeddings:
          sample_ids_j_e, decode_results_j_e = cd.sample(
              n,
              max_length // len(embeddings),
              z=e,
              temperature=temperature,
              start_inputs=(
                  sample_ids_j[-1][:, -1] if sample_ids_j else None),
              **core_sampler_kwargs)
          sample_ids_j.append(sample_ids_j_e)
          decode_results_j.append(decode_results_j_e)
          scope.reuse_variables()
        sample_ids.append(tf.concat(sample_ids_j, axis=1))
        decode_results.append(decode_results_j)

    return tf.concat(sample_ids, axis=-1), decode_results


class MultiLabelRnnNadeDecoder(BaseLstmDecoder):
  """LSTM decoder with multi-label output provided by a NADE."""

  def build(self, hparams, output_depth, is_training=False):
    self._nade = Nade(
        output_depth, hparams.nade_num_hidden, name='decoder/nade')
    super(MultiLabelRnnNadeDecoder, self).build(
        hparams, output_depth, is_training)
    # Overwrite output layer for NADE parameterization.
    self._output_layer = layers_core.Dense(
        self._nade.num_hidden + output_depth, name='output_projection')

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    b_enc, b_dec = tf.split(
        flat_rnn_output,
        [self._nade.num_hidden, self._output_depth], axis=1)
    ll, cond_probs = self._nade.log_prob(
        flat_x_target, b_enc=b_enc, b_dec=b_dec)
    r_loss = -ll
    flat_truth = tf.cast(flat_x_target, tf.bool)
    flat_predictions = tf.greater_equal(cond_probs, 0.5)

    metric_map = {
        'metrics/accuracy':
            tf.metrics.mean(
                tf.reduce_all(tf.equal(flat_truth, flat_predictions), axis=-1)),
        'metrics/recall':
            tf.metrics.recall(flat_truth, flat_predictions),
        'metrics/precision':
            tf.metrics.precision(flat_truth, flat_predictions),
    }

    return r_loss, metric_map, flat_truth, flat_predictions

  def _sample(self, rnn_output, temperature=None):
    """Sample from NADE, returning the argmax if no temperature is provided."""
    b_enc, b_dec = tf.split(
        rnn_output, [self._nade.num_hidden, self._output_depth], axis=1)
    sample, _ = self._nade.sample(
        b_enc=b_enc, b_dec=b_dec, temperature=temperature)
    return sample


def get_default_hparams():
  """Returns copy of default HParams for LSTM models."""
  hparams_map = base_model.get_default_hparams().values()
  hparams_map.update({
      'conditional': True,
      'dec_rnn_size': [512],  # Decoder RNN: number of units per layer.
      'enc_rnn_size': [256],  # Encoder RNN: number of units per layer per dir.
      'dropout_keep_prob': 1.0,  # Probability all dropout keep.
      'sampling_schedule': 'constant',  # constant, exponential, inverse_sigmoid
      'sampling_rate': 0.0,  # Interpretation is based on `sampling_schedule`.
      'use_cudnn': False,  # Uses faster CudnnLSTM to train. For GPU only.
  })
  return tf.contrib.training.HParams(**hparams_map)
