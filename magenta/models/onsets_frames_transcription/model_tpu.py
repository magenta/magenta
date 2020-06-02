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

"""Onset-focused model for piano and drum transcription, TPU compatible."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.contrib import rnn as contrib_rnn
from magenta.contrib import training as contrib_training
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import estimator_spec_util
import tensorflow.compat.v1 as tf
import tf_slim as slim


def conv_net(inputs, hparams):
  """Builds the ConvNet from Kelz 2016."""
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=slim.variance_scaling_initializer(
          factor=2.0, mode='FAN_AVG', uniform=True)):

    net = inputs
    i = 0
    for (conv_temporal_size, conv_freq_size,
         num_filters, freq_pool_size, dropout_amt) in zip(
             hparams.temporal_sizes, hparams.freq_sizes, hparams.num_filters,
             hparams.pool_sizes, hparams.dropout_keep_amts):
      net = slim.conv2d(
          net,
          num_filters, [conv_temporal_size, conv_freq_size],
          scope='conv' + str(i),
          normalizer_fn=slim.batch_norm)
      if freq_pool_size > 1:
        net = slim.max_pool2d(
            net, [1, freq_pool_size],
            stride=[1, freq_pool_size],
            scope='pool' + str(i))
      if dropout_amt < 1:
        net = slim.dropout(net, dropout_amt, scope='dropout' + str(i))
      i += 1

    # Flatten while preserving batch and time dimensions.
    dims = tf.shape(net)
    net = tf.reshape(
        net, (dims[0], dims[1], net.shape[2] * net.shape[3]),
        'flatten_end')

    net = slim.fully_connected(net, hparams.fc_size, scope='fc_end')
    net = slim.dropout(net, hparams.fc_dropout_keep_amt, scope='dropout_end')

    return net


def lstm_layer(inputs,
               num_units,
               bidirectional,
               is_training,
               lengths=None,
               stack_size=1,
               dropout_keep_prob=1):
  """Create a LSTM layer using the specified backend."""
  cells_fw = []
  for i in range(stack_size):
    del i
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob if is_training else 1.0)
    cells_fw.append(cell)

  if bidirectional:
    cells_bw = []
    for i in range(stack_size):
      del i
      cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, output_keep_prob=dropout_keep_prob if is_training else 1.0)
      cells_bw.append(cell)
    with tf.variable_scope('lstm'):
      (outputs, unused_state_f,
       unused_state_b) = contrib_rnn.stack_bidirectional_dynamic_rnn(
           cells_fw,
           cells_bw,
           inputs,
           dtype=tf.float32,
           sequence_length=lengths,
           parallel_iterations=1)

    return outputs
  else:
    with tf.variable_scope('lstm'):
      outputs, unused_state = tf.nn.dynamic_rnn(
          cell=tf.nn.rnn_cell.MultiRNNCell(cells_fw),
          inputs=inputs,
          dtype=tf.float32,
          sequence_length=lengths,
          parallel_iterations=1)
      return outputs


def lstm_layer_static_for_tflite(inputs,
                                 num_units,
                                 bidirectional,
                                 is_training,
                                 lengths=None,
                                 stack_size=1,
                                 dropout_keep_prob=1):
  """Create a LSTM layer using bidirectional static RNNs."""
  # Uses tflite compatible LSTM cell. The graph is name-compatible
  # with the graph created by lstm_layer.

  assert dropout_keep_prob == 1  # Dropout not implemented for this mode.
  del is_training  # Only used for dropout.

  # Static RNNs will take a list of input tensors instead of a single block
  # tensor. I.e. we need to convert from Tensor(size=[batch, time, width]) to
  # a list=[Tensor(size=[batch, width])] of length 'time'
  split_inputs = [
      tf.squeeze(t, axis=1)
      for t in tf.split(inputs, num_or_size_splits=inputs.shape[1], axis=1)
  ]

  # Here we use the basic LSTMCell which tflite knows about and fudging it so
  # that it sets it's scopes as if it was a
  # tf.contrib.cudnn_rnn.BasicLSTMCell
  cell_type = tf.nn.rnn_cell.LSTMCell
  cells_fw = [
      cell_type(num_units, name='basic_lstm_cell') for _ in range(stack_size)
  ]

  prev_layer = split_inputs
  if bidirectional:
    cells_bw = [
        cell_type(num_units, name='basic_lstm_cell') for _ in range(stack_size)
    ]

    # This emulates the scoping that contrib.rnn.stack_bidirectional_dynamic_rnn
    # does.
    with tf.variable_scope('lstm'):
      with tf.variable_scope('stack_bidirectional_rnn'):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
          with tf.variable_scope('cell_%d' % i):
            prev_layer, _, _ = tf.nn.static_bidirectional_rnn(
                cell_fw,
                cell_bw,
                inputs=prev_layer,
                sequence_length=lengths,
                dtype=tf.float32)
    outputs = prev_layer

  else:  # not bidirectional
    with tf.variable_scope('lstm'):
      outputs, unused_state = tf.nn.static_rnn(
          cell=tf.nn.rnn_cell.MultiRNNCell(cells_fw),
          inputs=prev_layer,
          sequence_length=lengths,
          dtype=tf.float32)

  # We gotta undo our earlier split and fuse it back into a single tensor.
  outputs = tf.stack(outputs, axis=1)
  return outputs


def acoustic_model(inputs, hparams, lstm_units, lengths, is_training):
  """Acoustic model that handles all specs for a sequence in one window."""
  conv_output = conv_net(inputs, hparams)
  if not lstm_units:
    return conv_output

  # default to lstm_layer
  if hparams.use_tflite_compatible:
    lstm_layer_builder = lstm_layer_static_for_tflite
  else:
    lstm_layer_builder = lstm_layer

  return lstm_layer_builder(
      conv_output,
      lstm_units,
      hparams.bidirectional,
      is_training=is_training,
      lengths=lengths if hparams.use_lengths else None,
      stack_size=hparams.acoustic_rnn_stack_size,
      dropout_keep_prob=hparams.acoustic_rnn_dropout_keep_prob)


def build_model(spec, length, hparams, is_training):
  """Builds a raw, API-independent onsets & frames."""

  if hparams.stop_activation_gradient and not hparams.activation_loss:
    raise ValueError(
        'If stop_activation_gradient is true, activation_loss must be true.')

  with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
    with tf.variable_scope('onsets'):
      onset_outputs = acoustic_model(
          spec,
          hparams,
          lstm_units=hparams.onset_lstm_units,
          lengths=length,
          is_training=is_training)
      onset_logits = slim.fully_connected(
          onset_outputs,
          constants.MIDI_PITCHES,
          activation_fn=None,
          scope='onset_logits')

    offset_logits = []
    if hparams.offset_network:
      with tf.variable_scope('offsets'):
        offset_outputs = acoustic_model(
            spec,
            hparams,
            lstm_units=hparams.offset_lstm_units,
            lengths=length,
            is_training=is_training)
        offset_logits = slim.fully_connected(
            offset_outputs,
            constants.MIDI_PITCHES,
            activation_fn=None,
            scope='offset_logits')

    with tf.variable_scope('velocity'):
      velocity_outputs = acoustic_model(
          spec,
          hparams,
          lstm_units=hparams.velocity_lstm_units,
          lengths=length,
          is_training=is_training)
      velocity_values = slim.fully_connected(
          velocity_outputs,
          constants.MIDI_PITCHES,
          activation_fn=None,
          scope='onset_velocities')

    with tf.variable_scope('frame'):
      if not hparams.share_conv_features:
        # TODO(eriche): this is broken when hparams.frame_lstm_units > 0
        activation_outputs = acoustic_model(
            spec,
            hparams,
            lstm_units=hparams.frame_lstm_units,
            lengths=length,
            is_training=is_training)
        activation_logits = slim.fully_connected(
            activation_outputs,
            constants.MIDI_PITCHES,
            activation_fn=None,
            scope='activation_logits')
      else:
        activation_logits = slim.fully_connected(
            onset_outputs,
            constants.MIDI_PITCHES,
            activation_fn=None,
            scope='activation_logits')

      logits = []
      if hparams.stop_onset_gradient:
        logits.append(tf.stop_gradient(onset_logits))
      else:
        logits.append(onset_logits)

      if hparams.stop_activation_gradient:
        logits.append(tf.stop_gradient(activation_logits))
      else:
        logits.append(activation_logits)

      if hparams.offset_network:
        if hparams.stop_offset_gradient:
          logits.append(tf.stop_gradient(offset_logits))
        else:
          logits.append(offset_logits)

      combined_logits = tf.concat(logits, 2)

      if hparams.combined_lstm_units > 0:
        if hparams.use_tflite_compatible:
          lstm_layer_builder = lstm_layer_static_for_tflite
        else:
          lstm_layer_builder = lstm_layer

        outputs = lstm_layer_builder(
            tf.sigmoid(combined_logits),
            hparams.combined_lstm_units,
            hparams.bidirectional,
            is_training=is_training,
            lengths=length if hparams.use_lengths else None,
            stack_size=hparams.combined_rnn_stack_size,
            dropout_keep_prob=hparams.combined_rnn_dropout_keep_prob)
      else:
        outputs = combined_logits

      frame_logits = slim.fully_connected(
          outputs,
          constants.MIDI_PITCHES,
          activation_fn=None,
          scope='frame_logits')

  return frame_logits, onset_logits, offset_logits, velocity_values


def model_fn(features, labels, mode, params, config):
  """Builds the acoustic (for Estimator API)."""
  del config
  hparams = params

  length = features.length
  spec = features.spec
  is_training = mode == tf.estimator.ModeKeys.TRAIN

  frame_logits, onset_logits, offset_logits, velocity_values = build_model(
      spec, length, hparams, is_training)

  # Hack to restore the batch dimension, which is lost in some cases.
  def fix_shape(output):
    shape = list(output.shape)
    shape[0] = spec.shape[0]
    output.set_shape(shape)
  for output in (frame_logits, onset_logits, offset_logits, velocity_values):
    fix_shape(output)

  return estimator_spec_util.get_estimator_spec(
      hparams, mode, features, labels, frame_logits, onset_logits,
      offset_logits, velocity_values, offset_network=hparams.offset_network)


def get_default_hparams():
  """Returns the default hyperparameters.

  Returns:
    A tf.contrib.training.HParams object representing the default
    hyperparameters for the model.
  """
  return contrib_training.HParams(
      batch_size=8,
      learning_rate=0.0006,
      decay_steps=10000,
      decay_rate=0.98,
      clip_norm=3.0,
      transform_audio=False,
      max_expected_train_example_len=625,  # 20 seconds
      offset_network=True,
      onset_lstm_units=256,
      offset_lstm_units=256,
      velocity_lstm_units=0,
      frame_lstm_units=0,
      combined_lstm_units=256,
      bidirectional=True,
      acoustic_rnn_stack_size=1,
      acoustic_rnn_dropout_keep_prob=1.0,
      combined_rnn_stack_size=1,
      combined_rnn_dropout_keep_prob=1.0,
      activation_loss=False,
      stop_activation_gradient=False,
      stop_onset_gradient=True,
      stop_offset_gradient=True,
      weight_frame_and_activation_loss=False,
      share_conv_features=False,
      temporal_sizes=[3, 3, 3],
      freq_sizes=[3, 3, 3],
      num_filters=[48, 48, 96],
      pool_sizes=[1, 2, 2],
      dropout_keep_amts=[1.0, 0.25, 0.25],
      fc_size=768,
      fc_dropout_keep_amt=0.5,
      use_tflite_compatible=False,
      use_lengths=False,
      predict_frame_threshold=0.5,
      predict_onset_threshold=0.5,
      predict_offset_threshold=0,
  )
