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

"""Onset-focused model for piano transcription."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports

from . import constants

import tensorflow as tf
import tensorflow.contrib.slim as slim

from magenta.common import flatten_maybe_padded_sequences
from magenta.common import tf_utils


def conv_net_kelz(inputs):
  """Builds the ConvNet from Kelz 2016."""
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_AVG', uniform=True)):
    net = slim.conv2d(
        inputs, 32, [3, 3], scope='conv1', normalizer_fn=slim.batch_norm)

    net = slim.conv2d(
        net, 32, [3, 3], scope='conv2', normalizer_fn=slim.batch_norm)
    net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
    net = slim.dropout(net, 0.25, scope='dropout2')

    net = slim.conv2d(
        net, 64, [3, 3], scope='conv3', normalizer_fn=slim.batch_norm)
    net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3')
    net = slim.dropout(net, 0.25, scope='dropout3')

    # Flatten while preserving batch and time dimensions.
    dims = tf.shape(net)
    net = tf.reshape(net, (dims[0], dims[1],
                           net.shape[2].value * net.shape[3].value), 'flatten4')

    net = slim.fully_connected(net, 512, scope='fc5')
    net = slim.dropout(net, 0.5, scope='dropout5')

    return net


def acoustic_model(inputs, hparams, lstm_units, lengths):
  """Acoustic model that handles all specs for a sequence in one window."""
  conv_output = conv_net_kelz(inputs)

  if lstm_units:
    rnn_cell_fw = tf.contrib.rnn.LSTMBlockCell(lstm_units)
    if hparams.onset_bidirectional:
      rnn_cell_bw = tf.contrib.rnn.LSTMBlockCell(lstm_units)
      outputs, unused_output_states = tf.nn.bidirectional_dynamic_rnn(
          rnn_cell_fw,
          rnn_cell_bw,
          inputs=conv_output,
          sequence_length=lengths,
          dtype=tf.float32)
      combined_outputs = tf.concat(outputs, 2)
    else:
      combined_outputs, unused_output_states = tf.nn.dynamic_rnn(
          rnn_cell_fw,
          inputs=conv_output,
          sequence_length=lengths,
          dtype=tf.float32)

    return combined_outputs
  else:
    return conv_output


def get_model(transcription_data, hparams, is_training=True):
  """Builds the acoustic model."""
  onset_labels = transcription_data.onsets
  velocity_labels = transcription_data.velocities
  frame_labels = transcription_data.labels
  frame_label_weights = transcription_data.label_weights
  lengths = transcription_data.lengths
  spec = transcription_data.spec

  if hparams.stop_activation_gradient and not hparams.activation_loss:
    raise ValueError(
        'If stop_activation_gradient is true, activation_loss must be true.')

  losses = {}
  with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
    with tf.variable_scope('onsets'):
      onset_outputs = acoustic_model(
          spec, hparams, lstm_units=hparams.onset_lstm_units, lengths=lengths)
      onset_probs = slim.fully_connected(
          onset_outputs,
          constants.MIDI_PITCHES,
          activation_fn=tf.sigmoid,
          scope='onset_probs')

      # onset_probs_flat is used during inference.
      onset_probs_flat = flatten_maybe_padded_sequences(onset_probs, lengths)
      onset_labels_flat = flatten_maybe_padded_sequences(onset_labels, lengths)
      tf.identity(onset_probs_flat, name='onset_probs_flat')
      tf.identity(onset_labels_flat, name='onset_labels_flat')
      tf.identity(
          tf.cast(tf.greater_equal(onset_probs_flat, .5), tf.float32),
          name='onset_predictions_flat')

      onset_losses = tf_utils.log_loss(onset_labels_flat, onset_probs_flat)
      tf.losses.add_loss(tf.reduce_mean(onset_losses))
      losses['onset'] = onset_losses

    with tf.variable_scope('velocity'):
      # TODO(eriche): this is broken when hparams.velocity_lstm_units > 0
      velocity_outputs = acoustic_model(
          spec,
          hparams,
          lstm_units=hparams.velocity_lstm_units,
          lengths=lengths)
      velocity_values = slim.fully_connected(
          velocity_outputs,
          constants.MIDI_PITCHES,
          activation_fn=None,
          scope='onset_velocities')

      velocity_values_flat = flatten_maybe_padded_sequences(
          velocity_values, lengths)
      tf.identity(velocity_values_flat, name='velocity_values_flat')
      velocity_labels_flat = flatten_maybe_padded_sequences(
          velocity_labels, lengths)
      velocity_loss = tf.reduce_sum(
          onset_labels_flat *
          tf.square(velocity_labels_flat - velocity_values_flat),
          axis=1)
      tf.losses.add_loss(tf.reduce_mean(velocity_loss))
      losses['velocity'] = velocity_loss

    with tf.variable_scope('frame'):
      if not hparams.share_conv_features:
        # TODO(eriche): this is broken when hparams.frame_lstm_units > 0
        activation_outputs = acoustic_model(
            spec, hparams, lstm_units=hparams.frame_lstm_units, lengths=lengths)
        activation_probs = slim.fully_connected(
            activation_outputs,
            constants.MIDI_PITCHES,
            activation_fn=tf.sigmoid,
            scope='activation_probs')
      else:
        activation_probs = slim.fully_connected(
            onset_outputs,
            constants.MIDI_PITCHES,
            activation_fn=tf.sigmoid,
            scope='activation_probs')

      combined_probs = tf.concat([
          tf.stop_gradient(onset_probs)
          if hparams.stop_onset_gradient else onset_probs,
          tf.stop_gradient(activation_probs)
          if hparams.stop_activation_gradient else activation_probs
      ], 2)

      if hparams.combined_lstm_units > 0:
        rnn_cell_fw = tf.contrib.rnn.LSTMBlockCell(hparams.combined_lstm_units)
        if hparams.frame_bidirectional:
          rnn_cell_bw = tf.contrib.rnn.LSTMBlockCell(
              hparams.combined_lstm_units)
          outputs, unused_output_states = tf.nn.bidirectional_dynamic_rnn(
              rnn_cell_fw, rnn_cell_bw, inputs=combined_probs, dtype=tf.float32)
          combined_outputs = tf.concat(outputs, 2)
        else:
          combined_outputs, unused_output_states = tf.nn.dynamic_rnn(
              rnn_cell_fw, inputs=combined_probs, dtype=tf.float32)
      else:
        combined_outputs = combined_probs

      frame_probs = slim.fully_connected(
          combined_outputs,
          constants.MIDI_PITCHES,
          activation_fn=tf.sigmoid,
          scope='frame_probs')

    frame_labels_flat = flatten_maybe_padded_sequences(frame_labels, lengths)
    frame_probs_flat = flatten_maybe_padded_sequences(frame_probs, lengths)
    tf.identity(frame_probs_flat, name='frame_probs_flat')
    frame_label_weights_flat = flatten_maybe_padded_sequences(
        frame_label_weights, lengths)
    frame_losses = tf_utils.log_loss(
        frame_labels_flat,
        frame_probs_flat,
        weights=frame_label_weights_flat
        if hparams.weight_frame_and_activation_loss else None)
    tf.losses.add_loss(tf.reduce_mean(frame_losses))
    losses['frame'] = frame_losses

    if hparams.activation_loss:
      activation_losses = tf_utils.log_loss(
          frame_labels_flat,
          flatten_maybe_padded_sequences(activation_probs, lengths),
          weights=frame_label_weights_flat
          if hparams.weight_frame_and_activation_loss else None)
      tf.losses.add_loss(tf.reduce_mean(activation_losses))
      losses['activation'] = activation_losses

  predictions_flat = tf.cast(tf.greater_equal(frame_probs_flat, .5), tf.float32)

  # Creates a pianoroll labels in red and probs in green [minibatch, 88]
  images = {}
  onset_pianorolls = tf.concat(
      [
          onset_labels[:, :, :, tf.newaxis], onset_probs[:, :, :, tf.newaxis],
          tf.zeros(tf.shape(onset_labels))[:, :, :, tf.newaxis]
      ],
      axis=3)
  images['OnsetPianorolls'] = onset_pianorolls
  activation_pianorolls = tf.concat(
      [
          frame_labels[:, :, :, tf.newaxis], frame_probs[:, :, :, tf.newaxis],
          tf.zeros(tf.shape(frame_labels))[:, :, :, tf.newaxis]
      ],
      axis=3)
  images['ActivationPianorolls'] = activation_pianorolls

  return (tf.losses.get_total_loss(), losses, frame_labels_flat,
          predictions_flat, images)


def get_default_hparams():
  """Returns the default hyperparameters.

  Returns:
    A tf.HParams object representing the default hyperparameters for the model.
  """
  return tf_utils.merge_hparams(
      constants.DEFAULT_HPARAMS,
      tf.contrib.training.HParams(
          activation_loss=False,
          batch_size=8,
          clip_norm=3,
          combined_lstm_units=128,
          frame_bidirectional=True,
          frame_lstm_units=0,
          learning_rate=0.0006,
          decay_steps=10000,
          decay_rate=0.98,
          min_duration_ms=0,
          min_frame_occupancy_for_label=0.0,
          normalize_audio=False,
          onset_bidirectional=True,
          onset_delay=0,
          onset_length=32,
          onset_lstm_units=128,
          velocity_lstm_units=0,
          onset_mode='length_ms',
          sample_rate=constants.DEFAULT_SAMPLE_RATE,
          share_conv_features=False,
          spec_fmin=30.0,
          spec_hop_length=512,
          spec_log_amplitude=True,
          spec_n_bins=229,
          spec_type='mel',
          stop_activation_gradient=False,
          stop_onset_gradient=False,
          truncated_length=1500,  # 48 seconds
          weight_frame_and_activation_loss=True))
