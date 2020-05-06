# Copyright 2020 Jack Spencer Smith.
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

"""Melodic Transcription Model based on the
Onsets and Frames model: Copyright 2020 The Magenta Authors
"""

from __future__ import absolute_import, division, print_function

from dotmap import DotMap
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Conv2D, Dense, Dropout, ELU, \
    Input, LSTM, Lambda, MaxPooling2D, Reshape, concatenate
from tensorflow.keras.models import Model

from magenta.models.polyamp import constants
from magenta.models.polyamp.accuracy_util import binary_accuracy_wrapper, \
    f1_wrapper
from magenta.models.polyamp.loss_util import melodic_loss_wrapper


def get_default_hparams():
    return {
        'learning_rate': 0.0006,
        'decay_steps': 10000,
        'decay_rate': 1e-5,
        'clip_norm': 1.0,
        'transform_audio': False,
        'onset_lstm_units': 256,
        'offset_lstm_units': 256,
        'velocity_lstm_units': 0,
        'frame_lstm_units': 0,
        'combined_lstm_units': 256,
        'acoustic_rnn_stack_size': 1,
        'combined_rnn_stack_size': 1,
        'share_conv_features': False,
        'temporal_sizes': [3, 3, 3, 3],
        'freq_sizes': [3, 3, 3, 5],
        'num_filters': [48, 48, 96, 96],
        'pool_sizes': [1, 2, 2, 1],
        'dropout_drop_amts': [0.0, 0.5, 0.5, 0.5],
        'use_batch_normalization': False,
        'fc_size': 768,
        'fc_dropout_drop_amt': 0.5,
        'rnn_dropout_drop_amt': 0.0,
        'predict_frame_threshold': 0.5,
        'predict_onset_threshold': 0.5,
        'active_onset_threshold': 0.6,
        'predict_offset_threshold': 0.0,
        'frames_true_weighing': 2,
        'onsets_true_weighing': 8,
        'offsets_true_weighing': 8,
        'input_shape': (None, 229, 1),
        'melodic_leaky_alpha': 0.33,
    }


def lstm_layer(stack_name,
               num_units,
               stack_size=1,
               rnn_dropout_drop_amt=0,
               implementation=2):
    def lstm_layer_fn(inputs):
        lstm_stack = inputs
        for i in range(stack_size):
            lstm_stack = Bidirectional(LSTM(
                num_units,
                recurrent_activation='sigmoid',
                implementation=implementation,
                return_sequences=True,
                recurrent_dropout=rnn_dropout_drop_amt,
                kernel_initializer='he_uniform',
                name=f'{stack_name}_bilstm_{i}')
            )(lstm_stack)
        return lstm_stack

    return lstm_layer_fn


def acoustic_dense_layer(stack_name, hparams, lstm_units):
    def acoustic_dense_fn(inputs):
        # shape: (None, None, 57, 96)
        outputs = Dense(hparams.fc_size, use_bias=False,
                        activation='sigmoid',
                        kernel_initializer='he_uniform',
                        name=f'{stack_name}_dense')(
            # Flatten while preserving batch and time dimensions.
            Reshape((-1, K.int_shape(inputs)[2] * K.int_shape(inputs)[3]))(
                inputs))
        outputs = Dropout(hparams.fc_dropout_drop_amt)(outputs)

        if lstm_units:
            outputs = lstm_layer(stack_name, lstm_units,
                                 stack_size=hparams.acoustic_rnn_stack_size)(outputs)
        return outputs

    return acoustic_dense_fn


def acoustic_model_layer(stack_name, hparams):
    if hparams.use_batch_normalization:
        bn_relu_fn = lambda x, name=None: ELU(hparams.timbre_leaky_alpha)(
            BatchNormalization(scale=False, name=f'{name}_batch_norm')(x))
    else:
        bn_relu_fn = lambda x, name=None: ELU(hparams.melodic_leaky_alpha)(x)

    conv_bn_relu_layer = lambda num_filters, conv_temporal_size, conv_freq_size, name=None: lambda \
            x: bn_relu_fn(
        Conv2D(
            num_filters,
            [conv_temporal_size, conv_freq_size],
            padding='same',
            use_bias=False,
            kernel_initializer='he_uniform',
            name=name
        )(x), name=name)

    def acoustic_model_fn(inputs):
        # inputs should be of type keras.layers.Input
        outputs = inputs

        i = 0
        for (conv_temporal_size, conv_freq_size,
             num_filters, freq_pool_size, dropout_amt) in zip(
            hparams.temporal_sizes, hparams.freq_sizes, hparams.num_filters,
            hparams.pool_sizes, hparams.dropout_drop_amts):

            outputs = conv_bn_relu_layer(num_filters, conv_freq_size, conv_temporal_size,
                                         name=f'{stack_name}_conv_{i}')(outputs)
            if freq_pool_size > 1:
                outputs = MaxPooling2D([1, freq_pool_size], strides=[1, freq_pool_size])(outputs)
            if dropout_amt > 0:
                outputs = Dropout(dropout_amt)(outputs)
            i += 1
        return outputs

    return acoustic_model_fn


def midi_pitches_layer(name=None):
    def midi_pitches_fn(inputs):
        outputs = Dense(constants.MIDI_PITCHES, activation='sigmoid', name=name)(inputs)
        return outputs

    return midi_pitches_fn


def get_melodic_model(hparams=None):
    if hparams is None:
        hparams = DotMap(get_default_hparams())
    inputs = Input(shape=(hparams.input_shape[0], hparams.input_shape[1], hparams.input_shape[2],),
                   name='spec')

    # Onset prediction model.
    onset_conv = acoustic_model_layer('onset', hparams)(inputs)

    onset_outputs = acoustic_dense_layer('onset', hparams, hparams.onset_lstm_units)(onset_conv)

    onset_probs = midi_pitches_layer('onsets')(onset_outputs)

    # Offset prediction model.
    offset_conv = acoustic_model_layer('offset', hparams)(inputs)
    offset_outputs = acoustic_dense_layer('offset', hparams, hparams.offset_lstm_units)(offset_conv)

    offset_probs = midi_pitches_layer('offsets')(offset_outputs)

    # Activation prediction model.
    if not hparams.share_conv_features:
        activation_conv = acoustic_model_layer('activation', hparams)(inputs)
        activation_outputs = acoustic_dense_layer('activation', hparams, hparams.frame_lstm_units)(
            activation_conv)
    else:
        activation_outputs = onset_outputs
    activation_probs = midi_pitches_layer('activations')(activation_outputs)

    probs = []
    probs.append(Lambda(lambda x: K.stop_gradient(x))(onset_probs))
    probs.append(Lambda(lambda x: K.stop_gradient(x))(offset_probs))
    probs.append(activation_probs)

    combined_probs = concatenate(probs, 2)

    if hparams.combined_lstm_units > 0:
        outputs = lstm_layer('frame', hparams.combined_lstm_units,
                             stack_size=hparams.combined_rnn_stack_size)(combined_probs)
    else:
        outputs = combined_probs

    # Frame prediction.
    frame_probs = midi_pitches_layer('frames')(outputs)

    # Use recall_weighing > 0 to care more about recall than precision.
    losses = {
        'frames': melodic_loss_wrapper(hparams.frames_true_weighing),
        'onsets': melodic_loss_wrapper(hparams.onsets_true_weighing),
        'offsets': melodic_loss_wrapper(hparams.offsets_true_weighing),
    }

    accuracies = {
        'frames': [binary_accuracy_wrapper(hparams.predict_frame_threshold),
                   f1_wrapper(hparams.predict_frame_threshold)],
        'onsets': [binary_accuracy_wrapper(hparams.predict_onset_threshold),
                   f1_wrapper(hparams.predict_onset_threshold)],
        'offsets': [binary_accuracy_wrapper(hparams.predict_offset_threshold),
                    f1_wrapper(hparams.predict_offset_threshold)]
    }

    return (
        Model(inputs=[inputs],
              outputs=[frame_probs, onset_probs, offset_probs]),
        losses,
        accuracies
    )
