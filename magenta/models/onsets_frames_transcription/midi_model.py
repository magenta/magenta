from __future__ import absolute_import, division, print_function

import os

import sklearn
from dotmap import DotMap

# if not using plaidml, use tensorflow.keras.* instead of keras.*
# if using plaidml, use keras.*
import tensorflow.compat.v1 as tf
from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription.accuracy_util import binary_accuracy_wrapper, \
    f1_wrapper, true_positive_wrapper
from magenta.models.onsets_frames_transcription.loss_util import log_loss_wrapper, \
    log_loss_flattener
from sklearn.metrics import f1_score

FLAGS = tf.app.flags.FLAGS

if FLAGS.using_plaidml:
    import plaidml.keras

    plaidml.keras.install_backend()

    from keras import backend as K
    from keras.initializers import VarianceScaling
    from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, \
    Input, MaxPooling2D, Reshape, concatenate, ELU
    from keras.models import Model
else:
    # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    from tensorflow.keras import backend as K
    from tensorflow.keras.initializers import VarianceScaling
    from tensorflow.keras.layers import Activation, BatchNormalization, Bidirectional, Conv2D, \
        Dense, Dropout, \
        Input, LSTM, MaxPooling2D, Reshape, concatenate, ELU
    from tensorflow.keras.models import Model

from magenta.models.onsets_frames_transcription import constants


def get_default_hparams():
    return {
        'batch_size': 3,
        'learning_rate': 0.0006,
        'decay_steps': 10000,
        'decay_rate': 1e-4,
        'clip_norm': 1.0,
        'transform_audio': False,
        'onset_lstm_units': 256,
        'offset_lstm_units': 256,
        'velocity_lstm_units': 0,
        'frame_lstm_units': 0,
        'combined_lstm_units': 256,
        'acoustic_rnn_stack_size': 1,
        'combined_rnn_stack_size': 1,
        'activation_loss': False,
        'stop_activation_gradient': False,
        'stop_onset_gradient': True,
        'stop_offset_gradient': True,
        'weight_frame_and_activation_loss': False,
        'share_conv_features': False,
        'temporal_sizes': [3, 3, 3],
        'freq_sizes': [3, 3, 3],
        'num_filters': [48, 48, 96],
        'pool_sizes': [1, 2, 2],
        'dropout_drop_amts': [0.0, 0.25, 0.25],
        'fc_size': 768,
        'fc_dropout_drop_amt': 0.25,
        'use_lengths': False,
        'use_cudnn': True,
        'rnn_dropout_drop_amt': 0.0,
        'bidirectional': True,
        'predict_frame_threshold': 0.5,
        'predict_onset_threshold': 0.5,
        'predict_offset_threshold': 0.0,
        'frames_true_weighing': 2,
        'onsets_true_weighing': 8,
        'offsets_true_weighing': 8,
        'input_shape': (None, 229, 1),  # (None, 229, 1),
    }


def lstm_layer(num_units,
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
                kernel_initializer=VarianceScaling(2, distribution='uniform'))
            )(lstm_stack)
        return lstm_stack

    return lstm_layer_fn


def acoustic_dense_layer(hparams, lstm_units):
    def acoustic_dense_fn(inputs):
        # shape: (None, None, 57, 96)
        outputs = Dense(hparams.fc_size, use_bias=False,
                                   activation='sigmoid',
                                   kernel_initializer=VarianceScaling(scale=2, mode='fan_avg',
                                                                      distribution='uniform'))(
            # Flatten while preserving batch and time dimensions.
            Reshape((-1, K.int_shape(inputs)[2] * K.int_shape(inputs)[3]))(
                inputs))
        outputs = Dropout(hparams.fc_dropout_drop_amt)(outputs)

        if lstm_units:
            outputs = lstm_layer(lstm_units, stack_size=hparams.acoustic_rnn_stack_size)(outputs)
        return outputs

    return acoustic_dense_fn


def acoustic_model_layer(hparams):
    bn_relu_fn = lambda x: ELU(hparams.timbre_leaky_alpha)(BatchNormalization(scale=False)(x))
    conv_bn_relu_layer = lambda num_filters, conv_temporal_size, conv_freq_size: lambda \
            x: bn_relu_fn(
        Conv2D(
            num_filters,
            [conv_temporal_size, conv_freq_size],
            padding='same',
            use_bias=False,
            kernel_initializer=VarianceScaling(scale=2, mode='fan_avg', distribution='uniform')
        )(x))

    def acoustic_model_fn(inputs):
        # inputs should be of type keras.layers.Input
        outputs = inputs

        for (conv_temporal_size, conv_freq_size,
             num_filters, freq_pool_size, dropout_amt) in zip(
            hparams.temporal_sizes, hparams.freq_sizes, hparams.num_filters,
            hparams.pool_sizes, hparams.dropout_drop_amts):

            outputs = conv_bn_relu_layer(num_filters, conv_freq_size, conv_temporal_size)(outputs)
            if freq_pool_size > 1:
                outputs = MaxPooling2D([1, freq_pool_size], strides=[1, freq_pool_size])(outputs)
            if dropout_amt > 0:
                outputs = Dropout(dropout_amt)(outputs)
        return outputs

    return acoustic_model_fn


def midi_pitches_layer(name=None):
    def midi_pitches_fn(inputs):
        # inputs should be of type keras.layers.*
        outputs = Dense(constants.MIDI_PITCHES, activation='sigmoid', name=name)(inputs)
        return outputs

    return midi_pitches_fn


def midi_prediction_model(hparams=None):
    if hparams is None:
        hparams = DotMap(get_default_hparams())
    inputs = Input(shape=(hparams.input_shape[0], hparams.input_shape[1], hparams.input_shape[2],),
                   name='spec')
    weights = Input(shape=(None, constants.MIDI_PITCHES,), name='weights')

    # if K.learning_phase():

    # Onset prediction model
    onset_conv = acoustic_model_layer(hparams)(inputs)

    onset_outputs = acoustic_dense_layer(hparams, hparams.onset_lstm_units)(onset_conv)

    onset_probs = midi_pitches_layer('onsets')(onset_outputs)

    # Offset prediction model
    offset_conv = acoustic_model_layer(hparams)(inputs)
    offset_outputs = acoustic_dense_layer(hparams, hparams.offset_lstm_units)(offset_conv)

    offset_probs = midi_pitches_layer('offsets')(offset_outputs)

    # Activation prediction model
    if not hparams.share_conv_features:
        activation_conv = acoustic_model_layer(hparams)(inputs)
        activation_outputs = acoustic_dense_layer(hparams, hparams.frame_lstm_units)(
            activation_conv)
    else:
        activation_outputs = onset_outputs
    activation_probs = midi_pitches_layer('activations')(activation_outputs)

    probs = []
    probs.append(K.stop_gradient(onset_probs))
    probs.append(K.stop_gradient(offset_probs))
    probs.append(activation_probs)

    combined_probs = concatenate(probs, 2)

    if hparams.combined_lstm_units > 0:
        outputs = lstm_layer(hparams.combined_lstm_units,
                             stack_size=hparams.combined_rnn_stack_size)(combined_probs)
    else:
        outputs = combined_probs

    # Frame prediction
    frame_probs = midi_pitches_layer('frames')(outputs)

    # use recall_weighing > 0 to care more about recall than precision
    losses = {
        'frames': log_loss_wrapper(hparams.frames_true_weighing),
        'onsets': log_loss_wrapper(hparams.onsets_true_weighing),
        'offsets': log_loss_wrapper(hparams.offsets_true_weighing),
    }

    accuracies = {
        'frames': [binary_accuracy_wrapper(hparams.predict_frame_threshold),
                   f1_wrapper(hparams.predict_frame_threshold)],
        'onsets': [binary_accuracy_wrapper(hparams.predict_onset_threshold),
                   f1_wrapper(hparams.predict_onset_threshold)],
        'offsets': [binary_accuracy_wrapper(hparams.predict_offset_threshold),
                    f1_wrapper(hparams.predict_offset_threshold)]
    }

    return Model(inputs=[
        inputs,
        # weights,
        # Input(shape=(None,)),
    ],
        outputs=[frame_probs, onset_probs, offset_probs]), \
           losses, \
           accuracies
