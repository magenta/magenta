from __future__ import absolute_import, division, print_function

# if not using plaidml, use tensorflow.keras.* instead of keras.*
# if using plaidml, use keras.*
import functools

import tensorflow.compat.v1 as tf
from dotmap import DotMap
from magenta.models.onsets_frames_transcription import constants
from sklearn.metrics import f1_score

FLAGS = tf.app.flags.FLAGS

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal, VarianceScaling
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, \
    Dense, Dropout, \
    Input, MaxPooling2D, concatenate, Lambda, Multiply, Reshape, Bidirectional, LSTM, \
    LeakyReLU, TimeDistributed, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def get_default_hparams():
    return {
        'nsynth_batch_size': 16,
        'timbre_training_max_instruments': 2,
        'timbre_learning_rate': 0.0006,
        'timbre_decay_steps': 10000,
        'timbre_decay_rate': 0.98,
        'timbre_clip_norm': 3.0,
        'timbre_l2_regularizer': 1e-5,
        'timbre_filter_m_sizes': [5, 80],
        'timbre_filter_n_sizes': [1, 3, 5],
        'timbre_num_filters': [128, 64, 32],
        'timbre_filters_pool_size': (32, int(constants.BINS_PER_OCTAVE / 2)),
        # (int(constants.BINS_PER_OCTAVE/2), 16),#(22, 32),
        'timbre_pool_size': (2, 2),
        'timbre_num_layers': 2,
        'timbre_dropout_keep_amts': [1.0, 0.75, 0.75],
        'timbre_fc_size': 256,
        'timbre_fc_dropout_keep_amt': 0.5,
        'timbre_input_shape': (None, constants.SPEC_BANDS, 1),  # (None, 229, 1),
        'timbre_num_classes': 11,
        'timbre_lstm_units': 128,
        'timbre_rnn_stack_size': 2,
        'timbre_leaky_alpha': 0.33,  # per han et al 2016
        'timbre_max_time': 256,
    }


def filters_layer(hparams, channel_axis):
    def filters_layer_fn(inputs):
        parallel_layers = []

        bn_elu_fn = lambda x: MaxPooling2D(pool_size=hparams.timbre_filters_pool_size) \
            (LeakyReLU(hparams.timbre_leaky_alpha)
             (BatchNormalization(axis=channel_axis, scale=False)
              (x)))
        conv_bn_elu_layer = lambda num_filters, conv_temporal_size, conv_freq_size: lambda \
                x: bn_elu_fn(
            Conv2D(
                num_filters,
                [conv_temporal_size, conv_freq_size],
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(hparams.timbre_l2_regularizer),
                kernel_initializer=he_normal()
            )(x))

        for m_i in hparams.timbre_filter_m_sizes:
            for i, n_i in enumerate(hparams.timbre_filter_n_sizes):
                parallel_layers.append(
                    conv_bn_elu_layer(hparams.timbre_num_filters[i], m_i, n_i)(inputs))

        return concatenate(parallel_layers, channel_axis)

    return filters_layer_fn


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
                return_sequences=i < stack_size - 1,
                recurrent_dropout=rnn_dropout_drop_amt,
                kernel_initializer=VarianceScaling(2, distribution='uniform'))
            )(lstm_stack)
        return lstm_stack

    return lstm_layer_fn


def acoustic_model_layer(hparams, channel_axis):
    def acoustic_model_fn(inputs):
        # inputs should be of type keras.layers.Input
        outputs = inputs

        # batch norm, then leakyrelu activation, then maxpool
        bn_elu_fn = lambda x: MaxPooling2D(pool_size=hparams.timbre_pool_size,
                                           strides=hparams.timbre_pool_size) \
            (LeakyReLU(hparams.timbre_leaky_alpha)
             (BatchNormalization(axis=channel_axis, scale=False)
              (x)))
        # conv2d, then above
        conv_bn_elu_layer = lambda num_filters, conv_temporal_size, conv_freq_size: lambda \
                x: bn_elu_fn(
            Conv2D(
                num_filters,
                [conv_temporal_size, conv_freq_size],
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(hparams.timbre_l2_regularizer),
                kernel_initializer=he_normal()
            )(x))

        for i in range(hparams.timbre_num_layers):
            outputs = Dropout(0.25)(outputs)
            outputs = conv_bn_elu_layer(128, 3, 3)(outputs)

        return outputs

    return acoustic_model_fn


def instrument_prediction_layer(hparams):
    def instrument_prediction_fn(inputs):
        # inputs should be of type keras.layers.*
        # outputs = Flatten()(inputs)
        outputs = inputs
        outputs = Dropout(0.5)(outputs)
        outputs = Dense(hparams.timbre_fc_size,
                        kernel_initializer=he_normal(),
                        kernel_regularizer=l2(hparams.timbre_l2_regularizer))(outputs)
        outputs = Activation('elu')(outputs)
        outputs = Dropout(0.5)(outputs)
        outputs = Dense(hparams.timbre_num_classes,
                        activation='softmax',
                        name='timbre_prediction',
                        kernel_initializer=he_normal(),
                        kernel_regularizer=l2(hparams.timbre_l2_regularizer))(outputs)
        return outputs

    return instrument_prediction_fn


def high_pass_filter(input_list):
    spec = input_list[0]
    hp = input_list[1]

    reset_list = []
    seq_mask = K.cast(tf.math.logical_not(tf.sequence_mask(hp, constants.SPEC_BANDS)), tf.float32)
    seq_mask = K.expand_dims(seq_mask, 3)

    # this doesn't work with batches unfortunately
    # tf.slice(spec, K.cast(concatenate([0 * hp, 0 * hp, hp, 0 * hp]), dtype='int64'),
    #          K.cast(concatenate([0 * hp - 1, 0 * hp - 1, constants.SPEC_BANDS - hp, 0 * hp - 1]),
    #                 dtype='int64'))
    return Multiply()([spec, seq_mask])
'''
def crop_unstacked(input_list):

    conv_output = input_list[0]
    croppings = input_list[1]
    num_croppings = input_list[2]
    outputs = []
    for crop_idx in range(num_croppings.numpy()):
        rs = Cropping2D(cropping=croppings[crop_idx])(
            K.expand_dims(conv_output, 0))
        outputs.append(rs[0])
    return K.concatenate(outputs, axis=0)

def get_all_croppings(input_list, hparams):
    conv_output_list = tf.unstack(input_list[0], num=hparams.nsynth_batch_size)
    croppings_list = tf.unstack(input_list[1], num=hparams.nsynth_batch_size)
    num_croppings_list = tf.unstack(input_list[2], num=hparams.nsynth_batch_size)

    all_outputs = []
    # unbatch
    for batch_idx in range(hparams.nsynth_batch_size):
        all_outputs.append(tf.py_function(crop_unstacked,
                                               [
                                                   conv_output_list[batch_idx],
                                                   croppings_list[batch_idx],
                                                   num_croppings_list[batch_idx]
                                               ],
                                               tf.float32))

    # rebatch
    final_output = K.concatenate(all_outputs, axis=0)
    return final_output
'''
'''
def while_cond(conv_output, cropped_outputs, croppings, i, num_croppings):
    static_val = tf.get_static_value(num_croppings)
    if static_val is None:
        return i > 0
    return i < static_val

def while_body(conv_output, cropped_outputs, croppings, i, num_croppings):
    new_crop = Cropping2D(cropping=croppings[i])(
            K.expand_dims(conv_output, 0))
    if cropped_outputs == []:
        return (conv_output, new_crop, croppings, i + 1, num_croppings)
    concat = K.concatenate([cropped_outputs, new_crop])
    return (conv_output, concat, croppings, i + 1, num_croppings)


def get_all_croppings(input_list, hparams):
    conv_output_list = tf.unstack(input_list[0], num=hparams.nsynth_batch_size)
    croppings_list = tf.unstack(input_list[1], num=hparams.nsynth_batch_size)
    num_croppings_list = tf.unstack(input_list[2], num=hparams.nsynth_batch_size)

    all_outputs = []
    # unbatch
    for batch_idx in range(hparams.nsynth_batch_size):
        repeated_conv_output = tf.repeat(K.expand_dims(conv_output_list[batch_idx], axis=0),
                                         num_croppings_list[batch_idx], axis=0)
        cropped_outputs = tf.while_loop(while_cond, while_body, (conv_output_list[batch_idx], [], croppings_list[batch_idx], 0, num_croppings_list[batch_idx]))[1]
        print(cropped_outputs)



    # rebatch
    final_output = K.concatenate(all_outputs, axis=0)
    return final_output
'''

def get_all_croppings(input_list, hparams):
    conv_output_list = tf.unstack(input_list[0], num=hparams.nsynth_batch_size)
    croppings_list = tf.unstack(input_list[1], num=hparams.nsynth_batch_size)
    num_croppings_list = tf.unstack(input_list[2], num=hparams.nsynth_batch_size)

    all_outputs = []
    # unbatch
    for batch_idx in range(hparams.nsynth_batch_size):
        conv_output = conv_output_list[batch_idx]
        croppings = croppings_list[batch_idx]
        num_croppings = num_croppings_list[batch_idx]
        repeated_conv_output = tf.repeat(K.expand_dims(conv_output, axis=0),
                                         num_croppings, axis=0)

        pitch_mask = K.cast(tf.math.logical_not(tf.sequence_mask(
            K.expand_dims(tf.gather(croppings, indices=0, axis=1), 1), constants.SPEC_BANDS
        )), tf.float32)
        right_mask = K.cast(tf.sequence_mask(
            K.expand_dims(tf.gather(croppings, indices=1, axis=1), 1)
        ), tf.float32)
        left_mask = K.cast(tf.math.logical_not(tf.sequence_mask(
            K.expand_dims(tf.gather(croppings, indices=2, axis=1), 1), maxlen=right_mask.shape[1]
        )), tf.float32)

        # constant time for the pitch mask
        pitch_mask = K.expand_dims(pitch_mask, 3)

        # reorder to mask temporally
        right_mask = K.permute_dimensions(right_mask, (0, 2, 1))
        # constant pitch for right mask
        right_mask = K.expand_dims(right_mask, 2)

        # reorder to mask temporally
        left_mask = K.permute_dimensions(left_mask, (0, 2, 1))
        # constant pitch for left mask
        left_mask = K.expand_dims(left_mask, 2)

        out = Multiply()([repeated_conv_output, pitch_mask])
        out = Multiply()([out, right_mask])
        out = Multiply()([out, left_mask])
        all_outputs.append(K.expand_dims(out, 0))

    return K.concatenate(all_outputs, axis=0)


def timbre_prediction_model(hparams=None):
    if hparams is None:
        hparams = DotMap(get_default_hparams())

    input_shape = (
    hparams.timbre_input_shape[0], hparams.timbre_input_shape[1], hparams.timbre_input_shape[2],)
    channel_axis = 3

    inputs = Input(shape=input_shape,
                   name='spec')
    num_crops = Input(shape=(1,), dtype='int64')

    high_pass_filter([inputs, num_crops])

    # batched dimensions for cropping like:
    # ((top_crop, bottom_crop), (left_crop, right_crop))
    # with a high pass, top_crop will always be 0, bottom crop is relative to pitch
    croppings = Input(shape=(None, 3), name='cropping', dtype='int64')

    acoustic_outputs = acoustic_model_layer(hparams, channel_axis)(inputs)

    cropped_outputs = Lambda(functools.partial(get_all_croppings, hparams=hparams))([inputs, croppings, num_crops])
    filter_outputs = TimeDistributed(filters_layer(hparams, channel_axis), input_shape=())(cropped_outputs)

    # flatten while preserving batch and time dimensions
    timbre_outputs = Reshape((-1, K.int_shape(filter_outputs)[2] * K.int_shape(filter_outputs)[3]))(
        filter_outputs)

    if hparams.timbre_lstm_units:
        timbre_outputs = lstm_layer(hparams.timbre_lstm_units,
                                    stack_size=hparams.timbre_rnn_stack_size)(
            timbre_outputs)

    timbre_probs = instrument_prediction_layer(hparams)(timbre_outputs)

    losses = 'categorical_crossentropy'

    accuracies = ['categorical_accuracy']

    return Model(inputs=[inputs, pitch], outputs=timbre_probs), losses, accuracies
