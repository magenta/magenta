from __future__ import absolute_import, division, print_function

# if not using plaidml, use tensorflow.keras.* instead of keras.*
# if using plaidml, use keras.*
import functools

import tensorflow as tf
from dotmap import DotMap

from magenta.models.onsets_frames_transcription import constants
from sklearn.metrics import f1_score

from magenta.models.onsets_frames_transcription.accuracy_util import flatten_loss_wrapper, \
    flatten_accuracy_wrapper, \
    flatten_f1_wrapper

FLAGS = tf.compat.v1.app.flags.FLAGS

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal, VarianceScaling
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, \
    Dense, Dropout, \
    Input, MaxPooling2D, concatenate, Lambda, Multiply, Reshape, Bidirectional, LSTM, \
    LeakyReLU, TimeDistributed, Cropping2D, Layer, Flatten, Masking, ELU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy


def get_default_hparams():
    return {
        'timbre_learning_rate': 0.0006,
        'timbre_decay_steps': 10000,
        'timbre_decay_rate': 0.98,
        'timbre_clip_norm': 3.0,
        'timbre_l2_regularizer': 1e-5,
        'timbre_filter_frequency_sizes': [3, int(constants.BINS_PER_OCTAVE / 1)],  # [5, 80],
        'timbre_filter_temporal_sizes': [1, 3, 5],
        'timbre_num_filters': [128, 64, 32],
        'timbre_filters_pool_size': (int(64 / 4), int(constants.BINS_PER_OCTAVE / 6)),
        # (int(constants.BINS_PER_OCTAVE/2), 16),#(22, 32),
        'timbre_pool_size': (2, 2),
        'timbre_num_layers': 2,
        'timbre_dropout_drop_amts': [0.0, 0.25, 0.25],
        'timbre_rnn_dropout_drop_amt': 0.0,
        'timbre_fc_size': 512,
        'timbre_fc_dropout_keep_amt': 0.5,
        'timbre_input_shape': (None, constants.SPEC_BANDS, 1),  # (None, 229, 1),
        'timbre_num_classes': 11,
        'timbre_lstm_units': 128,
        'timbre_rnn_stack_size': 2,
        'timbre_leaky_alpha': 0.33,  # per han et al 2016
        'timbre_sharing_conv': True,
        'timbre_label_smoothing': 0.1,
        'timbre_class_weights': {
            0: 6000 / 68955,
            1: 6000 / 13830,
            2: 6000 / 9423,
            3: 6000 / 35423,
            4: 6000 / 54991,
            5: 6000 / 35066,
            6: 6000 / 36577,
            7: 6000 / 14866,
            8: 6000 / 20594,
            9: 6000 / 5501,
            10: 6000 / 10753,
        }
    }


# if we aren't coagulating the cropped mini-batches, then we use TimeDistributed
def time_distributed_wrapper(x, hparams):
    if hparams.timbre_coagulate_mini_batches:
        return x
    return TimeDistributed(x)


def filters_layer(hparams, channel_axis):
    if hparams.timbre_sharing_conv:
        # if we are sharing the filters then we do this before cropping
        # so adjust accordingly
        time_distributed_double_wrapper = lambda x, hparams=None: x
        # don't pool yet so we have more cropping accuracy
        pool_size = (1, 1)
    else:
        time_distributed_double_wrapper = time_distributed_wrapper
        pool_size = hparams.timbre_filters_pool_size

    def filters_layer_fn(inputs):
        parallel_layers = []

        bn_elu_fn = lambda x: time_distributed_double_wrapper(
            MaxPooling2D(pool_size=pool_size), hparams=hparams) \
            (ELU(hparams.timbre_leaky_alpha)
             (time_distributed_double_wrapper(BatchNormalization(axis=channel_axis, scale=False),
                                              hparams=hparams)
              (x)))
        conv_bn_elu_layer = lambda num_filters, conv_temporal_size, conv_freq_size: lambda \
                x: bn_elu_fn(
            time_distributed_double_wrapper(Conv2D(
                num_filters,
                [conv_temporal_size, conv_freq_size],
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(hparams.timbre_l2_regularizer),
                kernel_initializer=he_normal(),
                name='parallel_conv_{}_{}_{}'.format(num_filters, conv_temporal_size,
                                                     conv_freq_size)
            ), hparams=hparams)(x))

        bottleneck = Conv2D(32, (1, 1))(inputs)
        for f_i in hparams.timbre_filter_frequency_sizes:
            for i, t_i in enumerate(hparams.timbre_filter_temporal_sizes):
                parallel_layers.append(
                    conv_bn_elu_layer(hparams.timbre_num_filters[i], t_i, f_i)(bottleneck))

        K.print_tensor(parallel_layers[0], 'parallel')
        return concatenate(parallel_layers, axis=channel_axis)

    return filters_layer_fn


def lstm_layer(hparams,
               implementation=2):
    def lstm_layer_fn(inputs):
        lstm_stack = inputs
        for i in range(hparams.timbre_rnn_stack_size):
            lstm_stack = time_distributed_wrapper(LSTM(
                hparams.timbre_lstm_units,
                recurrent_activation='sigmoid',
                implementation=implementation,
                return_sequences=i < hparams.timbre_rnn_stack_size - 1,
                recurrent_dropout=hparams.timbre_rnn_dropout_drop_amt,
                kernel_initializer=VarianceScaling(2, distribution='uniform')
            ), hparams=hparams)(lstm_stack)
        return lstm_stack

    return lstm_layer_fn


def acoustic_model_layer(hparams, channel_axis):
    def acoustic_model_fn(inputs):
        # inputs should be of type keras.layers.Input
        outputs = inputs

        # batch norm, then elu activation, then maxpool
        bn_elu_fn = lambda x: MaxPooling2D(pool_size=hparams.timbre_pool_size,
                                           strides=hparams.timbre_pool_size) \
            (ELU(hparams.timbre_leaky_alpha)
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


def acoustic_dense_layer(hparams):
    def acoustic_dense_fn(inputs):
        outputs = inputs
        outputs = time_distributed_wrapper(Dropout(hparams.timbre_fc_dropout_keep_amt),
                                           hparams=hparams)(outputs)
        outputs = time_distributed_wrapper(Dense(hparams.timbre_fc_size,
                                                 kernel_initializer=he_normal(),
                                                 kernel_regularizer=l2(
                                                     hparams.timbre_l2_regularizer),
                                                 name='acoustic_dense'),
                                           hparams=hparams)(outputs)
        outputs = ELU(hparams.timbre_leaky_alpha)(outputs)
        return outputs

    return acoustic_dense_fn


def instrument_prediction_layer(hparams):
    def instrument_prediction_fn(inputs):
        # inputs should be of type keras.layers.*
        # outputs = Flatten()(inputs)
        outputs = inputs
        outputs = time_distributed_wrapper(Dropout(hparams.timbre_fc_dropout_keep_amt),
                                           hparams=hparams)(outputs)
        outputs = time_distributed_wrapper(Dense(hparams.timbre_num_classes,
                                                 activation='softmax',
                                                 name='timbre_prediction',
                                                 kernel_initializer=he_normal(),
                                                 kernel_regularizer=l2(
                                                     hparams.timbre_l2_regularizer)),
                                           hparams=hparams)(outputs)
        return outputs

    return instrument_prediction_fn


def high_pass_filter(input_list):
    spec = input_list[0]
    hp = input_list[1]

    seq_mask = K.cast(tf.math.logical_not(tf.sequence_mask(hp, constants.SPEC_BANDS)), tf.float32)
    seq_mask = K.expand_dims(seq_mask, 3)
    return Multiply()([spec, seq_mask])


def normalize_and_weigh(inputs, num_notes, pitches):
    gradient_pitch_mask = 1 + K.int_shape(inputs)[-2] - K.arange(
        K.int_shape(inputs)[-2])  # + K.int_shape(inputs)[-2]
    gradient_pitch_mask = gradient_pitch_mask / K.max(gradient_pitch_mask)
    gradient_pitch_mask = K.expand_dims(K.cast(gradient_pitch_mask, 'float64'), 0)
    gradient_pitch_mask = tf.repeat(gradient_pitch_mask, axis=0, repeats=num_notes)
    gradient_pitch_mask = gradient_pitch_mask + K.expand_dims(pitches / K.int_shape(inputs)[-2], -1)
    gradient_pitch_mask = tf.minimum((gradient_pitch_mask) ** 12, 1.0)
    gradient_pitch_mask = K.expand_dims(gradient_pitch_mask, 1)
    gradient_pitch_mask = K.expand_dims(gradient_pitch_mask, -1)
    gradient_product = Multiply()([inputs, gradient_pitch_mask])
    return gradient_product


# This increases dimensionality by duplicating our input spec image and getting differently
# cropped views of it (essentially, get a small view for each note being played in a piece)
# num_crops are the number of notes
# the cropping list shows the pitch, start, and end of the note
# we do a high pass masking
# so that we don't look at any frequency data below a note's pitch for classifying
def get_all_croppings(input_list, hparams):
    conv_output_list = tf.unstack(input_list[0], num=hparams.nsynth_batch_size)
    note_croppings_list = tf.unstack(input_list[1], num=hparams.nsynth_batch_size)
    num_notes_list = tf.unstack(input_list[2], num=hparams.nsynth_batch_size)

    all_outputs = []
    # unbatch / do different things for each batch (we kinda create mini-batches)
    for batch_idx in range(hparams.nsynth_batch_size):
        out = get_croppings_for_single_image(conv_output_list[batch_idx],
                                             note_croppings_list[batch_idx],
                                             num_notes_list[batch_idx],
                                             hparams=hparams,
                                             temporal_scale=1 / 4)
        all_outputs.append(out)

    if hparams.timbre_coagulate_mini_batches:
        return K.concatenate(all_outputs, axis=0)

    return tf.convert_to_tensor(all_outputs)


# conv output is the image we are generating crops of
# note croppings tells us the min pitch, start idx and end idx to crop
# num notes tells us how many crops to make
def get_croppings_for_single_image(conv_output, note_croppings,
                                   num_notes, hparams=None, temporal_scale=1.0):
    repeated_conv_output = tf.repeat(K.expand_dims(conv_output, axis=0),
                                     num_notes, axis=0)
    gathered_pitches = tf.gather(note_croppings, indices=0, axis=1) \
                       * K.int_shape(conv_output)[1] \
                       / constants.SPEC_BANDS
    pitch_mask = K.cast(tf.math.logical_not(tf.sequence_mask(
        K.expand_dims(
            K.cast(
                gathered_pitches, dtype='int32'
            ), 1), K.int_shape(conv_output)[1]
    )), tf.float32)
    end_mask = K.cast(tf.sequence_mask(
        K.expand_dims(
            K.cast(
                tf.gather(note_croppings, indices=2, axis=1)
                / hparams.timbre_hop_length
                / temporal_scale, dtype='int32'
            ), 1),
        maxlen=K.shape(conv_output)[0]
    ), tf.float32)
    start_mask = K.cast(tf.math.logical_not(tf.sequence_mask(
        K.expand_dims(
            K.cast(
                tf.gather(note_croppings, indices=1, axis=1)
                / hparams.timbre_hop_length
                / temporal_scale, dtype='int32'
            ), 1),
        maxlen=K.shape(conv_output)[0]  # K.int_shape(end_mask)[2]
    )), tf.float32)
    # constant time for the pitch mask
    pitch_mask = K.expand_dims(pitch_mask, 3)
    # reorder to mask temporally
    end_mask = K.permute_dimensions(end_mask, (0, 2, 1))
    # constant pitch for right mask
    end_mask = K.expand_dims(end_mask, 2)
    # reorder to mask temporally
    start_mask = K.permute_dimensions(start_mask, (0, 2, 1))
    # constant pitch for left mask
    start_mask = K.expand_dims(start_mask, 2)
    mask = Multiply()([repeated_conv_output, pitch_mask])
    mask = Multiply()([mask, pitch_mask])
    mask = Multiply()([mask, start_mask])
    mask = Multiply()([mask, end_mask])

    mask = normalize_and_weigh(mask, num_notes, gathered_pitches)
    # Tell downstream layers to skip timesteps that are fully masked out
    return Masking(mask_value=0.0)(mask)


def timbre_prediction_model(hparams=None):
    if hparams is None:
        hparams = DotMap(get_default_hparams())

    input_shape = (
        hparams.timbre_input_shape[0], hparams.timbre_input_shape[1],
        hparams.timbre_input_shape[2],)
    channel_axis = -1

    inputs = Input(shape=input_shape, batch_size=hparams.nsynth_batch_size,
                   name='spec')

    # batched dimensions for cropping like:
    # ((top_crop, bottom_crop), (left_crop, right_crop))
    # with a high pass, top_crop will always be 0, bottom crop is relative to pitch
    note_croppings = Input(shape=(None, 3), batch_size=hparams.nsynth_batch_size,
                           name='note_croppings', dtype='int64')

    num_notes = Input(shape=(1,), batch_size=hparams.nsynth_batch_size,
                      name='num_notes', dtype='int64')

    # acoustic_outputs shape: (None, None, 57, 128)
    # aka: (batch_size, length, freq_range, num_channels)
    acoustic_outputs = acoustic_model_layer(hparams, channel_axis)(inputs)
    K.print_tensor(acoustic_outputs.shape, 'acoustic_outputs')

    if hparams.timbre_sharing_conv:
        # filter_outputs shape: (None, None, 57, 448)
        # aka: (batch_size, length, freq_range, num_channels)
        filter_outputs = filters_layer(hparams, channel_axis)(acoustic_outputs)
    else:
        # if we aren't sharing the large filters, then just pass the simple conv output
        filter_outputs = acoustic_outputs

    K.print_tensor(filter_outputs.shape, 'filter_outputs')

    # cropped_outputs shape: (batch_size, None, None, 57, 128)
    # aka: (batch_size, num_notes, length, freq_range, num_channels)
    cropped_outputs = Lambda(
        functools.partial(get_all_croppings, hparams=hparams))(
        [filter_outputs, note_croppings, num_notes])
    K.print_tensor(cropped_outputs.shape, 'cropped_outputs')

    cropped_outputs = time_distributed_wrapper(BatchNormalization(axis=-1, scale=False), hparams)(
        cropped_outputs)

    if hparams.timbre_sharing_conv:
        # pooled_outputs shape: (None, None, None, 19, 448)
        # aka: (batch_size, num_notes, length, freq_range, num_channels)
        pooled_outputs = time_distributed_wrapper(
            MaxPooling2D(pool_size=hparams.timbre_filters_pool_size),
            hparams=hparams)(cropped_outputs)
    else:
        pooled_outputs = filters_layer(hparams, channel_axis)(cropped_outputs)

    # We now need to use TimeDistributed because we have 5 dimensions, and want to operate on the
    # last 3 independently (time, frequency, and number of channels/filters)

    K.print_tensor(pooled_outputs.shape, 'pooled_outputs')

    # flatten while preserving batch and time dimensions
    # shape: (None, None, None, 1792)
    # aka: (batch_size, num_notes, length, vec_size)
    flattened_outputs = time_distributed_wrapper(Reshape(
        (-1, K.int_shape(pooled_outputs)[-2] * K.int_shape(pooled_outputs)[-1])), hparams=hparams)(
        pooled_outputs)
    K.print_tensor(flattened_outputs.shape, 'flattened_outputs')

    dense_outputs = acoustic_dense_layer(hparams)(flattened_outputs)

    K.print_tensor(dense_outputs.shape, 'dense_outputs')

    # shape: (None, None, 512)
    # aka: (batch_size, num_notes, lstm_units)
    lstm_outputs = lstm_layer(hparams=hparams)(dense_outputs)
    K.print_tensor(lstm_outputs.shape, 'lstm_outputs')

    # shape: (None, None, 11)
    # aka: (batch_size, num_notes, num_classes)
    instrument_family_probs = instrument_prediction_layer(hparams)(lstm_outputs)
    K.print_tensor(instrument_family_probs.shape, 'instrument_family_probs')

    instrument_family_probs = Lambda(lambda x: x, name='family_probs')(instrument_family_probs)

    losses = {'family_probs': flatten_loss_wrapper(hparams)}

    accuracies = {'family_probs': [flatten_accuracy_wrapper(hparams)]}

    return Model(inputs=[inputs, note_croppings, num_notes],
                 outputs=instrument_family_probs), losses, accuracies
