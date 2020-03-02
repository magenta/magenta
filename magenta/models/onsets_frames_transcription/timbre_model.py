from __future__ import absolute_import, division, print_function

# if not using plaidml, use tensorflow.keras.* instead of keras.*
# if using plaidml, use keras.*
import functools

import tensorflow as tf
from dotmap import DotMap

from magenta.models.onsets_frames_transcription import constants
from sklearn.metrics import f1_score

FLAGS = tf.compat.v1.app.flags.FLAGS

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal, VarianceScaling
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, \
    Dense, Dropout, \
    Input, MaxPooling2D, concatenate, Lambda, Multiply, Reshape, Bidirectional, LSTM, \
    LeakyReLU, TimeDistributed, Cropping2D, Layer, Flatten, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy

def get_default_hparams():
    return {
        'nsynth_batch_size': 4,
        'timbre_training_max_instruments': 4,
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
        'timbre_pool_size': (1, 2),
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
        'timbre_max_start_offset': 4000,
    }


def filters_layer(hparams, channel_axis):
    def filters_layer_fn(inputs):
        parallel_layers = []

        bn_elu_fn = lambda x: TimeDistributed(
            MaxPooling2D(pool_size=hparams.timbre_filters_pool_size)) \
            (LeakyReLU(hparams.timbre_leaky_alpha)
             (TimeDistributed(BatchNormalization(axis=channel_axis, scale=False))
              (x)))
        conv_bn_elu_layer = lambda num_filters, conv_temporal_size, conv_freq_size: lambda \
                x: bn_elu_fn(
            TimeDistributed(Conv2D(
                num_filters,
                [conv_temporal_size, conv_freq_size],
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(hparams.timbre_l2_regularizer),
                kernel_initializer=he_normal()
            ))(x))

        for m_i in hparams.timbre_filter_m_sizes:
            for i, n_i in enumerate(hparams.timbre_filter_n_sizes):
                parallel_layers.append(
                    conv_bn_elu_layer(hparams.timbre_num_filters[i], m_i, n_i)(inputs))

        # add 1 to the channel axis because we are now "time distributed"
        K.print_tensor(parallel_layers[0], 'parallel')
        return concatenate(parallel_layers, channel_axis + 1)

    return filters_layer_fn


def lstm_layer(num_units,
               stack_size=1,
               rnn_dropout_drop_amt=0,
               implementation=2):
    def lstm_layer_fn(inputs):
        lstm_stack = inputs
        for i in range(stack_size):
            lstm_stack = TimeDistributed(Bidirectional(LSTM(
                num_units,
                recurrent_activation='sigmoid',
                implementation=implementation,
                return_sequences=i < stack_size - 1,
                recurrent_dropout=rnn_dropout_drop_amt,
                kernel_initializer=VarianceScaling(2, distribution='uniform'))
            ))(lstm_stack)
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
        outputs = TimeDistributed(Dropout(0.5))(outputs)
        outputs = TimeDistributed(Dense(hparams.timbre_fc_size,
                                        kernel_initializer=he_normal(),
                                        kernel_regularizer=l2(hparams.timbre_l2_regularizer)))(
            outputs)
        outputs = LeakyReLU(hparams.timbre_leaky_alpha)(outputs)
        outputs = TimeDistributed(Dropout(0.5))(outputs)
        outputs = TimeDistributed(Dense(hparams.timbre_num_classes,
                                        activation='softmax',
                                        name='timbre_prediction',
                                        kernel_initializer=he_normal(),
                                        kernel_regularizer=l2(hparams.timbre_l2_regularizer)))(
            outputs)
        return outputs

    return instrument_prediction_fn


def high_pass_filter(input_list):
    spec = input_list[0]
    hp = input_list[1]

    seq_mask = K.cast(tf.math.logical_not(tf.sequence_mask(hp, constants.SPEC_BANDS)), tf.float32)
    seq_mask = K.expand_dims(seq_mask, 3)
    return Multiply()([spec, seq_mask])


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
                                             temporal_scale=1/4)
        all_outputs.append(out)

    converted = tf.convert_to_tensor(all_outputs)

    return converted


# conv output is the image we are generating crops of
# note croppings tells us the min pitch, start idx and end idx to crop
# num notes tells us how many crops to make
def get_croppings_for_single_image(conv_output, note_croppings,
                                   num_notes, hparams=None, temporal_scale=1.0):
    repeated_conv_output = tf.repeat(K.expand_dims(conv_output, axis=0),
                                     num_notes, axis=0)
    pitch_mask = K.cast(tf.math.logical_not(tf.sequence_mask(
        K.expand_dims(
            K.cast(
                tf.gather(note_croppings, indices=0, axis=1)
                * K.int_shape(conv_output)[1]
                / constants.SPEC_BANDS, dtype='int32'
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
    out = repeated_conv_output
    out = Multiply()([out, pitch_mask])
    out = Multiply()([out, end_mask])
    out = Multiply()([out, start_mask])
    # Tell downstream layers to skip timesteps that are fully masked out
    #out = Masking(mask_value=0.0)(out)
    return out


def timbre_prediction_model(hparams=None):
    if hparams is None:
        hparams = DotMap(get_default_hparams())

    input_shape = (
        hparams.timbre_input_shape[0], hparams.timbre_input_shape[1],
        hparams.timbre_input_shape[2],)
    channel_axis = 3

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

    # cropped_outputs shape: (batch_size, None, None, 57, 128)
    # aka: (batch_size, num_notes, length, freq_range, num_channels)
    cropped_outputs = Lambda(
        functools.partial(get_all_croppings, hparams=hparams))(
        [acoustic_outputs, note_croppings, num_notes])
    K.print_tensor(cropped_outputs.shape, 'cropped_outputs')

    # We now need to use TimeDistributed because we have 5 dimensions, and want to operate on the
    # last 3 independently (time, frequency, and number of channels/filters)

    # filter_outputs shape: (None, None, None, 4, 448)
    # aka: (batch_size, num_notes, length, freq_range, num_channels)
    filter_outputs = filters_layer(hparams, channel_axis)(
        cropped_outputs)
    K.print_tensor(filter_outputs.shape, 'filter_outputs')

    # flatten while preserving batch and time dimensions
    # shape: (None, None, None, 1792)
    # aka: (batch_size, num_notes, length, vec_size)
    timbre_outputs = TimeDistributed(Reshape(
        (-1, K.int_shape(filter_outputs)[3] * K.int_shape(filter_outputs)[4])))(
        filter_outputs)
    K.print_tensor(timbre_outputs.shape, 'timbre_outputs')

    if hparams.timbre_lstm_units:
        # shape: (None, None, 256)
        # aka: (batch_size, num_notes, lstm_units)
        timbre_outputs = lstm_layer(hparams.timbre_lstm_units,
                                    stack_size=hparams.timbre_rnn_stack_size)(timbre_outputs)

    # shape: (None, None, 11)
    # aka: (batch_size, num_notes, num_classes)
    instrument_family_probs = instrument_prediction_layer(hparams)(timbre_outputs)
    K.print_tensor(instrument_family_probs.shape, 'instrument_family_probs')

    instrument_family_probs = Lambda(lambda x: x, name='family_probs')(instrument_family_probs)

    def flatten_loss(y_true, y_pred):
        rebatched_pred = K.reshape(y_pred, (-1, y_pred.shape[2]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        rebatched_true = K.reshape(y_true, (-1, y_pred.shape[2]))
        return categorical_crossentropy(rebatched_true, rebatched_pred)

    def flatten_accuracy(y_true, y_pred):
        rebatched_pred = K.reshape(y_pred, (-1, y_pred.shape[2]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        rebatched_true = K.reshape(y_true, (-1, y_pred.shape[2]))
        return categorical_accuracy(rebatched_true, rebatched_pred)

    losses = {'family_probs':  flatten_loss}

    accuracies = {'family_probs': flatten_accuracy}

    return Model(inputs=[inputs, note_croppings, num_notes],
                 outputs=instrument_family_probs), losses, accuracies
