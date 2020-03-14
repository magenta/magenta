from __future__ import absolute_import, division, print_function

# if not using plaidml, use tensorflow.keras.* instead of keras.*
# if using plaidml, use keras.*
import functools
import math

import tensorflow as tf
from dotmap import DotMap

from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription.accuracy_util import flatten_accuracy_wrapper, \
    flatten_loss_wrapper, WeightedCategoricalCrossentropy
from magenta.models.onsets_frames_transcription.layer_util import conv_bn_elu_layer, \
    get_all_croppings, time_distributed_wrapper

FLAGS = tf.compat.v1.app.flags.FLAGS
if FLAGS.using_plaidml:
    import plaidml.keras

    plaidml.keras.install_backend()

    from keras import backend as K
    from keras.initializers import he_normal, VarianceScaling
    from keras.layers import BatchNormalization, Conv2D, \
        Dense, Dropout, \
        Input, concatenate, Lambda, Reshape, LSTM, \
        Flatten, ELU, GlobalMaxPooling2D, Bidirectional, Add
    from keras.models import Model
    from keras.regularizers import l2
else:
    from tensorflow.keras import backend as K
    from tensorflow.keras.initializers import he_normal, VarianceScaling
    from tensorflow.keras.layers import BatchNormalization, Conv2D, \
        Dense, Dropout, \
        Input, concatenate, Lambda, Reshape, LSTM, \
        Flatten, ELU, GlobalMaxPooling2D, Bidirectional, Add
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2


def get_default_hparams():
    return {
        'timbre_learning_rate': 0.0003,
        'timbre_decay_steps': 10000,
        'timbre_decay_rate': 1e-3,
        'timbre_clip_norm': 3.0,
        'timbre_l2_regularizer': 1e-5,
        'timbre_filter_frequency_sizes': [3, int(constants.BINS_PER_OCTAVE / 1)],  # [5, 80],
        'timbre_filter_temporal_sizes': [1, 3, 5],
        'timbre_num_filters': [128, 64, 32],
        'timbre_filters_pool_size': (int(64 / 4), int(constants.BINS_PER_OCTAVE / 6)),
        'timbre_vertical_filter': (1, 513),
        'timbre_vertical_num': 50,
        'timbre_horizontal_filter': (12, 1),
        'timbre_horizontal_num': 30,
        # (int(constants.BINS_PER_OCTAVE/2), 16),#(22, 32),
        'timbre_architecture': 'parallel',
        'timbre_pool_size': (4, 2),
        'timbre_num_layers': 2,
        'timbre_dropout_drop_amts': [0.0, 0.0, 0.0],
        'timbre_rnn_dropout_drop_amt': 0.0,
        'timbre_fc_size': 256,
        'timbre_penultimate_fc_size': 512,
        'timbre_fc_num_layers': 2,
        'timbre_fc_dropout_drop_amt': 0.0,
        'timbre_input_shape': (None, constants.TIMBRE_SPEC_BANDS, 1),  # (None, 229, 1),
        'timbre_num_classes': 11,
        'timbre_lstm_units': 256,
        'timbre_rnn_stack_size': 1,
        'timbre_leaky_alpha': 0.33,  # per han et al 2016 OR no negatives
        'timbre_sharing_conv': True,
        'timbre_extra_conv': False,
        'timbre_global_pool': 1,
        'timbre_label_smoothing': 0.0,
        'timbre_bottleneck_filter_num': 0,
        'timbre_gradient_exp': 16,  # 16 for cqt no-log
        'timbre_spec_epsilon': 1e-8,
        'timbre_class_weights_list': [
            16000 / 68955,
            16000 / 13830,
            16000 / 9423,
            16000 / 35423,
            16000 / 54991,
            16000 / 35066,
            16000 / 36577,
            16000 / 14866,
            16000 / 20594,
            16000 / 5501,
            16000 / 10753,
        ],
        'timbre_class_weights': {
            0: 16000 / 68955,
            1: 16000 / 13830,
            2: 16000 / 9423,
            3: 16000 / 35423,
            4: 16000 / 54991,
            5: 16000 / 35066,
            6: 16000 / 36577,
            7: 16000 / 14866,
            8: 16000 / 20594,
            9: 16000 / 5501,
            10: 16000 / 10753,
        }
    }


def vertical_layer(hparams):
    def vertical_layer_fn(inputs):
        return conv_bn_elu_layer(hparams.timbre_vertical_num, *hparams.timbre_vertical_filter,
                                 hparams.timbre_pool_size, True, hparams)(inputs)

    return vertical_layer_fn


def horizontal_layer(hparams):
    def horizontal_layer_fn(inputs):
        return conv_bn_elu_layer(hparams.timbre_horizontal_num, *hparams.timbre_horizontal_filter,
                                 hparams.timbre_pool_size, True, hparams)(inputs)

    return horizontal_layer_fn


def filters_layer(hparams):
    if hparams.timbre_sharing_conv:
        # if we are sharing the filters then we do this before cropping so adjust accordingly
        # don't pool yet so we have more cropping accuracy
        pool_size = (1, 1)
    else:
        pool_size = hparams.timbre_filters_pool_size

    def filters_layer_fn(inputs):
        parallel_layers = []
        for f_i in hparams.timbre_filter_frequency_sizes:
            for i, t_i in enumerate(hparams.timbre_filter_temporal_sizes):
                parallel_layers.append(
                    conv_bn_elu_layer(hparams.timbre_num_filters[i], t_i, f_i, pool_size, True,
                                      hparams)(inputs))

        # K.print_tensor(parallel_layers[0], 'parallel')
        return concatenate(parallel_layers, axis=-1)

    return filters_layer_fn


def lstm_layer(hparams,
               implementation=2):
    def lstm_layer_fn(inputs):
        lstm_stack = inputs
        for i in range(hparams.timbre_rnn_stack_size):
            lstm_stack = time_distributed_wrapper(Bidirectional(LSTM(
                hparams.timbre_lstm_units,
                recurrent_activation='sigmoid',
                implementation=implementation,
                return_sequences=i < hparams.timbre_rnn_stack_size - 1,
                recurrent_dropout=hparams.timbre_rnn_dropout_drop_amt,
                kernel_initializer=VarianceScaling(2, distribution='uniform')
            )), hparams=hparams)(lstm_stack)
        return lstm_stack

    return lstm_layer_fn


def acoustic_model_layer(hparams, pre_crop=True):
    if pre_crop:
        num_filters = 128
        num_layers = hparams.timbre_num_layers
    else:
        num_filters = 512
        num_layers = 1

    def acoustic_model_fn(inputs):
        # inputs should be of type keras.layers.Input
        outputs = inputs

        for i in range(num_layers):
            outputs = Dropout(hparams.timbre_fc_dropout_drop_amt)(outputs)
            outputs = conv_bn_elu_layer(num_filters, 3, 3, hparams.timbre_pool_size, pre_crop,
                                        hparams)(outputs)

        return outputs

    return acoustic_model_fn


def acoustic_dense_layer(hparams):
    # batch norm, then elu activation, then maxpool
    bn_elu_fn = lambda x: ELU(hparams.timbre_leaky_alpha)(
        time_distributed_wrapper(
            BatchNormalization(axis=-1, scale=False), hparams=hparams)
        (x))

    def acoustic_dense_fn(inputs):
        outputs = inputs
        for i in range(hparams.timbre_fc_num_layers):
            outputs = time_distributed_wrapper(Dropout(hparams.timbre_fc_dropout_drop_amt),
                                               hparams=hparams)(outputs)
            outputs = time_distributed_wrapper(Dense(hparams.timbre_fc_size,
                                                     kernel_initializer=he_normal(),
                                                     kernel_regularizer=l2(
                                                         hparams.timbre_l2_regularizer),
                                                     bias_regularizer=l2(
                                                         hparams.timbre_l2_regularizer),
                                                     name=f'acoustic_dense_{i}'),
                                               hparams=hparams)(outputs)
            if i < hparams.timbre_fc_num_layers - 1:
                outputs = bn_elu_fn(outputs)
            else:
                outputs = ELU(hparams.timbre_leaky_alpha)(outputs)
        return outputs

    return acoustic_dense_fn


def instrument_prediction_layer(hparams):
    def instrument_prediction_fn(inputs):
        # inputs should be of type keras.layers.*
        # outputs = Flatten()(inputs)
        outputs = inputs
        outputs = time_distributed_wrapper(Dropout(hparams.timbre_fc_dropout_drop_amt),
                                           hparams=hparams)(outputs)
        outputs = time_distributed_wrapper(Dense(hparams.timbre_num_classes,
                                                 activation='softmax',
                                                 name='timbre_prediction',
                                                 kernel_initializer=he_normal(),
                                                 kernel_regularizer=l2(
                                                     hparams.timbre_l2_regularizer),
                                                 bias_regularizer=l2(
                                                     hparams.timbre_l2_regularizer),
                                                 ),
                                           hparams=hparams)(outputs)
        return outputs

    return instrument_prediction_fn


def timbre_prediction_model(hparams=None):
    if hparams is None:
        hparams = DotMap(get_default_hparams())

    input_shape = (
        hparams.timbre_input_shape[0], hparams.timbre_input_shape[1],
        hparams.timbre_input_shape[2],)

    spec = Input(shape=input_shape,
                 name='spec')

    # batched dimensions for cropping like:
    # ((top_crop, bottom_crop), (left_crop, right_crop))
    # with a high pass, top_crop will always be 0, bottom crop is relative to pitch
    note_croppings = Input(shape=(None, 3),
                           name='note_croppings', dtype='int64')

    num_notes = Input(shape=(1,),
                      name='num_notes', dtype='int64')

    spec_with_epsilon = spec + hparams.timbre_spec_epsilon

    if hparams.timbre_architecture == 'vh':
        vertical_outputs = vertical_layer(hparams)(spec_with_epsilon)
        filter_outputs = horizontal_layer(hparams)(vertical_outputs)
    else:
        # acoustic_outputs shape: (None, None, 57, 128)
        # aka: (batch_size, length, freq_range, num_channels)
        acoustic_outputs = acoustic_model_layer(hparams)(spec_with_epsilon)

        if hparams.timbre_sharing_conv:
            # filter_outputs shape: (None, None, 57, 448)
            # aka: (batch_size, length, freq_range, num_channels)
            filter_outputs = filters_layer(hparams)(acoustic_outputs)
        else:
            # if we aren't sharing the large filters, then just pass the simple conv output
            filter_outputs = acoustic_outputs

    # simplify to save memory
    if hparams.timbre_bottleneck_filter_num:
        filter_outputs = Conv2D(hparams.timbre_bottleneck_filter_num, (1, 1))(filter_outputs)

    # batch_size is excluded from this shape as it gets automatically inferred
    # ouput_shape with coagulation: (batch_size*num_notes, freq_range, num_filters)
    # output_shape without coagulation: (batch_size, num_notes, freq_range, num_filters)
    output_shape = (math.ceil(K.int_shape(filter_outputs)[2] / hparams.timbre_filters_pool_size[1]), K.int_shape(filter_outputs)[3]) \
        if hparams.timbre_coagulate_mini_batches \
        else (None, math.ceil(K.int_shape(filter_outputs)[2] / hparams.timbre_filters_pool_size[1]), K.int_shape(filter_outputs)[3])
    cropped_outputs = Lambda(
        functools.partial(get_all_croppings, hparams=hparams), dynamic=True,
        output_shape=output_shape)(
        [filter_outputs, note_croppings, num_notes])

    # cropped_outputs = time_distributed_wrapper(ConvLSTM2D(128, (3, 3), activation='elu', return_sequences=True),
    #                                            hparams=hparams)(K.expand_dims(cropped_outputs, -1)) #TODO maybe -2

    # cropped_outputs = time_distributed_wrapper(BatchNormalization(axis=-1, scale=False), hparams)(
    #     cropped_outputs)

    if hparams.timbre_sharing_conv:
        # pooled_outputs shape: (None, None, None, 19, 448)
        # aka: (batch_size, num_notes, length, freq_range, num_channels)
        pooled_outputs = cropped_outputs
    else:
        pooled_outputs = filters_layer(hparams)(cropped_outputs)

    # We now need to use TimeDistributed because we have 5 dimensions, and want to operate on the
    # last 3 independently (time, frequency, and number of channels/filters)

    if hparams.timbre_extra_conv:
        pooled_outputs = acoustic_model_layer(hparams, pre_crop=False)(pooled_outputs)

    # flatten while preserving batch and time dimensions
    # shape: (None, None, None, 1792)
    # aka: (batch_size, num_notes, length, vec_size)
    if hparams.timbre_global_pool:
        if hparams.timbre_global_pool == 2:
            pooled_outputs = time_distributed_wrapper(GlobalMaxPooling2D(),
                                                      hparams=hparams)(pooled_outputs)

        flattened_outputs = time_distributed_wrapper(Flatten(), hparams=hparams)(
            pooled_outputs)
        timeless_outputs = acoustic_dense_layer(hparams)(flattened_outputs)

        non_normalized_outputs = time_distributed_wrapper(
            Dense(hparams.timbre_penultimate_fc_size, kernel_initializer=VarianceScaling()),
            hparams)(
            timeless_outputs)
        non_normalized_outputs = ELU(hparams.timbre_leaky_alpha)(non_normalized_outputs)

    else:
        flattened_outputs = time_distributed_wrapper(Reshape(
            (-1, K.int_shape(pooled_outputs)[-2] * K.int_shape(pooled_outputs)[-1])),
            hparams=hparams)(pooled_outputs)

        dense_outputs = acoustic_dense_layer(hparams)(flattened_outputs)

        # shape: (None, None, 512)
        # aka: (batch_size, num_notes, lstm_units)
        timeless_outputs = lstm_layer(hparams=hparams)(dense_outputs)
        non_normalized_outputs = ELU(hparams.timbre_leaky_alpha)(timeless_outputs)

    # shape: (None, None, 11)
    # aka: (batch_size, num_notes, num_classes)
    instrument_family_probs = instrument_prediction_layer(hparams)(non_normalized_outputs)

    instrument_family_probs = Lambda(lambda x: x, name='family_probs')(instrument_family_probs)

    # weigh based on predicted value (expand on 0) instead?
    losses = {'family_probs': WeightedCategoricalCrossentropy(
        weights=tf.repeat(K.expand_dims(hparams.timbre_class_weights_list, -1),
                          len(hparams.timbre_class_weights_list), axis=-1))}

    accuracies = {'family_probs': [flatten_accuracy_wrapper(hparams)]}

    return Model(inputs=[spec, note_croppings, num_notes],
                 outputs=instrument_family_probs), losses, accuracies
