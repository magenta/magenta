from __future__ import absolute_import, division, print_function

# if not using plaidml, use tensorflow.keras.* instead of keras.*
# if using plaidml, use keras.*
import functools
import math

from dotmap import DotMap
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, Conv1D, Conv2D, ConvLSTM2D, Dense, Dropout, ELU, \
    Flatten, GlobalMaxPooling1D, Input, LSTM, Lambda, MaxPooling1D, Reshape, SpatialDropout2D, \
    TimeDistributed, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from magenta.models.polyamp import constants
from magenta.models.polyamp.accuracy_util import flatten_accuracy_wrapper, flatten_f1_wrapper
from magenta.models.polyamp.layer_util import conv_bn_elu_layer, \
    get_all_croppings, time_distributed_wrapper
# \[0\.[2-7][0-9\.,\ ]+\]$\n.+$\n\[0\.[2-7]
from magenta.models.polyamp.loss_util import timbre_loss_wrapper


def get_default_hparams():
    return {
        'timbre_learning_rate': 4e-4,
        'timbre_decay_steps': 10000,
        'timbre_decay_rate': 1e-2,
        'timbre_clip_norm': 2.0,
        'timbre_l2_regularizer': 1e-7,
        'timbre_filter_frequency_sizes': [3, 7],  # [5, 80],
        'timbre_filter_temporal_sizes': [1, 3, 5],
        'timbre_num_filters': [96, 64, 32],
        'timbre_filters_pool_size': (int(64 / 4), int(constants.BINS_PER_OCTAVE / 14)),
        'timbre_vertical_filter': (1, 513),
        'timbre_vertical_num': 50,
        'timbre_horizontal_filter': (12, 1),
        'timbre_horizontal_num': 30,
        # (int(constants.BINS_PER_OCTAVE/2), 16),#(22, 32),
        'timbre_architecture': 'parallel',
        'timbre_pool_size': [(3, 2), (3, 2)],
        'timbre_conv_num_layers': 2,
        'timbre_dropout_drop_amts': [0.1, 0.1, 0.1],
        'timbre_rnn_dropout_drop_amt': [0.3, 0.5],
        'timbre_fc_size': 512,
        'timbre_penultimate_fc_size': 256,
        'timbre_fc_num_layers': 0,
        'timbre_fc_dropout_drop_amt': 0.0,
        'timbre_conv_drop_amt': 0.1,
        'timbre_final_dropout_amt': 0.0,
        'timbre_local_conv_size': 7,
        'timbre_local_conv_strides': 1,  # reduce memory requirement
        'timbre_local_conv_num_filters': 256,
        'timbre_input_shape': (None, constants.TIMBRE_SPEC_BANDS, 1),  # (None, 229, 1),
        'timbre_num_classes': constants.NUM_INSTRUMENT_FAMILIES + 1,  # include other and drums
        'timbre_lstm_units': 192,
        'timbre_rnn_stack_size': 0,
        'timbre_leaky_alpha': 0.33,
        'timbre_extra_conv': False,
        'timbre_penultimate_activation': 'elu',
        'timbre_final_activation': 'sigmoid',
        'timbre_spatial_dropout': True,
        'timbre_global_pool': 1,
        'timbre_label_smoothing': 0.0,
        'timbre_bottleneck_filter_num': 0,
        'timbre_gradient_exp': 12,  # 16 for cqt no-log
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
        },
        'weight_correct_multiplier': 0.08,  # DON'T trend towards the classes with most support
    }


def print_fn(x):
    """Dynamic print and pass the tensor"""
    # print(x)
    return x


def acoustic_model_layer(hparams):
    num_filters = 128

    def acoustic_model_fn(inputs):
        # inputs should be of type keras.layers.Input
        outputs = inputs

        for i in range(hparams.timbre_conv_num_layers):
            outputs = conv_bn_elu_layer(num_filters, 3, 3, hparams.timbre_pool_size[i],
                                        time_distributed_bypass=True,
                                        hparams=hparams,
                                        activation_fn=ELU(hparams.timbre_leaky_alpha))(outputs)
            if hparams.timbre_spatial_dropout:
                outputs = SpatialDropout2D(hparams.timbre_dropout_drop_amts[i])(outputs)
            else:
                outputs = Dropout(hparams.timbre_dropout_drop_amts[i])(outputs)

        return outputs

    return acoustic_model_fn


def parallel_filters_layer(hparams):
    # don't pool yet so we have more cropping accuracy
    pool_size = (1, 1)

    def filters_layer_fn(inputs):
        parallel_layers = []
        for f_i in hparams.timbre_filter_frequency_sizes:
            for i, t_i in enumerate(hparams.timbre_filter_temporal_sizes):
                outputs = conv_bn_elu_layer(hparams.timbre_num_filters[i], t_i, f_i, pool_size,
                                            True,
                                            hparams,
                                            activation_fn=ELU(hparams.timbre_leaky_alpha))(inputs)
                if hparams.timbre_spatial_dropout:
                    outputs = SpatialDropout2D(hparams.timbre_conv_drop_amt)(outputs)
                else:
                    outputs = Dropout(hparams.timbre_conv_drop_amt)(outputs)
                parallel_layers.append(outputs)
        return concatenate(parallel_layers, axis=-1)

    return filters_layer_fn


def lstm_layer(hparams,
               implementation=2):
    def lstm_layer_fn(inputs):
        lstm_stack = inputs
        for i in range(hparams.timbre_rnn_stack_size):
            lstm_stack = TimeDistributed(Bidirectional(LSTM(
                hparams.timbre_lstm_units,
                implementation=implementation,
                return_sequences=True,
                dropout=hparams.timbre_rnn_dropout_drop_amt[0],
                recurrent_dropout=hparams.timbre_rnn_dropout_drop_amt[1],
                kernel_initializer='he_uniform')))(lstm_stack)
        return lstm_stack

    return lstm_layer_fn


def local_conv_layer(hparams):
    size = hparams.timbre_local_conv_size
    strides = hparams.timbre_local_conv_strides

    def local_conv_fn(inputs):
        outputs = ELU(hparams.timbre_leaky_alpha)(
            time_distributed_wrapper(Conv1D(
                hparams.timbre_local_conv_num_filters,
                size,
                strides,
                padding='same',
                use_bias=False,
            ), hparams,
                name=f'roi_conv1d_{size}_{strides}')(inputs))

        outputs = time_distributed_wrapper(GlobalMaxPooling1D(), hparams, name='global_max_pitch')(
            outputs)

        # outputs = time_distributed_wrapper(Dropout(hparams.timbre_conv_drop_amt),
        #                                    hparams,
        #                                    name='conv1d_dropout')(outputs)
        return outputs

    return local_conv_fn


def acoustic_dense_layer(hparams):
    def acoustic_dense_fn(inputs):
        outputs = inputs
        for i in range(hparams.timbre_fc_num_layers):
            outputs = time_distributed_wrapper(Dense(hparams.timbre_fc_size,
                                                     kernel_initializer='he_uniform',
                                                     bias_regularizer=l2(1e-1),
                                                     use_bias=False,
                                                     activation='sigmoid',
                                                     name=f'acoustic_dense_{i}'),
                                               hparams=hparams)(outputs)
            # don't do batch normalization because our samples are no longer independent
            # outputs = ELU(hparams.timbre_leaky_alpha)(outputs)
            outputs = Dropout(hparams.timbre_fc_dropout_drop_amt)(outputs)

        penultimate_outputs = time_distributed_wrapper(
            Dense(hparams.timbre_penultimate_fc_size,
                  use_bias=True,
                  # bias so that low bass notes can be predicted
                  bias_regularizer=l2(hparams.timbre_l2_regularizer),
                  activation=hparams.timbre_penultimate_activation,
                  kernel_initializer='he_uniform'),
            hparams, name=f'penultimate_dense_{hparams.timbre_local_conv_num_filters}')(outputs)
        # penultimate_outputs = Dropout(hparams.timbre_final_dropout_amt)(penultimate_outputs)
        return penultimate_outputs

    return acoustic_dense_fn


def instrument_prediction_layer(hparams):
    def instrument_prediction_fn(inputs):
        # inputs should be of type keras.layers.*
        # outputs = Flatten()(inputs)
        outputs = inputs
        outputs = time_distributed_wrapper(Dense(hparams.timbre_num_classes,
                                                 activation=hparams.timbre_final_activation,
                                                 use_bias=False,
                                                 kernel_initializer='he_uniform'),
                                           hparams=hparams,
                                           name='timbre_prediction', )(outputs)
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

    instrument_family_probs = get_timbre_output_layer(hparams)([spec, note_croppings])

    if hparams.timbre_final_activation == 'sigmoid':
        # This is the only supported option.
        losses = {'family_probs': timbre_loss_wrapper(hparams=hparams, recall_weighing=4.)}

    elif hparams.timbre_final_activation == 'softmax':
        # TODO use multi-class labelling loss function.
        losses = {'family_probs': timbre_loss_wrapper(hparams=hparams, recall_weighing=4.)}

    else:
        # TODO use logit loss function.
        losses = {'family_probs': timbre_loss_wrapper(hparams=hparams, recall_weighing=4.)}

    accuracies = {'family_probs': [flatten_accuracy_wrapper(),
                                   lambda *x: flatten_f1_wrapper()(*x)['f1_score']]}

    return Model(inputs=[spec, note_croppings],
                 outputs=instrument_family_probs), losses, accuracies


def get_timbre_output_layer(hparams):
    def get_timbre_output_fn(input_list):
        spec, note_croppings = input_list
        spec_with_epsilon = Lambda(lambda x: x + hparams.timbre_spec_epsilon)(spec)
        # acoustic_outputs shape: (None, None, 57, 128)
        # aka: (batch_size, length, freq_range, num_channels)
        acoustic_outputs = acoustic_model_layer(hparams)(spec_with_epsilon)
        # filter_outputs shape: (None, None, 57, 448)
        # aka: (batch_size, length, freq_range, num_channels)
        filter_outputs = parallel_filters_layer(hparams)(acoustic_outputs)
        # simplify to save memory
        if hparams.timbre_bottleneck_filter_num:
            filter_outputs = Conv2D(hparams.timbre_bottleneck_filter_num, (1, 1),
                                    activation='relu')(
                filter_outputs)
        if hparams.timbre_rnn_stack_size > 0:
            # run time distributed lstm distributing over pitch
            # lstm_input = Permute((2, 1, 3))(filter_outputs)
            lstm_input = Lambda(lambda x: K.expand_dims(x, -2))(filter_outputs)
            lstm_outputs = Bidirectional(ConvLSTM2D(
                hparams.timbre_lstm_units,
                kernel_size=(5, 1),
                padding='same',
                return_sequences=True,
                dropout=hparams.timbre_rnn_dropout_drop_amt[0],
                recurrent_dropout=hparams.timbre_rnn_dropout_drop_amt[1],
                kernel_initializer='he_uniform'))(lstm_input)
            # Reshape "does not include batch axis"
            reshaped_outputs = Reshape((-1,
                                        K.int_shape(lstm_outputs)[2] * K.int_shape(lstm_outputs)[3],
                                        K.int_shape(lstm_outputs)[4]))(lstm_outputs)
        else:
            reshaped_outputs = filter_outputs
        # batch_size is excluded from this shape as it gets automatically inferred
        # ouput_shape with coagulation: (batch_size*num_notes, freq_range, num_filters)
        # output_shape without coagulation: (batch_size, num_notes, freq_range, num_filters)
        output_shape = (
            None,
            math.ceil(K.int_shape(reshaped_outputs)[2]),
            K.int_shape(reshaped_outputs)[3]
        )
        pooled_outputs = Lambda(
            functools.partial(get_all_croppings, hparams=hparams), dynamic=True,
            output_shape=output_shape)(
            [reshaped_outputs, note_croppings])

        # We now need to use TimeDistributed because we have 5 dimensions,
        # and want to operate on thelast 3 independently:
        # (time, frequency, and number of channels/filters).
        pooled_outputs = time_distributed_wrapper(
            MaxPooling1D(pool_size=(hparams.timbre_filters_pool_size[1],),
                         padding='same'),
            hparams,
            name='post_crop_pool')(pooled_outputs)

        if hparams.timbre_local_conv_num_filters:
            pooled_outputs = local_conv_layer(hparams)(pooled_outputs)
        # flatten while preserving batch and time dimensions
        flattened_outputs = time_distributed_wrapper(Flatten(), hparams=hparams, name='flatten')(
            pooled_outputs)
        penultimate_outputs = acoustic_dense_layer(hparams)(flattened_outputs)
        # shape: (None, None, 11)
        # aka: (batch_size, num_notes, num_classes)
        instrument_family_probs = instrument_prediction_layer(hparams)(penultimate_outputs)

        # remove padded predictions
        def remove_padded(input_list):
            probs, croppings = input_list
            end_indices = K.expand_dims(K.permute_dimensions(croppings, (2, 0, 1))[-1], -1)
            # remove negative end_indices
            return probs * K.cast_to_floatx(end_indices >= 0)

        instrument_family_probs = Lambda(remove_padded, name='family_probs')(
            [instrument_family_probs, note_croppings])
        return instrument_family_probs

    return get_timbre_output_fn
