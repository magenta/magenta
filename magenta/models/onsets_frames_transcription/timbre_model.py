from __future__ import absolute_import, division, print_function

# if not using plaidml, use tensorflow.keras.* instead of keras.*
# if using plaidml, use keras.*
import tensorflow.compat.v1 as tf
import numpy as np
from dotmap import DotMap
from magenta.models.onsets_frames_transcription import constants
from sklearn.metrics import fbeta_score

FLAGS = tf.app.flags.FLAGS

if FLAGS.using_plaidml:
    import plaidml.keras

    plaidml.keras.install_backend()

    from keras import backend as K
    from keras.initializers import he_normal
    from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, \
    Input, MaxPooling2D, concatenate, Flatten, Lambda, Cropping2D, Multiply
    from keras.models import Model
    from keras.regularizers import l2
else:
    # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    from tensorflow.keras import backend as K
    from tensorflow.keras.initializers import he_normal
    from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, \
        Dense, Dropout, \
        Input, MaxPooling2D, concatenate, Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2



def get_default_hparams():
    return {
        'using_plaidml': True,
        'batch_size': 8,
        'epochs_per_save': 1,
        'learning_rate': 0.0006,
        'decay_steps': 10000,
        'decay_rate': 0.98,
        'clip_norm': 3.0,
        'transform_audio': False,
        'l2_regulizer': 1e-5,
        'filter_m_sizes': [5, 80],
        'filter_n_sizes': [1, 3, 5],
        'num_filters': [128, 64, 32],
        'filters_pool_size': (22, 32),
        'pool_size': (2, 2),
        'num_layers': 2,
        'dropout_keep_amts': [1.0, 0.75, 0.75],
        'fc_size': 256,
        'fc_dropout_keep_amt': 0.5,
        'input_shape': (None, constants.SPEC_BANDS, 1),  # (None, 229, 1),
        'model_id': None
    }


def filters_layer(hparams, channel_axis):
    def filters_layer_fn(inputs):
        parallel_layers = []

        bn_elu_fn = lambda x: MaxPooling2D(pool_size=hparams.filters_pool_size) \
            (Activation('elu')
             (BatchNormalization(axis=channel_axis, scale=False)
              (x)))
        conv_bn_elu_layer = lambda num_filters, conv_temporal_size, conv_freq_size: lambda \
                x: bn_elu_fn(
            Conv2D(
                num_filters,
                [conv_temporal_size, conv_freq_size],
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(hparams.l2_regulizer),
                kernel_initializer=he_normal()
            )(x))

        for m_i in hparams.filter_m_sizes:
            for i, n_i in enumerate(hparams.filter_n_sizes):
                parallel_layers.append(conv_bn_elu_layer(hparams.num_filters[i], m_i, n_i)(inputs))

        return concatenate(parallel_layers, channel_axis)

    return filters_layer_fn


def acoustic_model_layer(hparams, channel_axis, num_classes):
    def acoustic_model_fn(inputs):
        # inputs should be of type keras.layers.Input
        outputs = inputs

        # batch norm, then elu activation, then maxpool
        bn_elu_fn = lambda x: MaxPooling2D(pool_size=hparams.pool_size, strides=hparams.pool_size) \
            (Activation('elu')
             (BatchNormalization(axis=channel_axis, mode=0, scale=False)
              (x)))
        # conv2d, then above
        conv_bn_elu_layer = lambda num_filters, conv_temporal_size, conv_freq_size: lambda \
                x: bn_elu_fn(
            Conv2D(
                num_filters,
                [conv_temporal_size, conv_freq_size],
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(hparams.l2_regulizer),
                kernel_initializer=he_normal()
            )(x))

        outputs = filters_layer(hparams, channel_axis)(outputs)
        for i in range(hparams.num_layers):
            outputs = Dropout(0.25)(outputs)
            outputs = conv_bn_elu_layer(128, 3, 3)(outputs)

        return outputs

    return acoustic_model_fn


def instrument_prediction_layer(hparams, num_classes):
    def instrument_prediction_fn(inputs):
        # inputs should be of type keras.layers.*
        outputs = Flatten()(inputs)
        outputs = Dropout(0.5)(outputs)
        outputs = Dense(hparams.fc_size,
                        kernel_initializer=he_normal(),
                        kernel_regularizer=l2(hparams.l2_regulizer))(outputs)
        outputs = Activation('elu')(outputs)
        outputs = Dropout(0.5)(outputs)
        outputs = Dense(num_classes,
                        activation='softmax',
                        name='timbre_prediction',
                        kernel_initializer=he_normal(),
                        kernel_regularizer=l2(hparams.l2_regulizer))(outputs)
        return outputs

    return instrument_prediction_fn

def high_pass_filter(input_list):
    spec = input_list[0]
    hp = input_list[1]

    reset_list = []
    for batch_num in K.int_shape(hp)[0]:
        ones_list = np.ones((K.int_shape(spec)[1], constants.SPEC_BANDS - hp[batch_num]))
        zeroes_list = np.zeros((K.int_shape(spec)[1], hp[batch_num]))
        reset_list = np.append(reset_list, np.concatenate((zeroes_list, ones_list), axis=1))

    reset_tensor = K.constant(reset_list)
    return Multiply()([spec, reset_tensor])

def timbre_prediction_model(hparams=None, num_classes=2):
    if hparams is None:
        hparams = DotMap(get_default_hparams())

    if K.image_dim_ordering() == 'th':
        input_shape = (hparams.input_shape[2], hparams.input_shape[0], hparams.input_shape[1],)
        channel_axis = 1
    else:
        input_shape = (hparams.input_shape[0], hparams.input_shape[1], hparams.input_shape[2],)
        channel_axis = 3
    input = Input(shape=input_shape,
                  name='spec')
    lowest_band = Input(shape=(1,), name='high_pass')

    filtered_spec = Lambda(high_pass_filter)([input, lowest_band])

    timbre_outputs = acoustic_model_layer(hparams, channel_axis, num_classes)(filtered_spec)
    timbre_probs = instrument_prediction_layer(hparams, num_classes)(timbre_outputs)

    losses = {
        'timbre_prediction': 'categorical_crossentropy'
    }

    accuracies = {
        'timbre_prediction': ['accuracy', fbeta_score]
    }

    return Model(inputs=[input, lowest_band], outputs=[timbre_probs]), losses, accuracies
