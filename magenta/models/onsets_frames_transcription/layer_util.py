import functools
import math

import tensorflow as tf
from intervaltree.node import l2

from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription.nsynth_reader import get_cqt_index, get_mel_index

FLAGS = tf.compat.v1.app.flags.FLAGS
if FLAGS.using_plaidml:
    import plaidml.keras

    plaidml.keras.install_backend()
    from keras import backend as K
    from keras.layers import Multiply, Masking, BatchNormalization, Conv2D, ELU, \
        MaxPooling2D, TimeDistributed, \
        GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D
    from keras.regularizers import l2
    from keras.initializers import he_normal

else:
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Multiply, Masking, BatchNormalization, Conv2D, ELU, \
        MaxPooling2D, TimeDistributed, \
        GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.initializers import he_normal


# if we aren't coagulating the cropped mini-batches, then we use TimeDistributed
def time_distributed_wrapper(x, hparams):
    if hparams.timbre_coagulate_mini_batches:
        return x
    return TimeDistributed(x)


# if we are bypassing this, then return a function that returns itself
# otherwise maybe wrap x in a TimeDistributed wrapper
def get_time_distributed_wrapper(bypass=False):
    if bypass:
        return lambda x, hparams=None: x
    return time_distributed_wrapper


# batch norm, then elu activation, then maxpool
def bn_elu_layer(hparams, pool_size, time_distributed_bypass):
    def bn_elu_fn(x):
        return get_time_distributed_wrapper(bypass=time_distributed_bypass)(
            MaxPooling2D(pool_size=pool_size), hparams=hparams)(
            ELU(hparams.timbre_leaky_alpha)(
                get_time_distributed_wrapper(bypass=time_distributed_bypass)(
                    BatchNormalization(scale=False), hparams=hparams)
                (x)))

    return bn_elu_fn


# conv2d, then above
def conv_bn_elu_layer(num_filters, conv_temporal_size, conv_freq_size, pool_size,
                      time_distributed_bypass, hparams):
    def conv_bn_elu_fn(x):
        return bn_elu_layer(hparams, pool_size, time_distributed_bypass)(
            get_time_distributed_wrapper(bypass=time_distributed_bypass)(
                Conv2D(
                    num_filters,
                    [conv_temporal_size, conv_freq_size],
                    padding='same',
                    use_bias=False,
                    kernel_regularizer=l2(hparams.timbre_l2_regularizer),
                    kernel_initializer=he_normal(),
                    # name = 'conv_{}_{}_{}'.format(num_filters, conv_temporal_size, conv_freq_size)
                ), hparams=hparams)(x))

    return conv_bn_elu_fn


def high_pass_filter(input_list):
    spec = input_list[0]
    hp = input_list[1]

    seq_mask = K.cast(tf.math.logical_not(tf.sequence_mask(hp, constants.TIMBRE_SPEC_BANDS)),
                      tf.float32)
    seq_mask = K.expand_dims(seq_mask, 3)
    return Multiply()([spec, seq_mask])


def normalize_and_weigh(inputs, num_notes, pitches, hparams):
    gradient_pitch_mask = 1 + K.int_shape(inputs)[-2] - K.arange(
        K.int_shape(inputs)[-2])  # + K.int_shape(inputs)[-2]
    gradient_pitch_mask = gradient_pitch_mask / K.max(gradient_pitch_mask)
    gradient_pitch_mask = K.expand_dims(K.cast_to_floatx(gradient_pitch_mask), 0)
    gradient_pitch_mask = tf.repeat(gradient_pitch_mask, axis=0, repeats=num_notes)
    gradient_pitch_mask = gradient_pitch_mask + K.expand_dims(pitches / K.int_shape(inputs)[-2], -1)
    exp = math.log(hparams.timbre_gradient_exp) if hparams.timbre_spec_log_amplitude \
        else hparams.timbre_gradient_exp
    gradient_pitch_mask = tf.minimum(gradient_pitch_mask ** exp, 1.0)
    # gradient_pitch_mask = K.expand_dims(gradient_pitch_mask, 1)
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
    conv_output_list = input_list[0]
    note_croppings_list = input_list[1]
    num_notes_list = tf.reshape(input_list[2], (-1,))

    all_outputs = []
    # unbatch / do different things for each batch (we kinda create mini-batches)
    for batch_idx in range(K.int_shape(conv_output_list)[0]):
        if not num_notes_list[batch_idx]:
            out = K.zeros(shape=(0, *K.int_shape(conv_output_list[batch_idx])[1:]))
        else:
            out = get_croppings_for_single_image(conv_output_list[batch_idx],
                                                 note_croppings_list[batch_idx],
                                                 num_notes_list[batch_idx],
                                                 hparams=hparams,
                                                 temporal_scale=max(1, hparams.timbre_pool_size[0]
                                                                    ** hparams.timbre_num_layers))

            if hparams.timbre_sharing_conv:
                out = MaxPooling1D(pool_size=(hparams.timbre_filters_pool_size[1],),
                                   padding='same')(out)
            # out = tf.reshape(out, (-1, *out.shape[2:]))

        all_outputs.append(out)

    if hparams.timbre_coagulate_mini_batches:
        return K.concatenate(all_outputs, axis=0)

    return tf.convert_to_tensor(all_outputs)


def get_croppings_for_single_image(conv_output, note_croppings,
                                   num_notes, hparams=None, temporal_scale=1.0):
    pitch_idx_fn = functools.partial(get_cqt_index
                                     if hparams.timbre_spec_type == 'cqt'
                                     else get_mel_index,
                                     hparams=hparams)
    pitch_to_spec_index = tf.map_fn(
        pitch_idx_fn,
        tf.gather(note_croppings, indices=0, axis=1))
    gathered_pitches = K.cast_to_floatx(pitch_to_spec_index) \
                       * K.int_shape(conv_output)[1] \
                       / constants.TIMBRE_SPEC_BANDS
    pitch_mask = K.expand_dims(
        K.cast(tf.math.logical_not(tf.sequence_mask(
            K.cast(
                gathered_pitches, dtype='int32'
            ), K.int_shape(conv_output)[1]
        )), tf.float32), -1)

    trimmed_list = []
    start_idx = K.cast(
        tf.gather(note_croppings, indices=1, axis=1)
        / hparams.timbre_hop_length
        / temporal_scale, dtype='int32'
    )
    end_idx = K.cast(
        tf.gather(note_croppings, indices=2, axis=1)
        / hparams.timbre_hop_length
        / temporal_scale, dtype='int32'
    )
    for i in range(num_notes):
        trimmed_spec = conv_output[min(start_idx[i], K.int_shape(conv_output)[0] - 1):max(end_idx[i], start_idx[i] + 1)]
        if hparams.timbre_global_pool:
            # GlobalAveragePooling1D supports masking
            trimmed_spec = GlobalAveragePooling1D()(
                K.permute_dimensions(trimmed_spec, (1, 0, 2)))
        trimmed_list.append(K.expand_dims(trimmed_spec, 0))

    broadcasted_spec = K.concatenate(trimmed_list, axis=0)

    # do the masking
    mask = Multiply()([broadcasted_spec, pitch_mask])

    # mask = K.expand_dims(mask, axis=1)

    mask = normalize_and_weigh(mask, num_notes, gathered_pitches, hparams)

    # mask = tf.reshape(mask, (-1, *mask.shape[2:]))

    # Tell downstream layers to skip timesteps that are fully masked out
    return mask


# conv output is the image we are generating crops of
# note croppings tells us the min pitch, start idx and end idx to crop
# num notes tells us how many crops to make
def get_croppings_for_single_image_leg(conv_output, note_croppings,
                                       num_notes, hparams=None, temporal_scale=1.0):
    broadcasted_spec = K.expand_dims(conv_output, axis=0)
    gathered_pitches = K.cast_to_floatx(tf.gather(note_croppings, indices=0, axis=1)) \
                       * K.int_shape(conv_output)[1] \
                       / constants.TIMBRE_SPEC_BANDS
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
        maxlen=K.int_shape(conv_output)[0]
    ), tf.float32)
    start_mask = K.cast(tf.math.logical_not(tf.sequence_mask(
        K.expand_dims(
            K.cast(
                tf.gather(note_croppings, indices=1, axis=1)
                / hparams.timbre_hop_length
                / temporal_scale, dtype='int32'
            ), 1),
        maxlen=K.int_shape(conv_output)[0]  # K.int_shape(end_mask)[2]
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

    # expand dims so we can multiply different batch sizes
    broadcasted_spec = K.expand_dims(broadcasted_spec, 0)
    pitch_mask = K.expand_dims(pitch_mask, 0)
    start_mask = K.expand_dims(start_mask, 0)
    end_mask = K.expand_dims(end_mask, 0)

    # do the masking
    mask = Multiply()([broadcasted_spec, pitch_mask])
    mask = Multiply()([mask, start_mask])
    mask = Multiply()([mask, end_mask])

    # remove the extra fake batch dimension
    mask = tf.reshape(mask, K.int_shape(mask)[1:])

    mask = normalize_and_weigh(mask, num_notes, gathered_pitches, hparams)

    # Tell downstream layers to skip timesteps that are fully masked out
    return Masking(mask_value=0.0)(mask)
