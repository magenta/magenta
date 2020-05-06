import functools
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K, layers
from tensorflow.keras.layers import Conv2D, ELU, GlobalAveragePooling1D, GlobalMaxPooling1D, \
    MaxPooling2D, Multiply, TimeDistributed

from magenta.models.polyamp import constants, infer_util
from magenta.models.polyamp.timbre_dataset_reader import NoteCropping, get_cqt_index, get_mel_index
from magenta.music import midi_io


# if we aren't coagulating the cropped mini-batches, then we use TimeDistributed
def time_distributed_wrapper(x, hparams, name=None):
    return TimeDistributed(x, name=name)


# if we are bypassing this, then return a function that returns itself
# otherwise maybe wrap x in a TimeDistributed wrapper
def get_time_distributed_wrapper(bypass=False):
    if bypass:
        return lambda x, hparams=None: x
    return time_distributed_wrapper


# batch norm, then elu activation, then maxpool
def bn_elu_layer(hparams, pool_size, time_distributed_bypass, activation_fn):
    def bn_elu_fn(x):
        return get_time_distributed_wrapper(bypass=time_distributed_bypass)(
            MaxPooling2D(pool_size=pool_size), hparams=hparams)(
            activation_fn(
                # get_time_distributed_wrapper(bypass=time_distributed_bypass)(
                #     BatchNormalization(scale=False), hparams=hparams)
                (x)))

    return bn_elu_fn


# conv2d, then above
def conv_bn_elu_layer(num_filters, conv_temporal_size, conv_freq_size, pool_size,
                      time_distributed_bypass, hparams, activation_fn=ELU()):
    def conv_bn_elu_fn(x):
        return bn_elu_layer(hparams, pool_size, time_distributed_bypass,
                            activation_fn=activation_fn)(
            get_time_distributed_wrapper(bypass=time_distributed_bypass)(
                Conv2D(
                    num_filters,
                    [conv_temporal_size, conv_freq_size],
                    padding='same',
                    use_bias=False,
                    # kernel_regularizer=l2(hparams.timbre_l2_regularizer),
                    kernel_initializer='he_uniform',
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
    gradient_product = inputs * gradient_pitch_mask
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
    # num_notes_list = tf.reshape(input_list[2], (-1,))

    all_outputs = []
    # unbatch / do different things for each batch (we kinda create mini-batches)
    for batch_idx in range(K.int_shape(conv_output_list)[0]):
        if K.int_shape(note_croppings_list)[1] == 0:
            out = np.zeros(shape=(1, K.int_shape(conv_output_list[batch_idx])[1],
                                  K.int_shape(conv_output_list[batch_idx])[-1]))
        else:
            out = get_croppings_for_single_image(conv_output_list[batch_idx],
                                                 note_croppings_list[batch_idx],
                                                 hparams=hparams,
                                                 temporal_scale=max(1,
                                                                    hparams.timbre_pool_size[0][0]
                                                                    * hparams.timbre_pool_size[1][
                                                                        0]))

        # out = tf.reshape(out, (-1, *out.shape[2:]))

        all_outputs.append(out)

    return tf.convert_to_tensor(all_outputs)


def get_croppings_for_single_image(conv_output, note_croppings,
                                   hparams=None, temporal_scale=1.0):
    num_notes = K.int_shape(note_croppings)[0]
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
        K.cast(tf.where(tf.sequence_mask(
            K.cast(
                gathered_pitches, dtype='int32'
            ), K.int_shape(conv_output)[1]
        ), 2e-3, 1), tf.float32), -1)

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
        if end_idx[i] < 0:
            # is a padded value note
            trimmed_list.append(
                np.zeros(shape=(1, K.int_shape(conv_output)[1], K.int_shape(conv_output)[2]),
                         dtype=K.floatx()))
        else:
            trimmed_spec = conv_output[
                           min(start_idx[i], K.int_shape(conv_output)[0] - 1):max(end_idx[i],
                                                                                  start_idx[i] + 1)]
            length = K.int_shape(trimmed_spec)[1]
            # GlobalAveragePooling1D supports masking
            avg_pool = GlobalAveragePooling1D()(
                K.permute_dimensions(trimmed_spec, (1, 0, 2)))
            max_pool = GlobalMaxPooling1D()(
                K.permute_dimensions(trimmed_spec, (1, 0, 2)))
            max_pool = K.max(trimmed_spec, 0)
            # pools = K.concatenate([avg_pool, max_pool], -1)
            # single_note_data = K.concatenate([pools,
            #                                   tf.repeat(
            #                                       K.cast_to_floatx(K.expand_dims([length], 0)),
            #                                       K.int_shape(pools)[0],
            #                                       axis=0)],
            #                                  axis=-1)
            trimmed_list.append(K.expand_dims(max_pool, 0))

    broadcasted_spec = K.concatenate(trimmed_list, axis=0)

    # do the masking
    # don't lose gradient completely so don't multiply by 0
    mask = broadcasted_spec * pitch_mask

    # mask = K.expand_dims(mask, axis=1)

    mask = normalize_and_weigh(mask, num_notes, gathered_pitches, hparams)

    # mask = tf.reshape(mask, (-1, *mask.shape[2:]))

    # Tell downstream layers to skip timesteps that are fully masked out
    return mask  # Masking(mask_value=0.0)(mask)


class NoteCroppingsToPianorolls(layers.Layer):
    def __init__(self, hparams, **kwargs):
        self.hparams = hparams
        super(NoteCroppingsToPianorolls, self).__init__(**kwargs)

    def call(self, input_list, **kwargs):
        """
        Convert note croppings and their corresponding timbre predictions to a pianoroll that
        we can multiply by the melodic midi predictions
        :param input_list: note_croppings, timbre_probs, pianoroll_length
        :return: a pianoroll with shape (batches, pianoroll_length, 88, timbre_num_classes + 1)
        """
        batched_note_croppings, batched_timbre_probs, batched_pianorolls = input_list

        pianoroll_list = []
        for batch_idx in range(K.int_shape(batched_note_croppings)[0]):
            note_croppings = batched_note_croppings[batch_idx]
            timbre_probs = batched_timbre_probs[batch_idx]

            pianorolls = K.zeros(
                shape=(K.int_shape(batched_pianorolls[batch_idx])[0],
                       constants.MIDI_PITCHES,
                       self.hparams.timbre_num_classes))
            ones = np.ones(
                shape=(K.int_shape(batched_pianorolls[batch_idx])[0],
                       constants.MIDI_PITCHES,
                       self.hparams.timbre_num_classes))

            for i, note_cropping in enumerate(note_croppings):
                cropping = NoteCropping(*note_cropping)
                pitch = cropping.pitch - constants.MIN_MIDI_PITCH
                if cropping.end_idx < 0:
                    # don't fill padded notes
                    continue
                start_idx = K.cast(cropping.start_idx / self.hparams.spec_hop_length, 'int64')
                end_idx = K.cast(cropping.end_idx / self.hparams.spec_hop_length, 'int64')

                # end_pitch_mask = tf.sequence_mask(pitch, constants.MIDI_PITCHES)
                #
                # pitch_mask = K.cast_to_floatx(tf.logical_and(
                #     end_pitch_mask, tf.math.logical_not(tf.roll(end_pitch_mask, -1, axis=-1))))
                pitch_mask = K.cast_to_floatx(tf.one_hot(pitch, constants.MIDI_PITCHES))
                # end_pitch_mask = K.cast(tf.sequence_mask(
                #     pitch + 1, constants.MIDI_PITCHES
                # ), tf.float32)
                # pitch_mask = start_pitch_mask * end_pitch_mask
                end_time_mask = K.cast(tf.sequence_mask(
                    end_idx,
                    maxlen=K.int_shape(batched_pianorolls[batch_idx])[0]
                ), tf.float32)
                start_time_mask = K.cast(tf.math.logical_not(tf.sequence_mask(
                    start_idx,
                    maxlen=K.int_shape(batched_pianorolls[batch_idx])[0]
                )), tf.float32)
                time_mask = start_time_mask * end_time_mask
                # constant time for the pitch mask
                pitch_mask = K.expand_dims(K.expand_dims(pitch_mask, 0))
                # constant pitch for the time mask
                time_mask = K.expand_dims(K.expand_dims(time_mask, 1))
                mask = ones * pitch_mask
                mask = mask * time_mask
                cropped_probs = mask * (timbre_probs[i])
                if K.learning_phase() == 1:
                    # for training
                    pianorolls = pianorolls + cropped_probs
                else:
                    # for testing, this is faster
                    # pianorolls = pianorolls + cropped_probs
                    pianorolls.assign_add(cropped_probs)

            frame_predictions = pianorolls > self.hparams.multiple_instruments_threshold
            sequence = infer_util.predict_multi_sequence(
                frame_predictions=frame_predictions,
                min_pitch=constants.MIN_MIDI_PITCH,
                hparams=self.hparams)
            midi_filename = f'./out/{batch_idx}-of-{K.int_shape(batched_note_croppings)[0]}.midi'
            midi_io.sequence_proto_to_midi_file(sequence, midi_filename)
            # make time the first dimension
            pianoroll_list.append(pianorolls)

        pianoroll_tensor = tf.convert_to_tensor(pianoroll_list)
        # use nearby probabilities
        return pianoroll_tensor
        # return K.pool2d(pianoroll_tensor,
        #                 pool_size=(7, 3),
        #                 strides=(1, 1),
        #                 padding='same',
        #                 pool_mode='max')

    def compute_output_shape(self, input_shape):

        return input_shape[2]
