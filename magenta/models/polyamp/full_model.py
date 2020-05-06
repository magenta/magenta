import tensorflow as tf
from dotmap import DotMap
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from magenta.models.polyamp import constants, infer_util
from magenta.models.polyamp.accuracy_util import multi_track_present_accuracy_wrapper, multi_track_prf_wrapper, \
    single_track_present_accuracy_wrapper
from magenta.models.polyamp.loss_util import full_model_loss_wrapper
from magenta.models.polyamp.layer_util import NoteCroppingsToPianorolls
from magenta.models.polyamp.timbre_dataset_reader import NoteCropping


# \[0\.[2-7][0-9\.,\ ]+\]$\n.+$\n\[0\.[2-7]


def get_default_hparams():
    return {
        'full_learning_rate': 2e-4,
        'prediction_generosity': 1,
        'multiple_instruments_threshold': 0.5,
        'use_all_instruments': False,
        'midi_trainable': True,
        'family_recall_weight': [1.0, 1.4, 1.4, .225, .05, 1.2, 1., 1.3, .45, 1.0, 1.0, .7],
        'inv_frequency_weight': [.5, .4, .3, 1, 1, .3, .4, .5, .7, .3, .6, .3],
    }


def get_dynamic_length(batched_pianoroll):
    """
    Get the length of the pianoroll as it's dependent of the dynamic length of
    out input spectrogram
    :param batched_pianoroll: the batched pianoroll which we are determining the length of
    :return: the lengths of all pianorolls (repeated because they're padded to same length)
    """
    return K.expand_dims(
        tf.repeat(K.expand_dims(K.int_shape(batched_pianoroll)[1], axis=0),
                  K.int_shape(batched_pianoroll)[0])
    )


def print_fn(x):
    """Dynamic print and pass the tensor"""
    print(x)
    return x


class FullModel:
    def __init__(self, midi_model, timbre_model, hparams):
        if hparams is None:
            hparams = DotMap()
        self.hparams = hparams
        self.midi_model: Model = midi_model
        self.timbre_model: Model = timbre_model

    def fill_pianoroll_area(self, sorted_croppings, filling_fn):
        """
        Extrapolate predictions to nearby notes in case we missed feeding some notes to our
        timbre prediction.
        :param sorted_croppings: note croppings sorted either by pitch or start time
        :param filling_fn: the function that fills the pianoroll in a specific fashion
        :return: nothing
        """
        for *eager_cropping, i in sorted_croppings:
            cropping = NoteCropping(*eager_cropping)
            if cropping.end_idx < 0:
                # don't fill padded notes
                continue
            pitch = cropping.pitch - constants.MIN_MIDI_PITCH
            start_idx = K.cast(cropping.start_idx / self.hparams.spec_hop_length, 'int64')
            end_idx = K.cast(cropping.end_idx / self.hparams.spec_hop_length, 'int64')
            # guess that anything higher and to the right of this note has these probs
            # for each i, this area is partially overridden
            filling_fn(i, pitch, start_idx, end_idx)

    def sequence_to_note_croppings(self, sequence):
        """
        Converts a NoteSequence Proto to a list of note_croppings
        :param sequence: NoteSequence to convert
        :return: list of note_croppings generated from sequence
        """
        note_croppings = []
        for note in sequence.notes:
            note_croppings.append(NoteCropping(pitch=note.pitch,
                                               start_idx=note.start_time * self.hparams.sample_rate,
                                               end_idx=note.end_time * self.hparams.sample_rate))
        if len(note_croppings) == 0:
            note_croppings.append(NoteCropping(
                pitch=-1e+7,
                start_idx=-1e+7,
                end_idx=-1e+7
            ))
        return note_croppings

    def get_croppings(self, input_list):
        """
        Convert frame predictions into a sequence. Pad so all batches have same nof notes.
        :param input_list: frames, onsets, offsets
        :return: Tensor of padded cropping lists (padded with large negative numbers)
        """
        batched_frame_predictions, batched_onset_predictions, batched_offset_predictions = \
            input_list

        croppings_list = []
        for batch_idx in range(K.int_shape(batched_frame_predictions)[0]):
            frame_predictions = batched_frame_predictions[batch_idx]
            onset_predictions = batched_onset_predictions[batch_idx]
            offset_predictions = batched_offset_predictions[batch_idx]
            sequence = infer_util.predict_sequence(
                frame_predictions=frame_predictions,
                onset_predictions=onset_predictions,
                offset_predictions=offset_predictions,
                velocity_values=None,
                hparams=self.hparams, min_pitch=constants.MIN_MIDI_PITCH)
            croppings_list.append(self.sequence_to_note_croppings(sequence))

        padded = tf.keras.preprocessing.sequence.pad_sequences(croppings_list,
                                                               padding='post',
                                                               dtype='int64',
                                                               value=-1e+7)
        return tf.convert_to_tensor(padded)

    def get_model(self):
        spec_512 = Input(shape=(None, constants.SPEC_BANDS, 1), name='midi_spec')
        spec_256 = Input(shape=(None, constants.SPEC_BANDS, 1), name='timbre_spec')
        present_instruments = Input(shape=(self.hparams.timbre_num_classes,))

        self.midi_model.trainable = self.hparams.midi_trainable
        frame_probs, onset_probs, offset_probs = self.midi_model.call([spec_512])

        stop_gradient = Lambda(lambda x: K.stop_gradient(x))
        # decrease threshold to feed more notes into the timbre prediction
        # even if they don't make the final cut in accuracy_util.multi_track_accuracy_wrapper
        frame_predictions = stop_gradient(
            frame_probs > self.hparams.predict_frame_threshold / self.hparams.prediction_generosity)
        # generous onsets are used so that we can get more frame prediction data for the instruments
        # this will end up making our end-predicted notes much shorter though
        generous_onset_predictions = stop_gradient(onset_probs > (
                self.hparams.predict_onset_threshold))
        offset_predictions = stop_gradient(offset_probs > self.hparams.predict_offset_threshold)

        note_croppings = Lambda(self.get_croppings,
                                output_shape=(None, 3),
                                dynamic=True,
                                dtype='int64')(
            [frame_predictions, generous_onset_predictions, offset_predictions])

        timbre_probs = self.timbre_model.call([spec_256, note_croppings])
        # timbre_probs = get_timbre_output_layer(self.hparams)([spec_256, note_croppings])

        expand_dims = Lambda(lambda x_list: K.expand_dims(x_list[0], axis=x_list[1]))
        float_cast = Lambda(lambda x: K.cast_to_floatx(x))

        print_layer = Lambda(print_fn)

        pianoroll = Lambda(lambda x: tf.repeat(K.expand_dims(x),
                                               self.hparams.timbre_num_classes,
                                               -1),
                           output_shape=(None,
                                         constants.MIDI_PITCHES,
                                         self.hparams.timbre_num_classes),
                           dynamic=True)(frame_probs)

        timbre_pianoroll = NoteCroppingsToPianorolls(self.hparams,
                                                     dynamic=True, )(
            [stop_gradient(note_croppings), timbre_probs, stop_gradient(pianoroll)])

        expanded_present_instruments = float_cast(expand_dims([expand_dims([
            (present_instruments), -2]), -2]))

        present_pianoroll = (
            Multiply(name='apply_present')([timbre_pianoroll, expanded_present_instruments]))

        pianoroll_no_gradient = stop_gradient(present_pianoroll)

        # roll the pianoroll to get instrument predictions for offsets which is normally where
        # we stop the pianoroll fill
        # rolled_pianoroll = Lambda(lambda x: tf.roll(x, 1, axis=-3))(pianoroll_no_gradient)

        expanded_frames = expand_dims([frame_probs, -1])
        expanded_onsets = expand_dims([onset_probs, -1])
        expanded_offsets = expand_dims([offset_probs, -1])

        # Use the last channel for instrument-agnostic midi
        broadcasted_frames = Concatenate(name='multi_frames')(
            [present_pianoroll, expanded_frames])
        broadcasted_onsets = Concatenate(name='multi_onsets')(
            [present_pianoroll, expanded_onsets])
        broadcasted_offsets = Concatenate(name='multi_offsets')(
            [pianoroll_no_gradient, expanded_offsets])

        losses = {
            'multi_frames': full_model_loss_wrapper(self.hparams,
                                                    self.hparams.frames_true_weighing),
            'multi_onsets': full_model_loss_wrapper(self.hparams,
                                                    self.hparams.onsets_true_weighing),
            'multi_offsets': full_model_loss_wrapper(self.hparams,
                                                     self.hparams.offsets_true_weighing),
        }

        accuracies = {
            'multi_frames': [
                multi_track_present_accuracy_wrapper(
                    self.hparams.predict_frame_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold),
                single_track_present_accuracy_wrapper(
                    self.hparams.predict_frame_threshold),
                multi_track_prf_wrapper(
                    self.hparams.predict_frame_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold,
                    print_report=True,
                    hparams=self.hparams)
            ],
            'multi_onsets': [
                multi_track_present_accuracy_wrapper(
                    self.hparams.predict_onset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold),
                single_track_present_accuracy_wrapper(
                    self.hparams.predict_onset_threshold),
                multi_track_prf_wrapper(
                    self.hparams.predict_onset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold,
                    print_report=True,
                    hparams=self.hparams)
            ],
            'multi_offsets': [
                multi_track_present_accuracy_wrapper(
                    self.hparams.predict_offset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold),
                single_track_present_accuracy_wrapper(
                    self.hparams.predict_offset_threshold),
                multi_track_prf_wrapper(
                    self.hparams.predict_offset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold,
                    hparams=self.hparams)
            ]
        }

        return Model(inputs=[spec_512, spec_256, present_instruments],
                     outputs=[broadcasted_frames, broadcasted_onsets,
                              broadcasted_offsets]), losses, accuracies
