import operator

import numpy as np
import tensorflow as tf
from dotmap import DotMap
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from magenta.models.onsets_frames_transcription import constants, data, infer_util
from magenta.models.onsets_frames_transcription.accuracy_util import flatten_accuracy_wrapper, \
    multi_track_prf_wrapper, flatten_loss_wrapper, multi_track_loss_wrapper, \
    multi_track_present_accuracy_wrapper, single_track_present_accuracy_wrapper
from magenta.models.onsets_frames_transcription.instrument_family_mappings import \
    family_to_midi_instrument
from magenta.models.onsets_frames_transcription.loss_util import log_loss_wrapper
from magenta.models.onsets_frames_transcription.timbre_dataset_reader import NoteCropping

# \[0\.[2-7][0-9\.,\ ]+\]$\n.+$\n\[0\.[2-7]
from magenta.music import midi_io


def get_default_hparams():
    return {
        'full_learning_rate': 5e-4,
        'prediction_generosity': 1,
        'multiple_instruments_threshold': 0.5,
        'use_all_instruments': False
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
        self.midi_model = midi_model
        self.timbre_model = timbre_model

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

    def note_croppings_to_pianorolls(self, input_list):
        """
        Convert note croppings and their corresponding timbre predictions to a pianoroll that
        we can multiply by the melodic midi predictions
        :param input_list: note_croppings, timbre_probs, pianoroll_length
        :return: a pianoroll with shape (batches, pianoroll_length, 88, timbre_num_classes + 1)
        """
        batched_note_croppings, batched_timbre_probs, batched_pianoroll_length = input_list
        pianoroll_list = []
        for batch_idx in range(K.int_shape(batched_note_croppings)[0]):
            note_croppings = batched_note_croppings[batch_idx]

            # add the index here so that we still know it after sorting
            note_croppings = K.concatenate([note_croppings,
                                            K.expand_dims(K.arange(K.int_shape(note_croppings)[0],
                                                                   dtype='int64'))])
            timbre_probs = batched_timbre_probs[batch_idx]

            # Make pitch the first dimension for easy manipulation
            # default to equal guess for the instruments
            pianorolls = np.ones(
                shape=(constants.MIDI_PITCHES,
                       batched_pianoroll_length[batch_idx][0],
                       self.hparams.timbre_num_classes)) / (1 + self.hparams.timbre_num_classes)
            # Guess that things without timbre predictions are the instrument within one octave
            fill_pitch_range = 12

            def negative_pitch_fill(i, pitch, start_idx, end_idx):
                """1. Fill the pitches below and until the end of the pianoroll"""
                pianorolls[:pitch, start_idx:] = timbre_probs[i]

            def full_fill(i, pitch, start_idx, end_idx):
                """2. Fill the upward octave and until the end of the pianoroll"""
                pianorolls[pitch:pitch + fill_pitch_range, start_idx:] = timbre_probs[i]

            def time_fill(i, pitch, start_idx, end_idx):
                """3. Fill the exact pitch to the end of the pianoroll"""
                pianorolls[pitch, start_idx:] = timbre_probs[i]

            def pitch_fill(i, pitch, start_idx, end_idx):
                """4. Fill the upward octave for the exact duration"""
                pianorolls[pitch:pitch + fill_pitch_range, start_idx:end_idx + 1] = timbre_probs[i]

            def exact_fill(i, pitch, start_idx, end_idx):
                """5. Fill exactly the note cropping that the probs is for"""
                pianorolls[pitch, start_idx:end_idx + 1] = timbre_probs[i]

            pitch_sorted = sorted(note_croppings, key=operator.itemgetter(0, 1))
            start_time_sorted = sorted(note_croppings, key=operator.itemgetter(1, 0))

            # pitch is more important than start time for what an instrument is
            # we also don't want to overwrite definitive predictions
            self.fill_pianoroll_area(reversed(pitch_sorted), negative_pitch_fill)
            self.fill_pianoroll_area(pitch_sorted, full_fill)
            self.fill_pianoroll_area(start_time_sorted, time_fill)
            self.fill_pianoroll_area(pitch_sorted, pitch_fill)
            self.fill_pianoroll_area(start_time_sorted, exact_fill)

            permuted_time_first = K.permute_dimensions(pianorolls, (1, 0, 2))

            # frame_predictions = tf.logical_or(
            #     tf.logical_and(
            #         tf.equal(permuted_time_first, K.expand_dims(K.max(permuted_time_first, -1))),
            #         permuted_time_first > (1 / self.hparams.timbre_num_classes)),
            #     permuted_time_first > self.hparams.predict_frame_threshold)

            frame_predictions = permuted_time_first > self.hparams.multiple_instruments_threshold
            sequence = infer_util.predict_multi_sequence(
                frame_predictions=frame_predictions,
                min_pitch=constants.MIN_MIDI_PITCH,
                hparams=self.hparams)
            midi_filename = f'./out/{batch_idx}-of-{K.int_shape(batched_note_croppings)[0]}.midi'
            midi_io.sequence_proto_to_midi_file(sequence, midi_filename)
            # make time the first dimension
            pianoroll_list.append(permuted_time_first)

        return tf.convert_to_tensor(pianoroll_list)

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
        return padded

    def separate_batches(self, input_list):
        # TODO implement
        raise NotImplementedError

    def get_model(self):
        spec_512 = Input(shape=(None, constants.SPEC_BANDS, 1), name='midi_spec')
        spec_256 = Input(shape=(None, constants.SPEC_BANDS, 1), name='timbre_spec')
        present_instruments = Input(shape=(self.hparams.timbre_num_classes,))

        frame_probs, onset_probs, offset_probs = self.midi_model.call([spec_512])

        stop_gradient = Lambda(lambda x: K.stop_gradient(x))
        # decrease threshold to feed more notes into the timbre prediction
        # even if they don't make the final cut in accuracy_util.multi_track_accuracy_wrapper
        frame_predictions = stop_gradient(frame_probs > self.hparams.predict_frame_threshold)
        # generous onsets are used so that we can get more frame prediction data for the instruments
        # this will end up making our end-predicted notes much shorter though
        generous_onset_predictions = stop_gradient(onset_probs > (
                self.hparams.predict_onset_threshold / self.hparams.prediction_generosity))
        offset_predictions = stop_gradient(offset_probs > self.hparams.predict_offset_threshold)

        note_croppings = Lambda(self.get_croppings,
                                output_shape=(None, 3),
                                dynamic=True,
                                dtype='int64')(
            [frame_predictions, generous_onset_predictions, offset_predictions])

        pianoroll_length = Lambda(get_dynamic_length,
                                  output_shape=(1,),
                                  dtype='int64',
                                  dynamic=True)(frame_predictions)

        timbre_probs = self.timbre_model.call([spec_256, note_croppings])

        if self.hparams.timbre_coagulate_mini_batches:
            # re-separate
            timbre_probs = Lambda(self.separate_batches,
                                  dynamic=True,
                                  output_shape=(None, self.hparams.timbre_num_classes))(
                [timbre_probs])
        expand_dims = Lambda(lambda x_list: K.expand_dims(x_list[0], axis=x_list[1]))
        float_cast = Lambda(lambda x: K.cast_to_floatx(x))

        print_layer = Lambda(print_fn)

        timbre_pianoroll = Lambda(self.note_croppings_to_pianorolls,
                                  dynamic=True,
                                  output_shape=(None,
                                                constants.MIDI_PITCHES,
                                                self.hparams.timbre_num_classes))(
            [note_croppings, timbre_probs, pianoroll_length])

        expanded_present_instruments = float_cast(expand_dims([expand_dims([
            print_layer(present_instruments), -2]), -2]))

        present_pianoroll = (
            Multiply(name='apply_present')([timbre_pianoroll, expanded_present_instruments]))

        pianoroll_no_gradient = stop_gradient(present_pianoroll)

        expanded_frames = expand_dims([frame_probs, -1])
        expanded_onsets = expand_dims([onset_probs, -1])
        expanded_offsets = expand_dims([offset_probs, -1])

        expanded_frames_no_gradient = stop_gradient(expanded_frames)
        expanded_onsets_no_gradient = stop_gradient(expanded_onsets)
        expanded_offsets_no_gradient = stop_gradient(expanded_offsets)

        # timbre loss for frames such as string which have very difficult onsets to predict
        broadcasted_frames = Multiply(name='multi_frames_timbre_only')(
            [present_pianoroll, expanded_frames])
        # timbre loss for instruments that have lots of onset support
        # Getting onsets correct is very good for our model
        broadcasted_onsets = Multiply(name='multi_onsets_timbre_only')(
            [present_pianoroll, expanded_onsets])
        broadcasted_offsets = Multiply(name='multi_offsets_timbre_only')(
            [pianoroll_no_gradient, expanded_offsets])

        # Use the last channel for instrument-agnostic midi
        broadcasted_frames = Concatenate(name='multi_frames')(
            [broadcasted_frames, expanded_frames])
        broadcasted_onsets = Concatenate(name='multi_onsets')(
            [broadcasted_onsets, expanded_onsets])
        broadcasted_offsets = Concatenate(name='multi_offsets')(
            [broadcasted_offsets, expanded_offsets])


        losses = {
            'multi_frames': multi_track_loss_wrapper(self.hparams,
                                                     self.hparams.frames_true_weighing),
            'multi_onsets': multi_track_loss_wrapper(self.hparams,
                                                     self.hparams.onsets_true_weighing),
            'multi_offsets': multi_track_loss_wrapper(self.hparams,
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
                    print_report=True)
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
                    print_report=True)
            ],
            'multi_offsets': [
                multi_track_present_accuracy_wrapper(
                    self.hparams.predict_offset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold),
                single_track_present_accuracy_wrapper(
                    self.hparams.predict_offset_threshold),
                multi_track_prf_wrapper(
                    self.hparams.predict_offset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold)
            ]
        }

        return Model(inputs=[spec_512, spec_256, present_instruments],
                     outputs=[broadcasted_frames, broadcasted_onsets,
                              broadcasted_offsets]), losses, accuracies
