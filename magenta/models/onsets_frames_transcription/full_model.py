import operator

import numpy as np
import tensorflow as tf
from dotmap import DotMap
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Multiply
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
def get_default_hparams():
    return {
        'prediction_generosity': 4,
        'multiple_instruments_threshold': 0.16
    }


class FullModel:
    def __init__(self, midi_model, timbre_model, hparams):
        if hparams is None:
            hparams = DotMap()
        self.hparams = hparams
        self.midi_model = midi_model
        self.timbre_model = timbre_model

    def fill_pianoroll_area(self, sorted_croppings, filling_fn):
        for i, eager_cropping in enumerate(sorted_croppings):
            cropping = NoteCropping(*eager_cropping)
            pitch = cropping.pitch - constants.MIN_MIDI_PITCH
            start_idx = K.cast(cropping.start_idx / self.hparams.timbre_hop_length, 'int64')
            end_idx = K.cast(cropping.end_idx / self.hparams.timbre_hop_length, 'int64')
            if pitch > 0:
                # guess that anything higher and to the right of this note has these probs
                # for each i, this area is partially overridden
                filling_fn(i, pitch, start_idx, end_idx)

    def note_croppings_to_pianorolls(self, input_list):
        batched_note_croppings, batched_timbre_probs, batched_pianoroll_length = input_list
        pianoroll_list = []
        for batch_idx in range(K.int_shape(batched_note_croppings)[0]):
            note_croppings = batched_note_croppings[batch_idx]
            timbre_probs = batched_timbre_probs[batch_idx]

            # Make pitch the first dimension for easy manipulation
            # default to equal guess for the instruments
            pianorolls = np.ones(
                shape=(constants.MIDI_PITCHES,
                       batched_pianoroll_length[batch_idx][0],
                       self.hparams.timbre_num_classes)) / (1 + self.hparams.timbre_num_classes)

            def full_fill(i, pitch, start_idx, end_idx):
                pianorolls[pitch:, start_idx:] = timbre_probs[i]

            def time_fill(i, pitch, start_idx, end_idx):
                pianorolls[pitch, start_idx:] = timbre_probs[i]

            def pitch_fill(i, pitch, start_idx, end_idx):
                pianorolls[pitch:, start_idx:end_idx + 1] = timbre_probs[i]

            def exact_fill(i, pitch, start_idx, end_idx):
                pianorolls[pitch, start_idx:end_idx + 1] = timbre_probs[i]

            pitch_first = sorted(note_croppings, key=operator.itemgetter(0, 1))
            time_first = sorted(note_croppings, key=operator.itemgetter(1, 0))

            self.fill_pianoroll_area(pitch_first, full_fill)
            self.fill_pianoroll_area(time_first, time_fill)
            self.fill_pianoroll_area(pitch_first, pitch_fill)
            self.fill_pianoroll_area(note_croppings, exact_fill)

            # make time the first dimension
            pianoroll_list.append(K.permute_dimensions(pianorolls, (1, 0, 2)))

        return pianoroll_list

    def sequence_to_note_croppings(self, sequence):
        note_croppings = []
        for note in sequence.notes:
            note_croppings.append(NoteCropping(pitch=note.pitch,
                                               start_idx=note.start_time * self.hparams.sample_rate,
                                               end_idx=note.end_time * self.hparams.sample_rate))
        # pitch is more important than start time for what an instrument is
        # we also don't want to overwrite definitive predictions
        return note_croppings

    def get_croppings(self, input_list):
        """Convert frame predictions into a sequence."""
        batched_frame_predictions, batched_onset_predictions, batched_offset_predictions = \
            input_list

        print(f'num_onsets: {K.sum(K.cast_to_floatx(batched_onset_predictions))}')
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
                                                               dtype=K.floatx(),
                                                               value=-1.0)
        return padded

    def get_dynamic_length(self, batched_dynamic_tensor):
        print(f'dynamic length: {K.int_shape(batched_dynamic_tensor)[1]}')
        return K.expand_dims(
            tf.repeat(K.expand_dims(K.int_shape(batched_dynamic_tensor)[1], axis=0),
                      K.int_shape(batched_dynamic_tensor)[0])
        )

    def separate_batches(self, input_list):
        # TODO implement
        raise NotImplementedError

    def get_model(self):
        spec_512 = Input(shape=(None, constants.SPEC_BANDS, 1), name='midi_spec')
        spec_256 = Input(shape=(None, constants.SPEC_BANDS, 1), name='timbre_spec')
        present_instruments = Input(shape=(self.hparams.timbre_num_classes,))

        frame_probs, onset_probs, offset_probs = self.midi_model.call([spec_512])

        # decrease threshold to feed more notes into the timbre prediction
        # even if they don't make the final cut in accuracy_util.multi_track_accuracy_wrapper
        frame_predictions = K.stop_gradient(frame_probs > self.hparams.predict_frame_threshold)
        # generous onsets are used so that we can get more frame prediction data for the instruments
        # this will end up making our end-predicted notes much shorter though
        generous_onset_predictions = K.stop_gradient(onset_probs > (
                self.hparams.predict_onset_threshold / self.hparams.prediction_generosity))
        offset_predictions = K.stop_gradient(offset_probs > self.hparams.predict_offset_threshold)

        note_croppings = Lambda(self.get_croppings,
                                output_shape=(None, 3),
                                dynamic=True)(
            [frame_predictions, generous_onset_predictions, offset_predictions])

        note_croppings = K.cast(note_croppings, 'int64')
        # num_notes = Lambda(self.get_dynamic_length,
        #                    output_shape=(1,),
        #                    dtype='int64',
        #                    dynamic=True)(note_croppings)

        pianoroll_length = Lambda(self.get_dynamic_length,
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

        expanded_present_instruments = K.expand_dims(present_instruments, 1)
        present_timbre_probs = Multiply()([timbre_probs, expanded_present_instruments])

        norm_sum = Multiply()([1 / K.sum(timbre_probs, -1), K.sum(present_timbre_probs, -1)])
        # normalize
        present_timbre_probs = Multiply()([present_timbre_probs,
                                           K.expand_dims(1 / norm_sum)])

        timbre_pianoroll = Lambda(self.note_croppings_to_pianorolls,
                                  dynamic=True,
                                  output_shape=(None,
                                                constants.MIDI_PITCHES,
                                                self.hparams.timbre_num_classes))(
            [note_croppings, present_timbre_probs, pianoroll_length])

        expanded_frames = K.cast_to_floatx(K.expand_dims(frame_probs))
        expanded_onsets = K.cast_to_floatx(K.expand_dims(onset_probs))
        expanded_offsets = K.cast_to_floatx(K.expand_dims(offset_probs))

        broadcasted_frames = Multiply(name='multi_frames')([timbre_pianoroll, expanded_frames])
        broadcasted_onsets = Multiply(name='multi_onsets')([timbre_pianoroll, expanded_onsets])
        broadcasted_offsets = Multiply(name='multi_offsets')([timbre_pianoroll, expanded_offsets])

        # multi_sequence = populate_instruments(sequence, timbre_probs, present_instruments)

        losses = {
            'multi_frames': multi_track_loss_wrapper(self.hparams.frames_true_weighing),
            'multi_onsets': multi_track_loss_wrapper(self.hparams.onsets_true_weighing),
            'multi_offsets': multi_track_loss_wrapper(self.hparams.offsets_true_weighing),
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
