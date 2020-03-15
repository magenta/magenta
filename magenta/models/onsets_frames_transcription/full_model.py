import tensorflow as tf
from dotmap import DotMap

from magenta.models.onsets_frames_transcription import constants, infer_util, data
from magenta.models.onsets_frames_transcription.accuracy_util import binary_accuracy_wrapper, \
    f1_wrapper
from magenta.models.onsets_frames_transcription.instrument_family_mappings import \
    family_to_midi_instrument
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Multiply
from tensorflow.keras.models import Model

from magenta.models.onsets_frames_transcription.loss_util import log_loss_wrapper
from magenta.models.onsets_frames_transcription.nsynth_reader import NoteCropping


def note_croppings_to_pianorolls(input_list):
    batched_note_croppings, batched_timbre_probs = input_list
    pianoroll_list = []
    for batch_idx in range(K.int_shape(batched_note_croppings)[0]):
        note_croppings = batched_note_croppings[batch_idx]
        timbre_probs = batched_timbre_probs[batch_idx]
        pianorolls = K.zeros(
            shape=(None, constants.MIDI_PITCHES, constants.NUM_INSTRUMENT_FAMILIES))
        for i, cropping in enumerate(note_croppings):
            pitch = cropping.pitch - constants.MIN_MIDI_PITCH
            start_idx = cropping.start_idx
            end_idx = cropping.end_idx
            pianorolls[start_idx:end_idx][pitch] += timbre_probs[i]

        pianoroll_list.append(pianorolls)

    return tf.convert_to_tensor(pianoroll_list)


def populate_instruments(sequence, timbre_probs, present_instruments):
    masked_probs = Multiply()([timbre_probs, present_instruments])
    timbre_preds = K.flatten(tf.nn.top_k(masked_probs).indices)
    for i, note in enumerate(sequence.notes):
        note.instrument = family_to_midi_instrument[timbre_preds[i]]

    return sequence


class FullModel:
    def __init__(self, midi_model, timbre_model, hparams):
        if hparams is None:
            hparams = DotMap()
        self.hparams = hparams
        self.midi_model = midi_model
        self.timbre_model = timbre_model

    def sequence_to_note_croppings(self, sequence):
        note_croppings = []
        num_notes = 0
        for note in sequence.notes:
            frames_per_second = data.hparams_frames_per_second(self.hparams)
            note_croppings.append(NoteCropping(pitch=note.pitch,
                                               start_idx=note.start_time * frames_per_second,
                                               end_idx=note.end_time * frames_per_second))
            num_notes += 1
        return note_croppings

    def get_croppings(self, input_list):
        """Convert frame predictions into a sequence."""
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
        return tf.convert_to_tensor(croppings_list)

    def get_num_notes(self, batched_note_croppings):
        num_notes_list = []
        for batch_idx in range(K.int_shape(batched_note_croppings)[0]):
            num_notes = K.int_shape(batched_note_croppings[batch_idx])[0]
            num_notes_list.append([num_notes])
        return tf.convert_to_tensor(num_notes_list)

    def separate_batches(self, input_list):
        # TODO implement
        raise NotImplementedError

    def get_model(self):
        spec = Input(shape=(None, constants.SPEC_BANDS, 1))
        present_instruments = Input(shape=(self.hparams.timbre_num_classes,))

        frame_probs, onset_probs, offset_probs = self.midi_model.call([spec])

        frame_predictions = frame_probs > self.hparams.predict_frame_threshold
        onset_predictions = onset_probs > self.hparams.predict_onset_threshold
        offset_predictions = offset_probs > self.hparams.predict_offset_threshold

        note_croppings = Lambda(self.get_croppings, output_shape=(None, 3), dynamic=True)(
            [frame_predictions, onset_predictions, offset_predictions])
        note_croppings = K.cast(note_croppings, 'int64')
        num_notes = Lambda(self.get_num_notes, output_shape=(1,), dtype='int64', dynamic=True)(
            note_croppings)
        # num_notes = K.cast(num_notes, 'int64')

        timbre_probs = self.timbre_model.call([spec, note_croppings, num_notes])

        if self.hparams.timbre_coagulate_mini_batches:
            # re-separate
            timbre_probs = Lambda(self.separate_batches, dynamic=True,
                                  output_shape=(None, constants.NUM_INSTRUMENT_FAMILIES))(
                [timbre_probs, num_notes])

        expanded_present_instruments = K.expand_dims(present_instruments, 1)
        present_timbre_probs = Multiply()([timbre_probs, expanded_present_instruments])

        timbre_pianoroll = Lambda(note_croppings_to_pianorolls,
                                  dynamic=True,
                                  output_shape=(None,
                                                constants.MIDI_PITCHES,
                                                constants.NUM_INSTRUMENT_FAMILIES))(
            [note_croppings, present_timbre_probs])

        expanded_frames = K.cast_to_floatx(K.expand_dims(frame_predictions))
        expanded_onsets = K.cast_to_floatx(K.expand_dims(onset_predictions))
        expanded_offsets = K.cast_to_floatx(K.expand_dims(offset_predictions))

        broadcasted_frames = Multiply(name='multi_frames')([timbre_pianoroll, expanded_frames])
        broadcasted_onsets = Multiply(name='multi_onsets')([timbre_pianoroll, expanded_onsets])
        broadcasted_offsets = Multiply(name='multi_offsets')([timbre_pianoroll, expanded_offsets])

        # multi_sequence = populate_instruments(sequence, timbre_probs, present_instruments)

        losses = {
            'multi_frames': log_loss_wrapper(self.hparams.frames_true_weighing),
            'multi_onsets': log_loss_wrapper(self.hparams.onsets_true_weighing),
            'multi_offsets': log_loss_wrapper(self.hparams.offsets_true_weighing),
        }

        accuracies = {
            'multi_frames': [binary_accuracy_wrapper(self.hparams.predict_frame_threshold),
                             f1_wrapper(self.hparams.predict_frame_threshold)],
            'multi_onsets': [binary_accuracy_wrapper(self.hparams.predict_onset_threshold),
                             f1_wrapper(self.hparams.predict_onset_threshold)],
            'multi_offsets': [binary_accuracy_wrapper(self.hparams.predict_offset_threshold),
                              f1_wrapper(self.hparams.predict_offset_threshold)]
        }

        return Model(inputs=[spec, present_instruments],
                     outputs=[broadcasted_frames, broadcasted_onsets,
                              broadcasted_offsets]), losses, accuracies
