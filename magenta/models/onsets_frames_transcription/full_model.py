import tensorflow as tf
from dotmap import DotMap
from keras.layers import Multiply

from magenta.models.onsets_frames_transcription import constants, infer_util, data
from magenta.models.onsets_frames_transcription.instrument_family_mappings import \
    family_to_midi_instrument
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from magenta.models.onsets_frames_transcription.nsynth_reader import NoteCropping


def note_croppings_to_pianorolls(note_croppings, timbre_probs):
    pianorolls = K.zeros(shape=(None, constants.MIDI_PITCHES, constants.NUM_INSTRUMENT_FAMILIES))
    for i, cropping in enumerate(note_croppings):
        pitch = cropping.pitch
        start_idx = cropping.start_idx
        end_idx = cropping.end_idx
        pianorolls[start_idx:end_idx][pitch] += timbre_probs[i]

    return pianorolls


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
        frame_predictions, onset_predictions, offset_predictions = input_list

        sequence = infer_util.predict_sequence(
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            velocity_values=None,
            hparams=self.hparams, min_pitch=constants.MIN_MIDI_PITCH)
        return self.sequence_to_note_croppings(sequence)

    def get_model(self):
        spec = Input(shape=(None, constants.SPEC_BANDS, 1))
        present_instruments = Input(shape=(self.hparams.timbre_num_classes,))

        frame_probs, onset_probs, offset_probs = self.midi_model(spec)

        frame_predictions = frame_probs > self.hparams.predict_frame_threshold
        onset_predictions = onset_probs > self.hparams.predict_onset_threshold
        offset_predictions = offset_probs > self.hparams.predict_offset_threshold

        note_croppings = Lambda(self.get_croppings, output_shape=(None, 3), dynamic=True)([frame_predictions, onset_predictions, offset_predictions])
        num_notes = Lambda(lambda x: x.shape[0], output_shape=(1,), dynamic=True)(note_croppings)
        num_notes = K.cast(num_notes, 'int64')

        timbre_probs = self.timbre_model([spec, note_croppings, num_notes])

        timbre_pianoroll = note_croppings_to_pianorolls(note_croppings, timbre_probs)

        expanded_frames = K.expand_dims(frame_predictions)
        expanded_onsets = K.expand_dims(onset_predictions)
        expanded_offsets = K.expand_dims(offset_predictions)

        broadcasted_frames = Multiply()([timbre_pianoroll, expanded_frames])
        broadcasted_onsets = Multiply()([timbre_pianoroll, expanded_onsets])
        broadcasted_offsets = Multiply()([timbre_pianoroll, expanded_offsets])

        # multi_sequence = populate_instruments(sequence, timbre_probs, present_instruments)

        return Model(inputs=[spec, present_instruments],
                     outputs=[broadcasted_frames, broadcasted_onsets, broadcasted_offsets])

