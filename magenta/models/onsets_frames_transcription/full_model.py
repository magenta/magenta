import functools

import tensorflow as tf
from dotmap import DotMap
from keras.layers import Multiply

from magenta.models.onsets_frames_transcription import constants, infer_util, data
from magenta.models.onsets_frames_transcription.accuracy_util import flatten_accuracy_wrapper, \
    flatten_loss_wrapper
from magenta.models.onsets_frames_transcription.layer_util import conv_bn_elu_layer, \
    get_all_croppings, time_distributed_wrapper
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal, VarianceScaling
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    Dense, Dropout, \
    Input, concatenate, Lambda, Reshape, LSTM, \
    Flatten, ELU, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import librosa

from magenta.models.onsets_frames_transcription.nsynth_reader import NoteCropping


def sequence_to_note_croppings(sequence, hparams):
    note_croppings = []
    num_notes = 0
    for note in sequence.notes:
        frames_per_second = data.hparams_frames_per_second(hparams)
        note_croppings.append(NoteCropping(pitch=note.pitch,
                                           start_idx=note.start_time * frames_per_second,
                                           end_idx=note.end_time * frames_per_second))
        num_notes += 1
    return note_croppings, num_notes


# TODO is there a better method for this
family_idx_to_midi_instrument = {
    0: 33,  # Acoustic Bass
    1: 62,  # Brass Section
    2: 74,  # Flute
    3: 25,  # Acoustic Nylon Guitar
    4: 1,  # keyboard / Acoustic Grand Piano
    5: 9,  # mallet / Celesta
    6: 17,  # organ / Drawbar Organ
    7: 72,  # reed / Clarinet
    8: 49,  # string / String Ensemble
    9: 82,  # synth lead / Sawtooth
    10: 55,  # vocal / Synth Voice
}


def populate_instruments(sequence, timbre_probs, present_instruments):
    masked_probs = Multiply()([timbre_probs, present_instruments])
    timbre_preds = K.flatten(tf.nn.top_k(masked_probs).indices)
    for i, note in enumerate(sequence.notes):
        note.instrument = family_idx_to_midi_instrument[timbre_preds[i]]

    return sequence


class FullModel:
    def __init__(self, midi_model, timbre_model, hparams):
        if hparams is None:
            hparams = DotMap()
        self.hparams = hparams
        self.midi_model = midi_model
        self.timbre_model = timbre_model

    def get_model(self):
        spec = Input(shape=(None, constants.SPEC_BANDS, 1))
        present_instruments = Input(shape=(self.hparams.timbre_num_classes,))

        frame_probs, onset_probs, offset_probs = self.midi_model(spec)

        frame_predictions = frame_probs > self.hparams.predict_frame_threshold
        onset_predictions = onset_probs > self.hparams.predict_onset_threshold
        offset_predictions = offset_probs > self.hparams.predict_offset_threshold

        sequence = infer_util.predict_sequence(
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            velocity_values=None,
            hparams=self.hparams, min_pitch=constants.MIN_MIDI_PITCH)
        note_croppings, num_notes = sequence_to_note_croppings(sequence, self.hparams)

        timbre_probs = self.timbre_model([spec, note_croppings, num_notes])

        multi_sequence = populate_instruments(sequence, timbre_probs, present_instruments)

        return Model(inputs=[spec, present_instruments], outputs=multi_sequence)
