import collections
import functools
import functools
import itertools
import random

import librosa
import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add
import numpy as np
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription.data import create_batch, hparams_frame_size, \
    hparams_frames_per_second, read_examples, wav_to_num_frames_op, wav_to_spec_op


def parse_nsynth_example(example_proto):
    features = {
        'pitch': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'instrument': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'instrument_family': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'velocity': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'audio': tf.io.VarLenFeature(dtype=tf.float32),
    }
    record = tf.io.parse_single_example(example_proto, features)
    return record


def create_spectrogram(audio, hparams):
    audio = audio.numpy()
    if hparams.timbre_spec_type == 'mel':
        spec = librosa.feature.melspectrogram(
            audio,
            hparams.sample_rate,
            hop_length=hparams.timbre_hop_length,
            fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
            fmax=librosa.midi_to_hz(constants.MAX_TIMBRE_PITCH),
            n_mels=constants.TIMBRE_SPEC_BANDS,
            pad_mode='symmetric',
            htk=hparams.spec_mel_htk,
            power=2
        ).T
        spec = librosa.power_to_db(spec)

    else:
        spec = librosa.core.cqt(
            audio,
            hparams.sample_rate,
            hop_length=hparams.timbre_hop_length,
            fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
            n_bins=constants.TIMBRE_SPEC_BANDS,
            bins_per_octave=constants.BINS_PER_OCTAVE,
            pad_mode='symmetric'
        ).T
        spec = librosa.amplitude_to_db(np.abs(spec))

    # convert amplitude to power
    if hparams.timbre_spec_log_amplitude:
        spec = spec - librosa.power_to_db(np.array([0]))[0]
    else:
        spec = librosa.db_to_power(spec)
    return spec


def get_cqt_index(pitch, hparams):
    frequencies = librosa.cqt_frequencies(constants.TIMBRE_SPEC_BANDS,
                                          fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
                                          bins_per_octave=constants.BINS_PER_OCTAVE)

    return np.abs(frequencies - librosa.midi_to_hz(pitch.numpy() - 1)).argmin()


def get_mel_index(pitch, hparams):
    frequencies = librosa.mel_frequencies(constants.TIMBRE_SPEC_BANDS,
                                          fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
                                          fmax=librosa.midi_to_hz(constants.MAX_TIMBRE_PITCH),
                                          htk=hparams.spec_mel_htk)

    return np.abs(frequencies - librosa.midi_to_hz(pitch.numpy())).argmin()


FeatureTensors = collections.namedtuple('FeatureTensors', ('spec', 'note_croppings', 'num_notes'))

LabelTensors = collections.namedtuple('LabelTensors', ('instrument_families',))

NoteCropping = collections.namedtuple('NoteCropping', ('pitch', 'start_idx', 'end_idx'))

NoteLabel = collections.namedtuple('NoteLabel', ('instrument_family'))


def nsynth_input_tensors_to_model_input(
        input_tensors, hparams, is_training):
    """Processes an InputTensor into FeatureTensors and LabelTensors."""
    # length = tf.cast(input_tensors['length'], tf.int32)
    spec = tf.reshape(input_tensors['spec'], (-1, constants.TIMBRE_SPEC_BANDS, 1))
    note_croppings = input_tensors['note_croppings']
    instrument_families = input_tensors['instrument_families']
    num_notes = input_tensors['num_notes']
    # instrument_family = input_tensors['instrument_family']
    # tf.print(instrument_family)

    features = FeatureTensors(
        spec=spec,
        note_croppings=note_croppings,
        num_notes=num_notes
        # length=truncated_length,
        # sequence_id=tf.constant(0) if is_training else input_tensors.sequence_id
    )
    labels = LabelTensors(
        instrument_families=instrument_families
    )

    return features, labels


# combine the batched datasets' audio together
def reduce_batch_fn(tensor, hparams=None, is_training=True):
    # randomly leave out some instruments
    instrument_count = hparams.timbre_training_max_instruments
    note_croppping_list = []
    instrument_family_list = []
    audios = []
    max_length = 0
    pitch_idx_fn = get_cqt_index if hparams.timbre_spec_type == 'cqt' else get_mel_index
    for i in range(instrument_count):
        # otherwise move the audio so diff attack times
        pitch = tensor['pitch'][i]
        pitch = tf.py_function(functools.partial(pitch_idx_fn, hparams=hparams),
                               [pitch],
                               tf.int64)
        start_idx = tf.random.uniform((), minval=0, maxval=hparams.timbre_max_start_offset,
                                      dtype='int64')
        audio = K.concatenate([tf.zeros(start_idx), tf.sparse.to_dense(tensor['audio'])[i]])

        end_idx = len(audio)
        if hparams.timbre_max_len and end_idx > hparams.timbre_max_len:
            audio = tf.slice(audio, begin=[0], size=[hparams.timbre_max_len])
            end_idx = hparams.timbre_max_len
        if end_idx > max_length:
            max_length = end_idx

        audios.append(audio)

        instrument = tensor['instrument'][i]
        instrument_family = tensor['instrument_family'][i]
        velocity = tensor['velocity'][i]
        note_croppping_list.append(NoteCropping(
            pitch=pitch,
            start_idx=start_idx,
            end_idx=end_idx
        ))
        instrument_family_list.append(tf.one_hot(tf.cast(instrument_family, tf.int32),
                                                 hparams.timbre_num_classes))

    # pad the end of the shorter audio clips
    audios = list(map(lambda x: tf.pad(x, [[0, max_length - len(x)]]), audios))

    combined_audio = tf.reduce_sum(tf.convert_to_tensor(audios), axis=0) / instrument_count

    # ensure all audios in batches are the same length
    if hparams.timbre_max_len:
        pad_length = hparams.timbre_max_len
    else:
        pad_length = hparams.timbre_max_start_offset + 5 * hparams.sample_rate
    combined_audio = tf.pad(combined_audio, [[0, pad_length - tf.shape(combined_audio)[0]]])
    note_croppings = tf.convert_to_tensor(note_croppping_list, dtype=tf.int32)
    instrument_families = tf.convert_to_tensor(instrument_family_list, dtype=tf.int32)

    # Transpose so that the data is in [frame, bins] format.
    spec = tf.py_function(
        functools.partial(create_spectrogram, hparams=hparams),
        [combined_audio],
        tf.float32,
        name='samples_to_spec')

    return dict(
        spec=spec,
        note_croppings=note_croppings,
        instrument_families=instrument_families,
        num_notes=instrument_count
    )


def provide_batch(examples,
                  params,
                  preprocess_examples,
                  is_training,
                  shuffle_examples,
                  skip_n_initial_records, ):
    """Returns batches of tensors read from TFRecord files.

    Args:
      examples: A string path to a TFRecord file of examples, a python list of
        serialized examples, or a Tensor placeholder for serialized examples.
      params: HParams object specifying hyperparameters. Called 'pK.constant(note_croppings)arams' here
        because that is the interface that TPUEstimator expects.
      is_training: Whether this is a training run.
      shuffle_examples: Whether examples should be shuffled.
      skip_n_initial_records: Skip this many records at first.

    Returns:
      Batched tensors in a dict.
    """
    hparams = params

    input_dataset = read_examples(
        examples, is_training, shuffle_examples, skip_n_initial_records, hparams)

    # if shuffle_examples:
    #     input_dataset = input_dataset.shuffle(hparams.nsynth_shuffle_buffer_size)

    # if is_training:
    #     input_dataset = input_dataset.repeat()

    input_dataset = input_dataset.map(parse_nsynth_example)

    # foo = next(iter(input_dataset.batch(hparams.timbre_training_max_instruments)))
    # reduce_batch_fn(foo, hparams, is_training)

    reduced_dataset = input_dataset.batch(hparams.timbre_training_max_instruments).map(
        functools.partial(reduce_batch_fn, hparams=hparams, is_training=is_training))

    # input_map_fn = functools.partial(
    #     preprocess_nsynth_example, hparams=hparams, is_training=is_training)
    # input_tensors = reduced_dataset.map(input_map_fn)

    model_input = reduced_dataset.map(
        functools.partial(
            nsynth_input_tensors_to_model_input,
            hparams=hparams, is_training=is_training))

    dataset = model_input.batch(hparams.nsynth_batch_size)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
