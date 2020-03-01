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
    # return librosa.power_to_db(librosa.feature.melspectrogram(audio, hparams.sample_rate,
    #                                       hop_length=hparams.spec_hop_length,
    #                                       fmin=librosa.note_to_hz('A0'),
    #                                         fmax=librosa.note_to_hz('C9'),
    #                                       n_mels=hparams.spec_n_bins,
    #                                       htk=hparams.spec_mel_htk).T)
    return (librosa.core.cqt(
        audio,
        hparams.sample_rate,
        hop_length=hparams.timbre_hop_length,
        fmin=constants.MIN_TIMBRE_PITCH,
        n_bins=constants.SPEC_BANDS,
        bins_per_octave=constants.BINS_PER_OCTAVE
    ).T)


def get_cqt_index(pitch, hparams):
    frequencies = librosa.cqt_frequencies(hparams.spec_n_bins, fmin=constants.MIN_TIMBRE_PITCH,
                                          bins_per_octave=constants.BINS_PER_OCTAVE)

    return np.abs(frequencies - librosa.midi_to_hz(pitch.numpy() - 1)).argmin()


def get_mel_index(pitch, hparams):
    frequencies = librosa.mel_frequencies(hparams.spec_n_bins, fmin=librosa.note_to_hz('A0'),
                                          fmax=librosa.note_to_hz('C9'), htk=hparams.spec_mel_htk)

    return np.abs(frequencies - librosa.midi_to_hz(pitch.numpy())).argmin()


def preprocess_nsynth_example(example_proto, hparams, is_training):
    record = example_proto
    pitch = record['pitch']
    pitch = tf.py_function(functools.partial(get_cqt_index, hparams=hparams),
                           [pitch],
                           tf.int64)

    #tf.print(pitch)
    instrument = record['instrument']
    audio = record['audio']
    velocity = record['velocity']
    instrument_family = record['instrument_family']

    # Transpose so that the data is in [frame, bins] format.
    spec = tf.py_function(
        functools.partial(create_spectrogram, hparams=hparams),
        [audio],
        tf.float32,
        name='samples_to_spec')

    return dict(
        spec=spec,
        pitch=pitch,
        velocity=velocity,
        instrument=instrument,
        instrument_family=instrument_family,
        length=len(audio),
    )


FeatureTensors = collections.namedtuple('FeatureTensors', ('spec', 'pitch'))

LabelTensors = collections.namedtuple('LabelTensors', ('instrument_family',))


def nsynth_input_tensors_to_model_input(
        input_tensors, hparams, is_training):
    """Processes an InputTensor into FeatureTensors and LabelTensors."""
    length = tf.cast(input_tensors['length'], tf.int32)
    spec = tf.reshape(input_tensors['spec'], (-1, hparams_frame_size(hparams), 1))
    pitch = tf.expand_dims(tf.cast(input_tensors['pitch'], tf.int32), 0)
    instrument_family = tf.one_hot(tf.cast(input_tensors['instrument_family'], tf.int32),
                                   hparams.timbre_num_classes)
    # tf.print(instrument_family)

    features = FeatureTensors(
        spec=spec,
        pitch=pitch
        # length=truncated_length,
        # sequence_id=tf.constant(0) if is_training else input_tensors.sequence_id
    )
    labels = LabelTensors(
        instrument_family=instrument_family
    )

    return features, labels


# combine the batched datasets' audio together
def reduce_batch_fn(tensor, hparams=None, is_training=True):
    #tf.print(tensor['pitch'])
    pitch = K.constant(128, dtype=tf.int64)
    instrument = K.constant(0, dtype=tf.int64)
    instrument_family = K.constant(0, dtype=tf.int64)
    velocity = K.constant(0, dtype=tf.int64)

    # randomly leave out some instruments
    instrument_count = random.randint(1, hparams.timbre_training_max_instruments)
    for i in range(instrument_count):
        # otherwise move the audio so diff attack times
        if tensor['pitch'][i] < pitch:
            pitch = tensor['pitch'][i]
            instrument = tensor['instrument'][i]
            instrument_family = tensor['instrument_family'][i]
            velocity = tensor['velocity'][i]
    audio = tf.reduce_sum(tf.sparse.to_dense(tensor['audio']), 0) / instrument_count

    # audio = tf.py_function(lambda a: [sum(x) for x in itertools.zip_longest(*a, fillvalue=0)],
    #                        [audio],
    #                        tf.float32)

    # audio = [sum(x) for x in itertools.zip_longest(*audio, fillvalue=0)]

    return dict(
        pitch=pitch,
        instrument=instrument,
        instrument_family=instrument_family,
        velocity=velocity,
        audio=audio,
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
      params: HParams object specifying hyperparameters. Called 'params' here
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

    if shuffle_examples:
        input_dataset = input_dataset.shuffle(hparams.nsynth_shuffle_buffer_size)

    if is_training:
        input_dataset = input_dataset.repeat()

    input_dataset = input_dataset.map(parse_nsynth_example)

    reduced_dataset = input_dataset.batch(hparams.timbre_training_max_instruments).map(
        functools.partial(reduce_batch_fn, hparams=hparams, is_training=is_training))

    input_map_fn = functools.partial(
        preprocess_nsynth_example, hparams=hparams, is_training=is_training)
    # foo = next(iter(input_dataset.batch(hparams.timbre_training_max_instruments)))
    # reduce_batch_fn(foo, hparams, is_training)
    input_tensors = reduced_dataset.map(input_map_fn)

    model_input = input_tensors.map(
        functools.partial(
            nsynth_input_tensors_to_model_input,
            hparams=hparams, is_training=is_training))

    dataset = model_input.batch(hparams.nsynth_batch_size)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
