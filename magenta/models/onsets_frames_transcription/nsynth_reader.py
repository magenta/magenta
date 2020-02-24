import collections
import functools
import itertools
import math
import random
import re
import wave
import zlib
from enum import Enum

import librosa
from magenta.models.onsets_frames_transcription import audio_transform
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription.data import wav_to_spec_op, wav_to_num_frames_op, \
    hparams_frames_per_second, read_examples, create_batch, hparams_frame_size
from magenta.music import audio_io
from magenta.music import melspec_input
from magenta.music import sequences_lib
from magenta.music.protobuf import music_pb2
import numpy as np
import six
import tensorflow.compat.v2 as tf
from tensorflow_core import TensorSpec, TensorShape


def parse_nsynth_example(example_proto):
    features = {
        'pitch': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'instrument': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'instrument_family': tf.io.VarLenFeature(shape=(), dtype=tf.int64),
        'velocity': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'audio': tf.io.VarLenFeature(dtype=tf.float32),
    }
    record = tf.io.parse_single_example(example_proto, features)
    return record


def preprocess_nsynth_example(example_proto, hparams, is_training):
    record = parse_nsynth_example(example_proto)
    pitch = record['pitch']
    instrument = record['instrument']
    audio = record['audio']
    velocity = record['velocity']
    instrument_family = record['instrument_family']

    spec = wav_to_spec_op(audio, hparams=hparams)

    length = wav_to_num_frames_op(audio, hparams_frames_per_second(hparams))

    return dict(
        spec=spec,
        pitch=pitch,
        velocity=velocity,
        instrument=instrument,
        instrument_family=instrument_family,
        length=length,
    )


def nsynth_input_tensors_to_model_input(
        input_tensors, hparams, is_training, num_classes=constants.MIDI_PITCHES):
    """Processes an InputTensor into FeatureTensors and LabelTensors."""
    length = tf.cast(input_tensors.length, tf.int32)
    spec = tf.reshape(input_tensors.spec, (-1, hparams_frame_size(hparams)))
    pitch = tf.reshape(input_tensors.spec, tf.int32)
    instrument_family = tf.reshape(input_tensors.instrument_family, tf.int32)

    features = dict(
        spec=spec,
        high_pass=pitch * 2
        # length=truncated_length,
        # sequence_id=tf.constant(0) if is_training else input_tensors.sequence_id
    )
    labels = dict(
        instrument_family=instrument_family
    )

    return features, labels


# combine the batched datasets' audio together
def reduce_batch_fn(dataset, hparams, is_training):
    unbatched = dataset.unbatch()
    pitch = int('inf')
    instrument = 0
    instrument_family = 0
    velocity = 0
    audio = []
    for element in unbatched:
        # randomly leave out some instruments
        if len(audio) and random.randrange(2) == 0:
            continue
        if element['pitch'] < pitch:
            pitch = element['pitch']
            instrument = element['instrument']
            instrument_family = element['instrument_family']
            velocity = element['velocity']
        audio.append(element['audio'])

    audio = [sum(x) for x in itertools.zip_longest(*audio, fillvalue=0)]

    return dict(
        pitch=pitch,
        instrument=instrument,
        instrument_family=instrument_family,
        velocity=velocity,
        audio=audio
    )


def provide_batch(examples,
                  preprocess_examples,
                  params,
                  is_training,
                  shuffle_examples,
                  skip_n_initial_records, ):
    """Returns batches of tensors read from TFRecord files.

    Args:
      examples: A string path to a TFRecord file of examples, a python list of
        serialized examples, or a Tensor placeholder for serialized examples.
      preprocess_examples: Whether to preprocess examples. If False, assume they
        have already been preprocessed.
      params: HParams object specifying hyperparameters. Called 'params' here
        because that is the interface that TPUEstimator expects.
      is_training: Whether this is a training run.
      shuffle_examples: Whether examples should be shuffled.
      skip_n_initial_records: Skip this many records at first.

    Returns:
      Batched tensors in a TranscriptionData NamedTuple.
    """
    hparams = params

    input_dataset = read_examples(
        examples, is_training, shuffle_examples, skip_n_initial_records, hparams)

    input_dataset.batch(hparams).map(functools.partial(
        reduce_batch_fn, hparams=hparams, is_training=is_training))

    input_map_fn = functools.partial(
        preprocess_nsynth_example, hparams=hparams, is_training=is_training)

    input_tensors = input_dataset.map(input_map_fn)

    model_input = input_tensors.map(
        functools.partial(
            nsynth_input_tensors_to_model_input ,
            hparams=hparams, is_training=is_training))

    dataset = create_batch(model_input, hparams=hparams, is_training=is_training)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
