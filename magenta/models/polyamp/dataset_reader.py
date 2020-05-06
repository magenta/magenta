# Copyright 2020 The Magenta Authors.
# Modifications Copyright 2020 Jack Spencer Smith.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared methods for providing data to transcription models.

Glossary (definitions may not hold outside of this particular file):
  sample: The value of an audio waveform at a discrete timepoint.
  frame: An individual row of a constant-Q transform computed from some
      number of audio samples.
  example: An individual training example. The number of frames in an example
      is determined by the sequence length.
"""
from __future__ import absolute_import, division, print_function

import collections
import copy
import functools
import re
import wave

import librosa
import numpy as np
import six
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_core import TensorShape

from magenta.models.polyamp import audio_transform, constants, instrument_family_mappings
from magenta.music import audio_io, sequences_lib
from magenta.music.protobuf import music_pb2


def hparams_frame_size(hparams):
    """Find the frame size of the input conditioned on the input type."""
    if hparams.spec_type == 'raw':
        return hparams.spec_hop_length
    return hparams.spec_n_bins


def hparams_frames_per_second(hparams):
    """Compute frames per second as a function of HParams."""
    return hparams.sample_rate / hparams.spec_hop_length


def _wav_to_cqt(wav_audio, hparams):
    """Transforms the contents of a wav file into a series of CQT frames."""
    y = audio_io.wav_data_to_samples(wav_audio, hparams.sample_rate)

    return samples_to_cqt(y, hparams)


def samples_to_cqt(y, hparams):
    cqt = np.abs(
        librosa.core.cqt(
            y,
            hparams.sample_rate,
            hop_length=hparams.spec_hop_length,
            fmin=hparams.spec_fmin,
            n_bins=hparams.spec_n_bins,
            bins_per_octave=hparams.cqt_bins_per_octave),
        dtype=np.float32)
    # Transpose so that the data is in [frame, bins] format.
    cqt = cqt.T
    return cqt


def _wav_to_mel(wav_audio, hparams):
    """Transforms the contents of a wav file into a series of mel spec frames."""
    y = audio_io.wav_data_to_samples(wav_audio, hparams.sample_rate)

    mel = librosa.feature.melspectrogram(
        y,
        hparams.sample_rate,
        hop_length=hparams.spec_hop_length,
        fmin=hparams.spec_fmin,
        n_mels=hparams.spec_n_bins,
        htk=hparams.spec_mel_htk).astype(np.float32)

    # Transpose so that the data is in [frame, bins] format.
    mel = mel.T
    return mel


def _wav_to_framed_samples(wav_audio, hparams):
    """Transforms the contents of a wav file into a series of framed samples."""
    y = audio_io.wav_data_to_samples(wav_audio, hparams.sample_rate)

    hl = hparams.spec_hop_length
    n_frames = int(np.ceil(y.shape[0] / hl))
    frames = np.zeros((n_frames, hl), dtype=np.float32)

    # Fill in everything but the last frame which may not be the full length
    cutoff = (n_frames - 1) * hl
    frames[:n_frames - 1, :] = np.reshape(y[:cutoff], (n_frames - 1, hl))
    # Fill the last frame
    remain_len = len(y[cutoff:])
    frames[n_frames - 1, :remain_len] = y[cutoff:]

    return frames


def wav_to_spec(wav_audio, hparams):
    """Transforms the contents of a wav file into a series of spectrograms."""
    wav_audio = wav_audio.numpy()  # convert eager tensor to numpy
    if hparams.spec_type == 'raw':
        spec = _wav_to_framed_samples(wav_audio, hparams)
    else:
        if hparams.spec_type == 'cqt':
            spec = _wav_to_cqt(wav_audio, hparams)
        elif hparams.spec_type == 'mel':
            spec = _wav_to_mel(wav_audio, hparams)
        else:
            raise ValueError('Invalid spec_type: {}'.format(hparams.spec_type))

        if hparams.spec_log_amplitude:
            spec = librosa.power_to_db(spec)

    return spec


def wav_to_spec_op(wav_audio, hparams):
    """Return an op for converting wav audio to a spectrogram."""
    spec = tf.py_function(
        functools.partial(wav_to_spec, hparams=hparams),
        [wav_audio],
        tf.float32,
        name='wav_to_spec')
    spec.set_shape([None, hparams_frame_size(hparams)])
    return spec


def wav_to_num_frames(wav_audio, frames_per_second):
    """Transforms a wav-encoded audio string into number of frames."""
    wav_audio = wav_audio.numpy()  # convert to numpy
    w = wave.open(six.BytesIO(wav_audio))
    return np.int32(w.getnframes() / w.getframerate() * frames_per_second)


def wav_to_num_frames_op(wav_audio, frames_per_second):
    """Transforms a wav-encoded audio string into number of frames."""
    res = tf.py_function(
        functools.partial(wav_to_num_frames, frames_per_second=frames_per_second),
        [wav_audio],
        tf.int32,
        name='wav_to_num_frames_op')
    res.set_shape(())
    return res


def transform_wav_data_op(wav_data_tensor, hparams, jitter_amount_sec):
    """Transforms with audio for data augmentation. Only for training."""

    def transform_wav_data(wav_data):
        """Transforms with sox."""
        if jitter_amount_sec:
            wav_data = audio_io.jitter_wav_data(wav_data, hparams.sample_rate,
                                                jitter_amount_sec)
        wav_data = audio_transform.transform_wav_audio(wav_data, hparams)

        return [wav_data]

    return tf.py_function(
        transform_wav_data, [wav_data_tensor],
        tf.string,
        name='transform_wav_data_op')


def _sequence_to_pianoroll_fn(sequence_tensor, velocity_range_tensor,
                              hparams, instrument_family=None):
    """Converts sequence to pianorolls."""
    if instrument_family is not None and instrument_family < 0:
        instrument_family = None
    velocity_range = music_pb2.VelocityRange.FromString(velocity_range_tensor.numpy())
    sequence = music_pb2.NoteSequence.FromString(sequence_tensor.numpy())
    sequence = sequences_lib.apply_sustain_control_changes(sequence)
    roll = sequences_lib.sequence_to_pianoroll(
        sequence,
        frames_per_second=hparams_frames_per_second(hparams),
        min_pitch=constants.MIN_MIDI_PITCH,
        max_pitch=constants.MAX_MIDI_PITCH,
        min_frame_occupancy_for_label=hparams.min_frame_occupancy_for_label,
        onset_mode=hparams.onset_mode,
        onset_length_ms=hparams.onset_length,
        offset_length_ms=hparams.offset_length,
        onset_delay_ms=hparams.onset_delay,
        min_velocity=velocity_range.min,
        max_velocity=velocity_range.max,
        instrument_family=instrument_family,
        use_drums=hparams.use_drums,
        timbre_num_classes=hparams.timbre_num_classes)
    return (roll.active, roll.weights, roll.onsets, roll.onset_velocities,
            roll.offsets)


def sequence_to_pianoroll_op(sequence_tensor, velocity_range_tensor, hparams):
    """Transforms a serialized NoteSequence to a pianoroll."""
    to_pianoroll_fn = functools.partial(_sequence_to_pianoroll_fn, hparams=hparams)
    res, weighted_res, onsets, velocities, offsets = tf.py_function(
        to_pianoroll_fn, [sequence_tensor, velocity_range_tensor],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name='sequence_to_pianoroll_op')
    res.set_shape([None, constants.MIDI_PITCHES])
    weighted_res.set_shape([None, constants.MIDI_PITCHES])
    onsets.set_shape([None, constants.MIDI_PITCHES])
    offsets.set_shape([None, constants.MIDI_PITCHES])
    velocities.set_shape([None, constants.MIDI_PITCHES])

    return res, weighted_res, onsets, offsets, velocities


def sequence_to_multi_pianoroll_op(sequence_tensor, velocity_range_tensor, hparams):
    """Transforms a serialized NoteSequence to a multi-instrument pianoroll."""
    to_pianoroll_fn = functools.partial(_sequence_to_pianoroll_fn, hparams=hparams)
    res_list = []
    weighted_res_list = []
    onsets_list = []
    velocities_list = []
    offsets_list = []
    for i in range(hparams.timbre_num_classes + 1):
        # Put instrument-agnostic labels in the last channel.
        if i == hparams.timbre_num_classes:
            i = -1
        res, weighted_res, onsets, velocities, offsets = tf.py_function(
            to_pianoroll_fn, [sequence_tensor, velocity_range_tensor, i],
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
            name='sequence_to_pianoroll_op')
        res.set_shape([None, constants.MIDI_PITCHES])
        weighted_res.set_shape([None, constants.MIDI_PITCHES])
        onsets.set_shape([None, constants.MIDI_PITCHES])
        offsets.set_shape([None, constants.MIDI_PITCHES])
        velocities.set_shape([None, constants.MIDI_PITCHES])
        res_list.append(K.expand_dims(res))
        weighted_res_list.append(K.expand_dims(weighted_res))
        onsets_list.append(K.expand_dims(onsets))
        velocities_list.append(K.expand_dims(velocities))
        offsets_list.append(K.expand_dims(offsets))
    return K.concatenate(res_list), \
           K.concatenate(weighted_res_list), \
           K.concatenate(onsets_list), \
           K.concatenate(offsets_list), \
           K.concatenate(velocities_list)


def jitter_label_op(sequence_tensor, jitter_amount_sec):
    def jitter_label(sequence_tensor):
        sequence = music_pb2.NoteSequence.FromString(sequence_tensor.numpy())
        sequence = sequences_lib.shift_sequence_times(sequence, jitter_amount_sec)
        return sequence.SerializeToString()

    return tf.py_function(jitter_label, [sequence_tensor], tf.string)


def truncate_note_sequence(sequence, truncate_secs):
    """Truncates a NoteSequence to the given length."""
    sus_sequence = sequences_lib.apply_sustain_control_changes(sequence)

    truncated_seq = music_pb2.NoteSequence()

    for note in sus_sequence.notes:
        start_time = note.start_time
        end_time = note.end_time

        if start_time > truncate_secs:
            continue

        if end_time > truncate_secs:
            end_time = truncate_secs

        modified_note = truncated_seq.notes.add()
        modified_note.MergeFrom(note)
        modified_note.start_time = start_time
        modified_note.end_time = end_time
    if truncated_seq.notes:
        truncated_seq.total_time = truncated_seq.notes[-1].end_time
    return truncated_seq


def get_present_instruments_op(sequence_tensor, hparams=None):
    """Get a list of instruments present in this NoteSequence"""

    def get_present_instruments_fn(sequence_tensor_):
        sequence = music_pb2.NoteSequence.FromString(sequence_tensor_.numpy())
        present_list = np.zeros(hparams.timbre_num_classes, dtype=np.bool)
        for note in sequence.notes:
            note_family = instrument_family_mappings.midi_instrument_to_family[note.program]
            if note_family.value < hparams.timbre_num_classes:
                present_list[note_family.value] = True

        return present_list

    if hparams.use_all_instruments:
        # Ignore what's there; allow raw predictions.
        return np.ones(hparams.timbre_num_classes)
    res = tf.py_function(
        get_present_instruments_fn,
        [sequence_tensor],
        tf.bool)
    res.set_shape(hparams.timbre_num_classes)
    return K.cast_to_floatx(res)


def truncate_note_sequence_op(sequence_tensor, truncated_length_frames,
                              hparams):
    """Truncates a NoteSequence to the given length."""

    def truncate(sequence_tensor, num_frames):
        sequence = music_pb2.NoteSequence.FromString(sequence_tensor.numpy())
        num_secs = num_frames / hparams_frames_per_second(hparams)
        return truncate_note_sequence(sequence, num_secs).SerializeToString()

    res = tf.py_function(
        truncate,
        [sequence_tensor, truncated_length_frames],
        tf.string)
    res.set_shape(())
    return res


InputTensors = collections.namedtuple('InputTensors',
                                      ('spec', 'labels',
                                       'length', 'onsets',
                                       'offsets', 'velocities',
                                       'note_sequence'))


def parse_example(example_proto):
    features = {
        'sequence': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'audio': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'velocity_range': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    }
    record = tf.io.parse_single_example(example_proto, features)
    return record


def preprocess_example(example_proto, hparams, is_training, parse_proto=True):
    """Compute spectral representation, labels, and length from sequence/audio.

    Args:
      example_proto: Example that has not been preprocessed.
      hparams: HParams object specifying hyperparameters.
      is_training: Whether or not this is a training run.

    Returns:
      An InputTensors tuple.

    Raises:
      ValueError: If hparams is contains an invalid spec_type.
    """
    record = parse_example(example_proto) if parse_proto else example_proto
    sequence = record['sequence']
    audio = record['audio']
    velocity_range = record['velocity_range']

    wav_jitter_amount_ms = label_jitter_amount_ms = 0.
    # if there is combined jitter, we must generate it once here
    if is_training and hparams.jitter_amount_ms > 0:
        wav_jitter_amount_ms = np.random.choice(hparams.jitter_amount_ms, size=1)
        label_jitter_amount_ms = wav_jitter_amount_ms

    if label_jitter_amount_ms > 0:
        sequence = jitter_label_op(sequence, label_jitter_amount_ms / 1000.)

    # possibly shift the entire sequence backward for better forward only training
    if hparams.backward_shift_amount_ms > 0:
        sequence = jitter_label_op(sequence,
                                   hparams.backward_shift_amount_ms / 1000.)

    if is_training:
        audio = transform_wav_data_op(
            audio,
            hparams=hparams,
            jitter_amount_sec=wav_jitter_amount_ms / 1000.)

    spec = wav_to_spec_op(audio, hparams=hparams)

    if hparams.split_pianoroll:
        # make a second spec that will be used for timbre prediction
        temp_hparams = copy.deepcopy(hparams)
        temp_hparams.spec_hop_length = hparams.timbre_hop_length
        temp_hparams.spec_type = hparams.timbre_spec_type
        temp_hparams.spec_log_amplitude = hparams.timbre_spec_log_amplitude
        timbre_spec = wav_to_spec_op(audio, hparams=temp_hparams)
        if hparams.timbre_spec_log_amplitude:
            timbre_spec = timbre_spec - librosa.power_to_db(np.array([1e-9]))[0]
            timbre_spec /= K.max(timbre_spec)
        spec = (spec, timbre_spec)

    labels, label_weights, onsets, offsets, velocities = sequence_to_pianoroll_op(
        sequence, velocity_range, hparams=hparams)

    length = wav_to_num_frames_op(audio, hparams_frames_per_second(hparams))

    asserts = []
    if hparams.max_expected_train_example_len and is_training:
        asserts.append(
            tf.debugging.assert_less_equal(length, hparams.max_expected_train_example_len))

    with tf.control_dependencies(asserts):
        return InputTensors(
            spec=spec,
            labels=labels,
            length=length,
            onsets=onsets,
            offsets=offsets,
            velocities=velocities,
            note_sequence=sequence)


def input_tensors_to_example(inputs, hparams):
    """Convert InputTensors to Example proto ready for serialization."""
    del hparams

    feature = {
        'spec': tf.train.Feature(
            float_list=tf.train.FloatList(value=inputs.spec.flatten())),
        'labels': tf.train.Feature(
            float_list=tf.train.FloatList(value=inputs.labels.flatten())),
        'length': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[inputs.length])),
        'onsets': tf.train.Feature(
            float_list=tf.train.FloatList(value=inputs.onsets.flatten())),
        'offsets': tf.train.Feature(
            float_list=tf.train.FloatList(value=inputs.offsets.flatten())),
        'velocities': tf.train.Feature(
            float_list=tf.train.FloatList(value=inputs.velocities.flatten())),
        'note_sequence': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[inputs.note_sequence])),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


FeatureTensors = collections.namedtuple('FeatureTensors',
                                        ('spec',))
MultiFeatureTensors = collections.namedtuple('MultiFeatureTensors',
                                             ('spec_512', 'spec_256', 'present_instruments'))
LabelTensors = collections.namedtuple('LabelTensors',
                                      ('labels', 'onsets', 'offsets'))


def input_tensors_to_model_input(input_tensors, hparams, is_training,
                                 num_classes=constants.MIDI_PITCHES):
    """Processes an InputTensor into FeatureTensors and LabelTensors."""
    length = tf.cast(input_tensors.length, tf.int32)
    labels = input_tensors.labels
    label_weights = input_tensors.label_weights
    onsets = input_tensors.onsets
    offsets = input_tensors.offsets
    velocities = input_tensors.velocities
    spec = input_tensors.spec

    # Slice specs and labels tensors so they are no longer than truncated_length.
    hparams_truncated_length = tf.cast(
        hparams.truncated_length_secs * hparams_frames_per_second(hparams),
        tf.int32)
    if hparams.truncated_length_secs:
        truncated_length = tf.reduce_min([hparams_truncated_length, length])
    else:
        truncated_length = length

    # If max_expected_train_example_len is set, ensure that all examples are
    # padded to this length. This results in a fixed shape that can work on TPUs.
    if hparams.max_expected_train_example_len and is_training:
        # In this case, final_length is a constant.
        if hparams.truncated_length_secs:
            assert_op = tf.assert_equal(hparams.max_expected_train_example_len,
                                        hparams_truncated_length)
            with tf.control_dependencies([assert_op]):
                final_length = hparams.max_expected_train_example_len
        else:
            final_length = hparams.max_expected_train_example_len
    else:
        # In this case, it is min(hparams.truncated_length, length).
        final_length = truncated_length

    spec_delta = tf.shape(spec)[0] - final_length
    spec = tf.case(
        [(spec_delta < 0,
          lambda: tf.pad(spec, tf.stack([(0, -spec_delta), (0, 0)]))),
         (spec_delta > 0, lambda: spec[0:-spec_delta])],
        default=lambda: spec)
    labels_delta = tf.shape(labels)[0] - final_length
    padding = tf.stack([(0, -labels_delta), (0, 0), (0, 0)]) if hparams.split_pianoroll \
        else tf.stack([(0, -labels_delta), (0, 0)])
    labels = tf.case(
        [(labels_delta < 0,
          lambda: tf.pad(labels, padding)),
         (labels_delta > 0, lambda: labels[0:-labels_delta])],
        default=lambda: labels)
    label_weights = tf.case(
        [(labels_delta < 0,
          lambda: tf.pad(label_weights, padding)
          ), (labels_delta > 0, lambda: label_weights[0:-labels_delta])],
        default=lambda: label_weights)
    onsets = tf.case(
        [(labels_delta < 0,
          lambda: tf.pad(onsets, padding)),
         (labels_delta > 0, lambda: onsets[0:-labels_delta])],
        default=lambda: onsets)
    offsets = tf.case(
        [(labels_delta < 0,
          lambda: tf.pad(offsets, padding)),
         (labels_delta > 0, lambda: offsets[0:-labels_delta])],
        default=lambda: offsets)
    velocities = tf.case(
        [(labels_delta < 0,
          lambda: tf.pad(velocities, padding)),
         (labels_delta > 0, lambda: velocities[0:-labels_delta])],
        default=lambda: velocities)

    if hparams.split_pianoroll:
        spec_256 = spec[1]
        spec = spec[0]
        spec_256_length = int(
            final_length * hparams.spec_hop_length / hparams.timbre_hop_length)
        spec_256_delta = tf.shape(spec_256)[0] - spec_256_length
        spec_256 = tf.case(
            [(spec_256_delta < 0,
              lambda: tf.pad(spec_256, tf.stack([(0, -spec_256_delta), (0, 0)]))),
             (spec_256_delta > 0, lambda: spec_256[0:-spec_256_delta])],
            default=lambda: spec_256)
        features = MultiFeatureTensors(
            spec_512=tf.reshape(spec, (final_length, hparams_frame_size(hparams), 1)),
            spec_256=tf.reshape(spec_256, (spec_256_length, hparams_frame_size(hparams), 1)),
            present_instruments=get_present_instruments_op(input_tensors.note_sequence,
                                                           hparams=hparams)
        )
        labels = LabelTensors(
            labels=tf.reshape(labels, (final_length, num_classes, hparams.timbre_num_classes + 1)),
            onsets=tf.reshape(onsets, (final_length, num_classes, hparams.timbre_num_classes + 1)),
            offsets=tf.reshape(offsets,
                               (final_length, num_classes, hparams.timbre_num_classes + 1)),
        )
    else:
        features = FeatureTensors(
            spec=tf.reshape(spec, (final_length, hparams_frame_size(hparams), 1)),
        )
        labels = LabelTensors(
            labels=tf.reshape(labels, (final_length, num_classes)),
            onsets=tf.reshape(onsets, (final_length, num_classes)),
            offsets=tf.reshape(offsets, (final_length, num_classes)),
        )

    return features, labels


def generate_sharded_filenames(sharded_filenames):
    """Generate a list of filenames from a filename with an @shards suffix."""
    filenames = []
    for filename in sharded_filenames.split(','):
        match = re.match(r'^([^@]+)@(\d+)$', filename)
        if not match:
            filenames.append(filename)
        else:
            num_shards = int(match.group(2))
            base = match.group(1)
            for i in range(num_shards):
                filenames.append('{}-{:0=5d}-of-{:0=5d}'.format(base, i, num_shards))
    return filenames


def read_examples(examples, is_training, shuffle_examples,
                  skip_n_initial_records, hparams):
    """Returns a tf.data.Dataset from TFRecord files.

    Args:
      examples: A string path to a TFRecord file of examples, a python list of
        serialized examples, or a Tensor placeholder for serialized examples.
      is_training: Whether this is a training run.
      shuffle_examples: Whether examples should be shuffled.
      skip_n_initial_records: Skip this many records at first.
      hparams: HParams object specifying hyperparameters.

    Returns:
      A tf.data.Dataset.
    """
    if is_training and not shuffle_examples:
        raise ValueError('shuffle_examples must be True if is_training is True')

    if isinstance(examples, str):
        # Read examples from a TFRecord file containing serialized NoteSequence
        # and audio.
        sharded_filenames = generate_sharded_filenames(examples)
        if len(sharded_filenames) >= 1:  # TODO better solution
            # Could be a glob pattern.
            filenames = tf.data.Dataset.list_files(
                sharded_filenames, shuffle=shuffle_examples)
        else:
            # If we've already expanded the list of sharded filenames, don't send to
            # Dataset.list_files because that will attempt to execute a potentially
            # slow (especially for network filesystems) glob expansion on each entry
            # in the already expanded list.
            if shuffle_examples:
                np.random.shuffle(sharded_filenames)
            filenames = tf.data.Dataset.from_tensor_slices(sharded_filenames)

        if shuffle_examples:
            input_dataset = filenames.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset, sloppy=True, cycle_length=8))
        else:
            input_dataset = tf.data.TFRecordDataset(filenames)
    else:
        input_dataset = tf.data.Dataset.from_tensor_slices(examples)
    print(f'Examples count: {input_dataset.reduce(0, lambda x, _: x + 1).numpy()}')
    if shuffle_examples:
        input_dataset = input_dataset.shuffle(hparams.shuffle_buffer_size)
    if is_training:
        input_dataset = input_dataset.repeat()
    if skip_n_initial_records:
        input_dataset = input_dataset.skip(skip_n_initial_records)

    return input_dataset


def parse_preprocessed_example(example_proto):
    """Process an already preprocessed Example proto into input tensors."""
    features = {
        'spec': tf.io.VarLenFeature(dtype=tf.float32),
        'labels': tf.io.VarLenFeature(dtype=tf.float32),
        'length': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'onsets': tf.io.VarLenFeature(dtype=tf.float32),
        'offsets': tf.io.VarLenFeature(dtype=tf.float32),
        'velocities': tf.io.VarLenFeature(dtype=tf.float32),
        'note_sequence': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    }
    record = tf.io.parse_single_example(example_proto, features)
    input_tensors = InputTensors(
        spec=tf.sparse.to_dense(record['spec']),
        labels=tf.sparse.to_dense(record['labels']),
        length=record['length'],
        onsets=tf.sparse.to_dense(record['onsets']),
        offsets=tf.sparse.to_dense(record['offsets']),
        velocities=tf.sparse.to_dense(record['velocities']),
        note_sequence=record['note_sequence'])
    return input_tensors


def create_batch(dataset, hparams, is_training, batch_size=None):
    """Batch a dataset, optional batch_size override."""
    if not batch_size:
        batch_size = hparams.batch_size
    if hparams.max_expected_train_example_len and is_training:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    elif hparams.split_pianoroll:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                MultiFeatureTensors(
                    TensorShape([None, 229, 1]),
                    TensorShape([None, 229, 1]),
                    TensorShape([hparams.timbre_num_classes])),
                LabelTensors(
                    TensorShape([None, 88, hparams.timbre_num_classes + 1]),
                    TensorShape([None, 88, hparams.timbre_num_classes + 1]),
                    TensorShape([None, 88, hparams.timbre_num_classes + 1]),
                )),
            drop_remainder=True)
    else:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                FeatureTensors(TensorShape([None, 229, 1])),
                LabelTensors(
                    TensorShape([None, 88]),
                    TensorShape([None, 88]),
                    TensorShape([None, 88]),
                )),
            drop_remainder=True)
    return dataset


def provide_batch(examples, preprocess_examples, hparams, is_training, shuffle_examples,
                  skip_n_initial_records, **kwargs):
    """Returns batches of tensors read from TFRecord files.
    Reads from maestro-like datasets.
    :param examples: TFRecord filenames.
    :param hparams: Hyperparameters.
    :param is_training: Is this a training dataset.
    :param shuffle_examples: Randomly shuffle the examples.
    :param skip_n_initial_records: Skip n records in the dataset.
    :param kwargs: Unused.
    :return: TensorFlow batched dataset.
    """
    hparams = hparams

    input_dataset = read_examples(
        examples, is_training, shuffle_examples, skip_n_initial_records, hparams)

    if preprocess_examples:
        input_map_fn = functools.partial(
            preprocess_example, hparams=hparams, is_training=is_training)
    else:
        input_map_fn = parse_preprocessed_example

    input_tensors = input_dataset.map(input_map_fn)
    functools.partial(
        input_tensors_to_model_input,
        hparams=hparams, is_training=is_training)(next(iter(input_tensors)))

    model_input = input_tensors.map(
        functools.partial(
            input_tensors_to_model_input,
            hparams=hparams, is_training=is_training))
    model_input = model_input.filter(lambda f, l: K.sum(K.cast_to_floatx(l[0][-1])) > 0)
    dataset = create_batch(model_input, hparams=hparams, is_training=is_training)

    # Use `buffer_size=tf.data.experimental.AUTOTUNE` if possible.
    return dataset.prefetch(buffer_size=1)


def merge_data_functions(data_fn_list):
    """Merge 2 or more ready-to-use datasets.
    :param data_fn_list: Dataset function partials with
    only `examples` defined
    :return: Function that chooses between the dataset functions
    provided as input.
    """

    def concat(**kwargs):
        return tf.data.experimental.choose_from_datasets(
            [f(**kwargs) for f in data_fn_list],
            tf.data.Dataset.range(len(data_fn_list)).map(
                lambda x: tf.cast(x, tf.int64)
            ).repeat())

    return concat
