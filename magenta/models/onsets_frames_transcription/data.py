# Copyright 2019 The Magenta Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import wave

import librosa
from magenta.models.onsets_frames_transcription import audio_transform
from magenta.models.onsets_frames_transcription import constants
from magenta.music import audio_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2
import numpy as np
import six
import tensorflow as tf


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
  spec = tf.py_func(
      functools.partial(wav_to_spec, hparams=hparams),
      [wav_audio],
      tf.float32,
      name='wav_to_spec')
  spec.set_shape([None, hparams_frame_size(hparams)])
  return spec


def wav_to_num_frames(wav_audio, frames_per_second):
  """Transforms a wav-encoded audio string into number of frames."""
  w = wave.open(six.BytesIO(wav_audio))
  return np.int32(w.getnframes() / w.getframerate() * frames_per_second)


def wav_to_num_frames_op(wav_audio, frames_per_second):
  """Transforms a wav-encoded audio string into number of frames."""
  res = tf.py_func(
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

  return tf.py_func(
      transform_wav_data, [wav_data_tensor],
      tf.string,
      name='transform_wav_data_op')


def sequence_to_pianoroll_op(sequence_tensor, velocity_range_tensor, hparams):
  """Transforms a serialized NoteSequence to a pianoroll."""

  def sequence_to_pianoroll_fn(sequence_tensor, velocity_range_tensor):
    """Converts sequence to pianorolls."""
    velocity_range = music_pb2.VelocityRange.FromString(velocity_range_tensor)
    sequence = music_pb2.NoteSequence.FromString(sequence_tensor)
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
        max_velocity=velocity_range.max)
    return (roll.active, roll.weights, roll.onsets, roll.onset_velocities,
            roll.offsets)

  res, weighted_res, onsets, velocities, offsets = tf.py_func(
      sequence_to_pianoroll_fn, [sequence_tensor, velocity_range_tensor],
      [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
      name='sequence_to_pianoroll_op')
  res.set_shape([None, constants.MIDI_PITCHES])
  weighted_res.set_shape([None, constants.MIDI_PITCHES])
  onsets.set_shape([None, constants.MIDI_PITCHES])
  offsets.set_shape([None, constants.MIDI_PITCHES])
  velocities.set_shape([None, constants.MIDI_PITCHES])

  return res, weighted_res, onsets, offsets, velocities


def jitter_label_op(sequence_tensor, jitter_amount_sec):

  def jitter_label(sequence_tensor):
    sequence = music_pb2.NoteSequence.FromString(sequence_tensor)
    sequence = sequences_lib.shift_sequence_times(sequence, jitter_amount_sec)
    return sequence.SerializeToString()

  return tf.py_func(jitter_label, [sequence_tensor], tf.string)


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


def truncate_note_sequence_op(sequence_tensor, truncated_length_frames,
                              hparams):
  """Truncates a NoteSequence to the given length."""
  def truncate(sequence_tensor, num_frames):
    sequence = music_pb2.NoteSequence.FromString(sequence_tensor)
    num_secs = num_frames / hparams_frames_per_second(hparams)
    return truncate_note_sequence(sequence, num_secs).SerializeToString()
  res = tf.py_func(
      truncate,
      [sequence_tensor, truncated_length_frames],
      tf.string)
  res.set_shape(())
  return res


InputTensors = collections.namedtuple(
    'InputTensors',
    ('spec', 'labels', 'label_weights', 'length', 'onsets', 'offsets',
     'velocities', 'velocity_range', 'sequence_id', 'note_sequence'))


def _preprocess_data(sequence_id, sequence, audio, velocity_range, hparams,
                     is_training):
  """Compute spectral representation, labels, and length from sequence/audio.

  Args:
    sequence_id: id of the sequence.
    sequence: String tensor containing serialized NoteSequence proto.
    audio: String tensor containing containing WAV data.
    velocity_range: String tensor containing max and min velocities of file as
        a serialized VelocityRange.
    hparams: HParams object specifying hyperparameters.
    is_training: Whether or not this is a training run.

  Returns:
    An InputTensors tuple.

  Raises:
    ValueError: If hparams is contains an invalid spec_type.
  """

  wav_jitter_amount_ms = label_jitter_amount_ms = 0
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

  labels, label_weights, onsets, offsets, velocities = sequence_to_pianoroll_op(
      sequence, velocity_range, hparams=hparams)

  length = wav_to_num_frames_op(audio, hparams_frames_per_second(hparams))

  return InputTensors(
      spec=spec,
      labels=labels,
      label_weights=label_weights,
      length=length,
      onsets=onsets,
      offsets=offsets,
      velocities=velocities,
      velocity_range=velocity_range,
      sequence_id=sequence_id,
      note_sequence=sequence)


FeatureTensors = collections.namedtuple('FeatureTensors',
                                        ('spec', 'length', 'sequence_id'))
LabelTensors = collections.namedtuple(
    'LabelTensors', ('labels', 'label_weights', 'onsets', 'offsets',
                     'velocities', 'note_sequence'))


def _provide_data(input_tensors, truncated_length, hparams):
  """Returns tensors for reading batches from provider."""
  length = tf.to_int32(input_tensors.length)
  labels = tf.reshape(input_tensors.labels, (-1, constants.MIDI_PITCHES))
  label_weights = tf.reshape(input_tensors.label_weights,
                             (-1, constants.MIDI_PITCHES))
  onsets = tf.reshape(input_tensors.onsets, (-1, constants.MIDI_PITCHES))
  offsets = tf.reshape(input_tensors.offsets, (-1, constants.MIDI_PITCHES))
  velocities = tf.reshape(input_tensors.velocities,
                          (-1, constants.MIDI_PITCHES))
  spec = tf.reshape(input_tensors.spec, (-1, hparams_frame_size(hparams)))

  truncated_length = (
      tf.reduce_min([truncated_length, length]) if truncated_length else length)

  # Pad or slice specs and labels tensors to have the same lengths,
  # truncating after truncated_length.
  spec_delta = tf.shape(spec)[0] - truncated_length
  spec = tf.case(
      [(spec_delta < 0,
        lambda: tf.pad(spec, tf.stack([(0, -spec_delta), (0, 0)]))),
       (spec_delta > 0, lambda: spec[0:-spec_delta])],
      default=lambda: spec)
  labels_delta = tf.shape(labels)[0] - truncated_length
  labels = tf.case(
      [(labels_delta < 0,
        lambda: tf.pad(labels, tf.stack([(0, -labels_delta), (0, 0)]))),
       (labels_delta > 0, lambda: labels[0:-labels_delta])],
      default=lambda: labels)
  label_weights = tf.case(
      [(labels_delta < 0,
        lambda: tf.pad(label_weights, tf.stack([(0, -labels_delta), (0, 0)]))
       ), (labels_delta > 0, lambda: label_weights[0:-labels_delta])],
      default=lambda: label_weights)
  onsets = tf.case(
      [(labels_delta < 0,
        lambda: tf.pad(onsets, tf.stack([(0, -labels_delta), (0, 0)]))),
       (labels_delta > 0, lambda: onsets[0:-labels_delta])],
      default=lambda: onsets)
  offsets = tf.case(
      [(labels_delta < 0,
        lambda: tf.pad(offsets, tf.stack([(0, -labels_delta), (0, 0)]))),
       (labels_delta > 0, lambda: offsets[0:-labels_delta])],
      default=lambda: offsets)
  velocities = tf.case(
      [(labels_delta < 0,
        lambda: tf.pad(velocities, tf.stack([(0, -labels_delta), (0, 0)]))),
       (labels_delta > 0, lambda: velocities[0:-labels_delta])],
      default=lambda: velocities)

  truncated_note_sequence = truncate_note_sequence_op(
      input_tensors.note_sequence, truncated_length, hparams)

  return (FeatureTensors(
      spec=tf.reshape(spec, (truncated_length, hparams_frame_size(hparams), 1)),
      length=truncated_length,
      sequence_id=input_tensors.sequence_id),
          LabelTensors(
              labels=tf.reshape(labels,
                                (truncated_length, constants.MIDI_PITCHES)),
              label_weights=tf.reshape(
                  label_weights, (truncated_length, constants.MIDI_PITCHES)),
              onsets=tf.reshape(onsets,
                                (truncated_length, constants.MIDI_PITCHES)),
              offsets=tf.reshape(offsets,
                                 (truncated_length, constants.MIDI_PITCHES)),
              velocities=tf.reshape(velocities,
                                    (truncated_length, constants.MIDI_PITCHES)),
              note_sequence=truncated_note_sequence))


def provide_batch(batch_size,
                  examples,
                  hparams,
                  truncated_length=0,
                  is_training=True,
                  shuffle_buffer_size=64):
  """Returns batches of tensors read from TFRecord files.

  Args:
    batch_size: The integer number of records per batch.
    examples: A string path to a TFRecord file of examples, a python list of
      serialized examples, or a Tensor placeholder for serialized examples.
    hparams: HParams object specifying hyperparameters.
    truncated_length: An optional integer specifying whether sequences should be
      truncated this length.
    is_training: Whether this is a training run.
    shuffle_buffer_size: Buffer size used to shuffle records.

  Returns:
    Batched tensors in a TranscriptionData NamedTuple.
  """
  if isinstance(examples, str):
    # Read examples from a TFRecord file containing serialized NoteSequence
    # and audio.
    filenames = tf.data.Dataset.list_files(examples)
    input_dataset = filenames.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset, sloppy=True, cycle_length=8))
  else:
    input_dataset = tf.data.Dataset.from_tensor_slices(examples)

  def _parse(example_proto):
    features = {
        'id': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'sequence': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'audio': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'velocity_range': tf.FixedLenFeature(shape=(), dtype=tf.string),
    }
    return tf.parse_single_example(example_proto, features)

  def _preprocess(record):
    input_tensors = _preprocess_data(record['id'], record['sequence'],
                                     record['audio'], record['velocity_range'],
                                     hparams, is_training)
    return _provide_data(
        input_tensors, truncated_length=truncated_length, hparams=hparams)

  input_dataset = input_dataset.map(_parse).map(
      _preprocess, num_parallel_calls=tf.contrib.data.AUTOTUNE)

  if is_training:
    input_dataset = input_dataset.apply(
        tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size))

  # batching/padding
  dataset = input_dataset.padded_batch(
      batch_size, padded_shapes=input_dataset.output_shapes)

  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  dataset_options = tf.data.Options()
  dataset_options.experimental_autotune = True
  dataset = dataset.with_options(dataset_options)

  return dataset
