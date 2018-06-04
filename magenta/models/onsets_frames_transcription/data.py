# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
import math
import os
import wave

# internal imports

from . import constants

import librosa
import numpy as np
import six
import tensorflow as tf
import tensorflow.contrib.slim as slim

import magenta.music as mm
from magenta.music import audio_io
from magenta.music import constants as mm_constants
from magenta.protobuf import music_pb2

BATCH_QUEUE_CAPACITY_SEQUENCES = 50

# This is the number of threads that take the records from the reader(s),
# load the audio, create the spectrograms, and put them into the batch queue.
NUM_BATCH_THREADS = 8


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
          bins_per_octave=hparams.cqt_bins_per_octave,
          real=False),
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
      n_mels=hparams.spec_n_bins).astype(np.float32)

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


def preprocess_sequence(sequence_tensor):
  """Preprocess a NoteSequence for training.

  Deserialize and apply sustain control changes.

  Args:
    sequence_tensor: The NoteSequence in serialized form.

  Returns:
    sequence: The preprocessed NoteSequence object.
  """
  sequence = music_pb2.NoteSequence.FromString(sequence_tensor)
  sequence = mm.apply_sustain_control_changes(sequence)

  return sequence


def transform_wav_data_op(wav_data_tensor, hparams, is_training,
                          jitter_amount_sec):
  """Transforms wav data."""
  def transform_wav_data(wav_data):
    """Transforms wav data."""
    # Only do audio transformations during training.
    if is_training:
      wav_data = audio_io.jitter_wav_data(wav_data, hparams.sample_rate,
                                          jitter_amount_sec)

    # Normalize.
    if hparams.normalize_audio:
      wav_data = audio_io.normalize_wav_data(wav_data, hparams.sample_rate)

    return [wav_data]

  return tf.py_func(
      transform_wav_data,
      [wav_data_tensor],
      tf.string,
      name='transform_wav_data_op')


def sequence_to_pianoroll(sequence,
                          frames_per_second,
                          min_pitch,
                          max_pitch,
                          onset_upweight=5.0,
                          onset_window=1,
                          onset_length_ms=0,
                          onset_mode='window',
                          onset_delay_ms=0.0,
                          min_frame_occupancy_for_label=0.0,
                          # pylint: disable=unused-argument
                          min_velocity=mm_constants.MIN_MIDI_VELOCITY,
                          # pylint: enable=unused-argument
                          max_velocity=mm_constants.MAX_MIDI_VELOCITY):
  """Transforms a NoteSequence to a pianoroll assuming a single instrument.

  This function uses floating point internally and may return different results
  on different platforms or with different compiler settings or with
  different compilers.

  Args:
    sequence: The NoteSequence to convert.
    frames_per_second: How many frames per second.
    min_pitch: pitches in the sequence below this will be ignored.
    max_pitch: pitches in the sequence above this will be ignored.
    onset_upweight: Factor by which to increase the weight assigned to onsets.
    onset_window: Fixed window size for labeling offsets. Used only if
        onset_mode is 'window'.
    onset_length_ms: Length in milliseconds for the onset. Used only if
        onset_mode is 'length_ms'.
    onset_mode: Either 'window', to use onset_window, or 'length_ms' to use
        onset_length_ms.
    onset_delay_ms: Number of milliseconds to delay the onset. Can be negative.
    min_frame_occupancy_for_label: floating point value in range [0, 1] a note
        must occupy at least this percentage of a frame, for the frame to be
        given a label with the note.
    min_velocity: minimum velocity for the track, currently unused.
    max_velocity: maximum velocity for the track, not just the local sequence,
        used to globally normalize the velocities between [0, 1].

  Raises:
    ValueError: When an unknown onset_mode is supplied.

  Returns:
    roll: A pianoroll as a 2D array.
    roll_weights: Weights to be used when calculating loss against roll.
    onsets: An onset-only pianoroll as a 2D array.
    velocities: Velocities of onsets scaled from [0, 1].
  """
  roll = np.zeros(
      (int(sequence.total_time * frames_per_second + 1),
       max_pitch - min_pitch + 1),
      dtype=np.float32)

  roll_weights = np.ones_like(roll)

  onsets = np.zeros_like(roll)

  def frames_from_times(start_time, end_time):
    """Converts start/end times to start/end frames."""
    # Will round down because note may start or end in the middle of the frame.
    start_frame = int(start_time * frames_per_second)
    start_frame_occupancy = (
        start_frame + 1 - start_time * frames_per_second)
    # check for > 0.0 to avoid possible numerical issues
    if (min_frame_occupancy_for_label > 0.0 and
        start_frame_occupancy < min_frame_occupancy_for_label):
      start_frame += 1

    end_frame = int(math.ceil(end_time * frames_per_second))
    end_frame_occupancy = end_time * frames_per_second - start_frame - 1
    if (min_frame_occupancy_for_label > 0.0 and
        end_frame_occupancy < min_frame_occupancy_for_label):
      end_frame -= 1
      # can be a problem for very short notes
      end_frame = max(start_frame, end_frame)

    return start_frame, end_frame

  velocities = np.zeros_like(roll, dtype=np.float32)

  for note in sorted(sequence.notes, key=lambda n: n.start_time):
    if note.pitch < min_pitch or note.pitch > max_pitch:
      tf.logging.warning('Skipping out of range pitch: %d', note.pitch)
      continue
    start_frame, end_frame = frames_from_times(note.start_time, note.end_time)

    roll[start_frame:end_frame, note.pitch - min_pitch] = 1.0

    # label onset events. Use a window size of onset_window to account of
    # rounding issue in the start_frame computation.
    onset_start_time = note.start_time + onset_delay_ms / 1000.
    onset_end_time = note.end_time + onset_delay_ms / 1000.
    if onset_mode == 'window':
      onset_start_frame_without_window, _ = frames_from_times(
          onset_start_time, onset_end_time)

      onset_start_frame = max(
          0, onset_start_frame_without_window - onset_window)
      onset_end_frame = min(onsets.shape[0],
                            onset_start_frame_without_window + onset_window + 1)
    elif onset_mode == 'length_ms':
      onset_end_time = min(onset_end_time,
                           onset_start_time + onset_length_ms / 1000.)
      onset_start_frame, onset_end_frame = frames_from_times(
          onset_start_time, onset_end_time)
    else:
      raise ValueError('Unknown onset mode: {}'.format(onset_mode))
    onsets[onset_start_frame:onset_end_frame, note.pitch - min_pitch] = 1.0

    if note.velocity > max_velocity:
      raise ValueError('Note velocity exceeds max velocity: %d > %d' %
                       (note.velocity, max_velocity))
    velocities[onset_start_frame:onset_end_frame,
               note.pitch - min_pitch] = float(note.velocity) / max_velocity

    roll_weights[onset_start_frame:onset_end_frame, note.pitch - min_pitch] = (
        onset_upweight)
    roll_weights[onset_end_frame:end_frame, note.pitch - min_pitch] = [
        onset_upweight / x for x in range(1, end_frame - onset_end_frame + 1)
    ]

  return roll, roll_weights, onsets, velocities


def sequence_to_pianoroll_op(sequence_tensor, velocity_range_tensor, hparams):
  """Transforms a serialized NoteSequence to a pianoroll."""
  def sequence_to_pianoroll_fn(sequence_tensor, velocity_range_tensor):
    velocity_range = music_pb2.VelocityRange.FromString(velocity_range_tensor)
    sequence = preprocess_sequence(sequence_tensor)
    return sequence_to_pianoroll(
        sequence,
        frames_per_second=hparams_frames_per_second(hparams),
        min_pitch=constants.MIN_MIDI_PITCH,
        max_pitch=constants.MAX_MIDI_PITCH,
        min_frame_occupancy_for_label=hparams.min_frame_occupancy_for_label,
        onset_mode=hparams.onset_mode, onset_length_ms=hparams.onset_length,
        onset_delay_ms=hparams.onset_delay,
        min_velocity=velocity_range.min,
        max_velocity=velocity_range.max)

  res, weighted_res, onsets, velocities = tf.py_func(
      sequence_to_pianoroll_fn, [sequence_tensor, velocity_range_tensor],
      [tf.float32, tf.float32, tf.float32, tf.float32],
      name='sequence_to_pianoroll_op')
  res.set_shape([None, constants.MIDI_PITCHES])
  weighted_res.set_shape([None, constants.MIDI_PITCHES])
  onsets.set_shape([None, constants.MIDI_PITCHES])
  velocities.set_shape([None, constants.MIDI_PITCHES])

  return res, weighted_res, onsets, velocities


def jitter_label_op(sequence_tensor, jitter_amount_sec):

  def jitter_label(sequence_tensor):
    sequence = music_pb2.NoteSequence.FromString(sequence_tensor)
    sequence, _ = mm.sequences_lib.shift_sequence_times(
        sequence, jitter_amount_sec)
    return sequence.SerializeToString()

  return tf.py_func(jitter_label, [sequence_tensor], tf.string)


def truncate_note_sequence(sequence, truncate_secs):
  """Truncates a NoteSequence to the given length."""
  sus_sequence = mm.apply_sustain_control_changes(sequence)

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
    ('spec', 'labels', 'label_weights', 'length', 'onsets', 'velocities',
     'velocity_range', 'filename', 'note_sequence'))


def _preprocess_data(sequence, audio, velocity_range, hparams, is_training):
  """Compute spectral representation, labels, and length from sequence/audio.

  Args:
    sequence: String tensor containing serialized NoteSequence proto.
    audio: String tensor WAV data.
    velocity_range: String tensor containing max and min velocities of file as
        a serialized VelocityRange.
    hparams: HParams object specifying hyperparameters.
    is_training: Whether or not this is a training run.

  Returns:
    A 3-tuple of tensors containing CQT, pianoroll labels, and number of frames
    respectively.

  Raises:
    ValueError: If hparams is contains an invalid spec_type.
  """

  wav_jitter_amount_ms = label_jitter_amount_ms = 0
  # if there is combined jitter, we must generate it once here
  if hparams.jitter_amount_ms > 0:
    if hparams.jitter_wav_and_label_separately:
      wav_jitter_amount_ms = np.random.choice(hparams.jitter_amount_ms, size=1)
      label_jitter_amount_ms = np.random.choice(
          hparams.jitter_amount_ms, size=1)
    else:
      wav_jitter_amount_ms = np.random.choice(hparams.jitter_amount_ms, size=1)
      label_jitter_amount_ms = wav_jitter_amount_ms

  if label_jitter_amount_ms > 0:
    sequence = jitter_label_op(sequence, label_jitter_amount_ms / 1000.)

  transformed_wav = transform_wav_data_op(
      audio,
      hparams=hparams,
      is_training=is_training,
      jitter_amount_sec=wav_jitter_amount_ms / 1000.)

  spec = wav_to_spec_op(transformed_wav, hparams=hparams)

  labels, label_weights, onsets, velocities = sequence_to_pianoroll_op(
      sequence, velocity_range, hparams=hparams)

  length = wav_to_num_frames_op(
      transformed_wav, hparams_frames_per_second(hparams))

  return spec, labels, label_weights, length, onsets, velocities, velocity_range


def _get_input_tensors_from_examples_list(examples_list, is_training):
  """Get input tensors from a list or placeholder of examples."""
  num_examples = 0
  if not is_training and not isinstance(examples_list, tf.Tensor):
    num_examples = len(examples_list)
  return tf.data.Dataset.from_tensor_slices(examples_list), num_examples


def _get_input_tensors_from_tfrecord(files, is_training):
  """Creates a Dataset to read transcription data from TFRecord."""
  # Iterate through all data to determine how many records there are.
  tf.logging.info('Finding number of examples in %s', files)
  data_files = slim.parallel_reader.get_data_files(files)
  num_examples = 0
  if not is_training:
    for filename in data_files:
      for _ in tf.python_io.tf_record_iterator(filename):
        num_examples += 1

  tf.logging.info('Found %d examples in %s', num_examples, files)

  return tf.data.TFRecordDataset(files), num_examples


class TranscriptionData(dict):
  """A dictionary with attribute access to keys for storing input Tensors."""

  def __init__(self, *args, **kwargs):
    super(TranscriptionData, self).__init__(*args, **kwargs)
    self.__dict__ = self


def _provide_data(input_tensors, truncated_length, hparams):
  """Returns tensors for reading batches from provider."""
  (spec, labels, label_weights, length, onsets, velocities,
   unused_velocity_range, filename, note_sequence) = input_tensors

  length = tf.to_int32(length)
  labels = tf.reshape(labels, (-1, constants.MIDI_PITCHES))
  label_weights = tf.reshape(label_weights, (-1, constants.MIDI_PITCHES))
  onsets = tf.reshape(onsets, (-1, constants.MIDI_PITCHES))
  velocities = tf.reshape(velocities, (-1, constants.MIDI_PITCHES))
  spec = tf.reshape(spec, (-1, hparams_frame_size(hparams)))

  truncated_length = (tf.reduce_min([truncated_length, length])
                      if truncated_length else length)

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
  velocities = tf.case(
      [(labels_delta < 0,
        lambda: tf.pad(velocities, tf.stack([(0, -labels_delta), (0, 0)]))),
       (labels_delta > 0, lambda: velocities[0:-labels_delta])],
      default=lambda: velocities)

  truncated_note_sequence = truncate_note_sequence_op(
      note_sequence, truncated_length, hparams)

  batch_tensors = {
      'spec': tf.reshape(
          spec, (truncated_length, hparams_frame_size(hparams), 1)),
      'labels': tf.reshape(labels, (truncated_length, constants.MIDI_PITCHES)),
      'label_weights': tf.reshape(
          label_weights, (truncated_length, constants.MIDI_PITCHES)),
      'lengths': truncated_length,
      'onsets': tf.reshape(onsets, (truncated_length, constants.MIDI_PITCHES)),
      'velocities':
          tf.reshape(velocities, (truncated_length, constants.MIDI_PITCHES)),
      'filenames': filename,
      'note_sequences': truncated_note_sequence,
  }

  return batch_tensors


def provide_batch(batch_size,
                  examples,
                  hparams,
                  truncated_length=0,
                  is_training=True):
  """Returns batches of tensors read from TFRecord files.

  Args:
    batch_size: The integer number of records per batch.
    examples: A string path to a TFRecord file of examples, a python list
      of serialized examples, or a Tensor placeholder for serialized examples.
    hparams: HParams object specifying hyperparameters.
    truncated_length: An optional integer specifying whether sequences should be
      truncated this length before (optionally) being split.
    is_training: Whether this is a training run.

  Returns:
    Batched tensors in a TranscriptionData NamedTuple.
  """
  # Do data pre-processing on the CPU instead of the GPU.
  with tf.device('/cpu:0'):
    if isinstance(examples, str):
      # Read examples from a TFRecord file containing serialized NoteSequence
      # and audio.
      files = tf.gfile.Glob(os.path.expanduser(examples))
      input_dataset, num_samples = _get_input_tensors_from_tfrecord(
          files, is_training)
    else:
      input_dataset, num_samples = _get_input_tensors_from_examples_list(
          examples, is_training)

    def _parse(example_proto):
      features = {
          'id': tf.FixedLenFeature(shape=(), dtype=tf.string),
          'sequence': tf.FixedLenFeature(shape=(), dtype=tf.string),
          'audio': tf.FixedLenFeature(shape=(), dtype=tf.string),
          'velocity_range': tf.FixedLenFeature(shape=(), dtype=tf.string),
      }
      return tf.parse_single_example(example_proto, features)

    def _preprocess(record):
      (spec, labels, label_weights, length, onsets, velocities,
       velocity_range) = _preprocess_data(
           record['sequence'], record['audio'], record['velocity_range'],
           hparams, is_training)
      return InputTensors(
          spec=spec,
          labels=labels,
          label_weights=label_weights,
          length=length,
          onsets=onsets,
          velocities=velocities,
          velocity_range=velocity_range,
          filename=record['id'],
          note_sequence=record['sequence'])

    input_dataset = input_dataset.map(_parse).map(
        _preprocess, num_parallel_calls=NUM_BATCH_THREADS)

    if is_training:
      input_dataset = input_dataset.repeat()

    batch_queue_capacity = BATCH_QUEUE_CAPACITY_SEQUENCES

    dataset = input_dataset.map(
        functools.partial(
            _provide_data, truncated_length=truncated_length, hparams=hparams),
        num_parallel_calls=NUM_BATCH_THREADS)

    if is_training:
      dataset = dataset.shuffle(buffer_size=batch_queue_capacity // 10)

    # batching/padding
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=dataset.output_shapes)

    # Round down because allow_smaller_final_batch=False.
    num_batches = None
    if num_samples:
      num_batches = num_samples // batch_size

    if not is_training and num_batches is not None:
      # Emulate behavior of train.batch with allow_smaller_final_batch=False.
      dataset = dataset.take(num_batches)

    dataset = dataset.prefetch(batch_queue_capacity)

    if isinstance(examples, tf.Tensor):
      iterator = dataset.make_initializable_iterator()
    else:
      iterator = dataset.make_one_shot_iterator()

    data = TranscriptionData(iterator.get_next())
    data['max_length'] = tf.reduce_max(data['lengths'])
    if num_batches:
      data['num_batches'] = num_batches

    return data, iterator
