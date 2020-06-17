# Copyright 2020 The Magenta Authors.
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

import collections
import functools
import io
import re
import wave
import zlib

import librosa
from magenta.models.onsets_frames_transcription import audio_transform
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import drum_mappings
from magenta.models.onsets_frames_transcription import melspec_input
from note_seq import audio_io
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2
import numpy as np
import tensorflow.compat.v1 as tf


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
  """Return an op for converting wav audio to a spectrogram."""
  if hparams.spec_type == 'tflite_compat_mel':
    assert hparams.spec_log_amplitude
    spec = tflite_compat_mel(wav_audio, hparams=hparams)
  else:
    spec = tf.py_func(
        functools.partial(wav_to_spec, hparams=hparams),
        [wav_audio],
        tf.float32,
        name='wav_to_spec')
    spec.set_shape([None, hparams_frame_size(hparams)])
  return spec


def tflite_compat_mel(wav_audio, hparams):
  """EXPERIMENTAL: Log mel spec with ops that can be made TFLite compatible."""
  samples, decoded_sample_rate = tf.audio.decode_wav(
      wav_audio, desired_channels=1)
  samples = tf.squeeze(samples, axis=1)
  # Ensure that we decoded the samples at the expected sample rate.
  with tf.control_dependencies(
      [tf.assert_equal(decoded_sample_rate, hparams.sample_rate)]):
    return tflite_compat_mel_from_samples(samples, hparams)


def tflite_compat_mel_from_samples(samples, hparams):
  """EXPERIMENTAL: Log mel spec with ops that can be made TFLite compatible."""
  features = melspec_input.build_mel_calculation_graph(
      samples, hparams.sample_rate,
      window_length_seconds=2048 / hparams.sample_rate,
      hop_length_seconds=(
          hparams.spec_hop_length / hparams.sample_rate),
      num_mel_bins=hparams.spec_n_bins,
      lower_edge_hz=hparams.spec_fmin,
      upper_edge_hz=hparams.sample_rate / 2.0,
      frame_width=1,
      frame_hop=1,
      tflite_compatible=False)  # False here, but would be True on device.
  return tf.squeeze(features, 1)


def get_spectrogram_hash_op(spectrogram):
  """Calculate hash of the spectrogram."""
  def get_spectrogram_hash(spectrogram):
    # Compute a hash of the spectrogram, save it as an int64.
    # Uses adler because it's fast and will fit into an int (md5 is too large).
    spectrogram_serialized = io.BytesIO()
    np.save(spectrogram_serialized, spectrogram)
    spectrogram_hash = np.int64(zlib.adler32(spectrogram_serialized.getvalue()))
    spectrogram_serialized.close()
    return spectrogram_hash
  spectrogram_hash = tf.py_func(get_spectrogram_hash, [spectrogram], tf.int64,
                                name='get_spectrogram_hash')
  spectrogram_hash.set_shape([])
  return spectrogram_hash


def wav_to_num_frames(wav_audio, frames_per_second):
  """Transforms a wav-encoded audio string into number of frames."""
  w = wave.open(io.BytesIO(wav_audio))
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
    'InputTensors', ('spec', 'spectrogram_hash', 'labels', 'label_weights',
                     'length', 'onsets', 'offsets', 'velocities', 'sequence_id',
                     'note_sequence'))


def parse_example(example_proto):
  features = {
      'id': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'sequence': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'audio': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'velocity_range': tf.FixedLenFeature(shape=(), dtype=tf.string),
  }
  record = tf.parse_single_example(example_proto, features)
  return record


def preprocess_example(example_proto, hparams, is_training):
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
  record = parse_example(example_proto)
  sequence_id = record['id']
  sequence = record['sequence']
  audio = record['audio']
  velocity_range = record['velocity_range']

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
  spectrogram_hash = get_spectrogram_hash_op(spec)

  labels, label_weights, onsets, offsets, velocities = sequence_to_pianoroll_op(
      sequence, velocity_range, hparams=hparams)

  length = wav_to_num_frames_op(audio, hparams_frames_per_second(hparams))

  asserts = []
  if hparams.max_expected_train_example_len and is_training:
    asserts.append(
        tf.assert_less_equal(length, hparams.max_expected_train_example_len))

  with tf.control_dependencies(asserts):
    return InputTensors(
        spec=spec,
        spectrogram_hash=spectrogram_hash,
        labels=labels,
        label_weights=label_weights,
        length=length,
        onsets=onsets,
        offsets=offsets,
        velocities=velocities,
        sequence_id=sequence_id,
        note_sequence=sequence)


def input_tensors_to_example(inputs, hparams):
  """Convert InputTensors to Example proto ready for serialization."""
  del hparams

  feature = {
      'spec': tf.train.Feature(
          float_list=tf.train.FloatList(value=inputs.spec.flatten())),
      'spectrogram_hash': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[inputs.spectrogram_hash])),
      'labels': tf.train.Feature(
          float_list=tf.train.FloatList(value=inputs.labels.flatten())),
      'label_weights': tf.train.Feature(
          float_list=tf.train.FloatList(value=inputs.label_weights.flatten())),
      'length': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[inputs.length])),
      'onsets': tf.train.Feature(
          float_list=tf.train.FloatList(value=inputs.onsets.flatten())),
      'offsets': tf.train.Feature(
          float_list=tf.train.FloatList(value=inputs.offsets.flatten())),
      'velocities': tf.train.Feature(
          float_list=tf.train.FloatList(value=inputs.velocities.flatten())),
      'sequence_id': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[inputs.sequence_id])),
      'note_sequence': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[inputs.note_sequence])),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))


FeatureTensors = collections.namedtuple(
    'FeatureTensors', ('spec', 'length', 'sequence_id'))
LabelTensors = collections.namedtuple(
    'LabelTensors', ('labels', 'label_weights', 'onsets', 'offsets',
                     'velocities', 'note_sequence'))


def input_tensors_to_model_input(
    input_tensors, hparams, is_training, num_classes=constants.MIDI_PITCHES):
  """Processes an InputTensor into FeatureTensors and LabelTensors."""
  length = tf.cast(input_tensors.length, tf.int32)
  labels = tf.reshape(input_tensors.labels, (-1, num_classes))
  label_weights = tf.reshape(input_tensors.label_weights, (-1, num_classes))
  onsets = tf.reshape(input_tensors.onsets, (-1, num_classes))
  offsets = tf.reshape(input_tensors.offsets, (-1, num_classes))
  velocities = tf.reshape(input_tensors.velocities, (-1, num_classes))
  spec = tf.reshape(input_tensors.spec, (-1, hparams_frame_size(hparams)))

  # Slice specs and labels tensors so they are no longer than truncated_length.
  hparams_truncated_length = tf.cast(
      hparams.truncated_length_secs * hparams_frames_per_second(hparams),
      tf.int32)
  if hparams.truncated_length_secs:
    truncated_length = tf.reduce_min([hparams_truncated_length, length])
  else:
    truncated_length = length

  if is_training:
    truncated_note_sequence = tf.constant(0)
  else:
    truncated_note_sequence = truncate_note_sequence_op(
        input_tensors.note_sequence, truncated_length, hparams)

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
    # In this case, it is min(hparams.truncated_length, length)
    final_length = truncated_length

  spec_delta = tf.shape(spec)[0] - final_length
  spec = tf.case(
      [(spec_delta < 0,
        lambda: tf.pad(spec, tf.stack([(0, -spec_delta), (0, 0)]))),
       (spec_delta > 0, lambda: spec[0:-spec_delta])],
      default=lambda: spec)
  labels_delta = tf.shape(labels)[0] - final_length
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

  features = FeatureTensors(
      spec=tf.reshape(spec, (final_length, hparams_frame_size(hparams), 1)),
      length=truncated_length,
      sequence_id=tf.constant(0) if is_training else input_tensors.sequence_id)
  labels = LabelTensors(
      labels=tf.reshape(labels, (final_length, num_classes)),
      label_weights=tf.reshape(label_weights, (final_length, num_classes)),
      onsets=tf.reshape(onsets, (final_length, num_classes)),
      offsets=tf.reshape(offsets, (final_length, num_classes)),
      velocities=tf.reshape(velocities, (final_length, num_classes)),
      note_sequence=truncated_note_sequence)

  if hparams.drum_data_map:
    labels_dict = labels._asdict()
    for k in ('labels', 'onsets', 'offsets'):
      labels_dict[k] = drum_mappings.map_pianoroll(
          labels_dict[k],
          mapping_name=hparams.drum_data_map,
          reduce_mode='any',
          min_pitch=constants.MIN_MIDI_PITCH)
    for k in ('label_weights', 'velocities'):
      labels_dict[k] = drum_mappings.map_pianoroll(
          labels_dict[k],
          mapping_name=hparams.drum_data_map,
          reduce_mode='max',
          min_pitch=constants.MIN_MIDI_PITCH)
    if labels_dict['note_sequence'].dtype == tf.string:
      labels_dict['note_sequence'] = tf.py_func(
          functools.partial(
              drum_mappings.map_sequences, mapping_name=hparams.drum_data_map),
          [labels_dict['note_sequence']],
          tf.string,
          name='get_drum_sequences',
          stateful=False)
      labels_dict['note_sequence'].set_shape(())
    labels = LabelTensors(**labels_dict)

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


def sharded_tfrecord_reader(fname):
  """Generator for reading TFRecord entries across multiple shards."""
  for sfname in generate_sharded_filenames(fname):
    for r in tf.python_io.tf_record_iterator(sfname):
      yield  r


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
    if len(sharded_filenames) == 1:
      # Could be a glob pattern.
      filenames = tf.data.Dataset.list_files(
          generate_sharded_filenames(examples), shuffle=shuffle_examples)
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
      'spec': tf.VarLenFeature(dtype=tf.float32),
      'spectrogram_hash': tf.FixedLenFeature(shape=(), dtype=tf.int64),
      'labels': tf.VarLenFeature(dtype=tf.float32),
      'label_weights': tf.VarLenFeature(dtype=tf.float32),
      'length': tf.FixedLenFeature(shape=(), dtype=tf.int64),
      'onsets': tf.VarLenFeature(dtype=tf.float32),
      'offsets': tf.VarLenFeature(dtype=tf.float32),
      'velocities': tf.VarLenFeature(dtype=tf.float32),
      'sequence_id': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'note_sequence': tf.FixedLenFeature(shape=(), dtype=tf.string),
  }
  record = tf.parse_single_example(example_proto, features)
  input_tensors = InputTensors(
      spec=tf.sparse.to_dense(record['spec']),
      spectrogram_hash=record['spectrogram_hash'],
      labels=tf.sparse.to_dense(record['labels']),
      label_weights=tf.sparse.to_dense(record['label_weights']),
      length=record['length'],
      onsets=tf.sparse.to_dense(record['onsets']),
      offsets=tf.sparse.to_dense(record['offsets']),
      velocities=tf.sparse.to_dense(record['velocities']),
      sequence_id=record['sequence_id'],
      note_sequence=record['note_sequence'])
  return input_tensors


def create_batch(dataset, hparams, is_training, batch_size=None):
  """Batch a dataset, optional batch_size override."""
  if not batch_size:
    batch_size = hparams.batch_size
  if hparams.max_expected_train_example_len and is_training:
    dataset = dataset.batch(batch_size, drop_remainder=True)
  else:
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=dataset.output_shapes,
        drop_remainder=True)
  return dataset


def combine_tensor_batch(tensor, lengths, max_length, batch_size):
  """Combine a batch of variable-length tensors into a single tensor."""
  combined = tf.concat([tensor[i, :lengths[i]] for i in range(batch_size)],
                       axis=0)
  final_length = max_length * batch_size
  combined_padded = tf.pad(combined,
                           [(0, final_length - tf.shape(combined)[0])] +
                           [(0, 0)] * (combined.shape.rank - 1))
  combined_padded.set_shape([final_length] + combined_padded.shape[1:])
  return combined_padded


def splice_examples(dataset, hparams, is_training):
  """Splice together several examples into a single example."""
  if (not is_training) or hparams.splice_n_examples == 0:
    return dataset
  else:
    dataset = dataset.padded_batch(
        hparams.splice_n_examples, padded_shapes=dataset.output_shapes)

    def _splice(features, labels):
      """Splice together a batch of examples into a single example."""
      combine = functools.partial(
          combine_tensor_batch,
          lengths=features.length,
          max_length=hparams.max_expected_train_example_len,
          batch_size=hparams.splice_n_examples)

      combined_features = FeatureTensors(
          spec=combine(features.spec),
          length=tf.reduce_sum(features.length),
          sequence_id=tf.constant(0))
      combined_labels = LabelTensors(
          labels=combine(labels.labels),
          label_weights=combine(labels.label_weights),
          onsets=combine(labels.onsets),
          offsets=combine(labels.offsets),
          velocities=combine(labels.velocities),
          note_sequence=tf.constant(0))
      return combined_features, combined_labels

    combined_dataset = dataset.map(_splice)
    return combined_dataset


def provide_batch(examples,
                  preprocess_examples,
                  params,
                  is_training,
                  shuffle_examples,
                  skip_n_initial_records):
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

  if preprocess_examples:
    input_map_fn = functools.partial(
        preprocess_example, hparams=hparams, is_training=is_training)
  else:
    input_map_fn = parse_preprocessed_example
  input_tensors = input_dataset.map(input_map_fn)

  model_input = input_tensors.map(
      functools.partial(
          input_tensors_to_model_input,
          hparams=hparams, is_training=is_training))

  model_input = splice_examples(model_input, hparams, is_training)
  dataset = create_batch(model_input, hparams=hparams, is_training=is_training)
  return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
