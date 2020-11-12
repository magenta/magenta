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

"""Tests for shared data lib."""
import copy
import tempfile
import time

from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data

from note_seq import audio_io
from note_seq import sequences_lib
from note_seq import testing_lib
from note_seq.protobuf import music_pb2

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class DataTest(tf.test.TestCase):

  def _FillExample(self, sequence, wav_data, filename):
    velocity_range = music_pb2.VelocityRange(min=0, max=127)
    feature_dict = {
        'id':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])
            ),
        'sequence':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[sequence.SerializeToString()])),
        'audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[wav_data])),
        'velocity_range':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[velocity_range.SerializeToString()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

  def _DataToInputs(self, spec, labels, weighted_labels, length, filename,
                    truncated_length):
    del weighted_labels
    # This method re-implements a portion of the TensorFlow graph using numpy.
    # While typically it is frowned upon to test complicated code with other
    # code, there is no way around this for testing the pipeline end to end,
    # which requires an actual spec computation. Furthermore, much of the
    # complexity of the pipeline is due to the TensorFlow implementation,
    # so comparing it against simpler numpy code still provides effective
    # coverage.
    truncated_length = (
        min(truncated_length, length) if truncated_length else length)

    # Pad or slice spec if differs from truncated_length.
    if len(spec) < truncated_length:
      pad_amt = truncated_length - len(spec)
      spec = np.pad(spec, [(0, pad_amt), (0, 0)], 'constant')
    else:
      spec = spec[0:truncated_length]

    # Pad or slice labels if differs from truncated_length.
    if len(labels) < truncated_length:
      pad_amt = truncated_length - len(labels)
      labels = np.pad(labels, [(0, pad_amt), (0, 0)], 'constant')
    else:
      labels = labels[0:truncated_length]

    inputs = [(spec, labels, truncated_length, filename)]

    return inputs

  def _ExampleToInputs(self,
                       ex,
                       truncated_length=0):
    hparams = copy.deepcopy(configs.DEFAULT_HPARAMS)

    filename = ex.features.feature['id'].bytes_list.value[0]
    sequence = music_pb2.NoteSequence.FromString(
        ex.features.feature['sequence'].bytes_list.value[0])
    wav_data = ex.features.feature['audio'].bytes_list.value[0]

    spec = data.wav_to_spec(wav_data, hparams=hparams)
    roll = sequences_lib.sequence_to_pianoroll(
        sequence,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_pitch=constants.MIN_MIDI_PITCH,
        max_pitch=constants.MAX_MIDI_PITCH,
        min_frame_occupancy_for_label=0.0,
        onset_mode='length_ms',
        onset_length_ms=32.,
        onset_delay_ms=0.)
    length = data.wav_to_num_frames(
        wav_data, frames_per_second=data.hparams_frames_per_second(hparams))

    return self._DataToInputs(spec, roll.active, roll.weights, length, filename,
                              truncated_length)

  def _ValidateProvideBatch(self,
                            examples,
                            truncated_length,
                            batch_size,
                            expected_inputs,
                            feed_dict=None):
    """Tests for correctness of batches."""
    hparams = copy.deepcopy(configs.DEFAULT_HPARAMS)
    hparams.batch_size = batch_size
    hparams.truncated_length_secs = (
        truncated_length / data.hparams_frames_per_second(hparams))

    with self.test_session() as sess:
      dataset = data.provide_batch(
          examples=examples,
          preprocess_examples=True,
          params=hparams,
          is_training=False,
          shuffle_examples=False,
          skip_n_initial_records=0)
      iterator = tf.data.make_initializable_iterator(dataset)
      next_record = iterator.get_next()
      sess.run([
          tf.initializers.local_variables(),
          tf.initializers.global_variables(),
          iterator.initializer
      ], feed_dict=feed_dict)
      for i in range(0, len(expected_inputs), batch_size):
        # Wait to ensure example is pre-processed.
        time.sleep(0.1)
        features, labels = sess.run(next_record)
        inputs = [
            features.spec, labels.labels, features.length, features.sequence_id]
        max_length = np.max(inputs[2])
        for j in range(batch_size):
          # Add batch padding if needed.
          input_length = expected_inputs[i + j][2]
          if input_length < max_length:
            expected_inputs[i + j] = list(expected_inputs[i + j])
            pad_amt = max_length - input_length
            expected_inputs[i + j][0] = np.pad(
                expected_inputs[i + j][0], [(0, pad_amt), (0, 0)], 'constant')
            expected_inputs[i + j][1] = np.pad(
                expected_inputs[i + j][1],
                [(0, pad_amt), (0, 0)], 'constant')
          for exp_input, input_ in zip(expected_inputs[i + j], inputs):
            self.assertAllEqual(np.squeeze(exp_input), np.squeeze(input_[j]))

      with self.assertRaisesOpError('End of sequence'):
        _ = sess.run(next_record)

  def _SyntheticSequence(self, duration, note):
    seq = music_pb2.NoteSequence(total_time=duration)
    testing_lib.add_track_to_sequence(
        seq, 0, [(note, 100, 0, duration)])
    return seq

  def _CreateExamplesAndExpectedInputs(self,
                                       truncated_length,
                                       lengths,
                                       expected_num_inputs):
    hparams = copy.deepcopy(configs.DEFAULT_HPARAMS)
    examples = []
    expected_inputs = []

    for i, length in enumerate(lengths):
      wav_samples = np.zeros(
          (np.int((length / data.hparams_frames_per_second(hparams)) *
                  hparams.sample_rate), 1), np.float32)
      wav_data = audio_io.samples_to_wav_data(wav_samples, hparams.sample_rate)

      num_frames = data.wav_to_num_frames(
          wav_data, frames_per_second=data.hparams_frames_per_second(hparams))

      seq = self._SyntheticSequence(
          num_frames / data.hparams_frames_per_second(hparams),
          i + constants.MIN_MIDI_PITCH)

      examples.append(self._FillExample(seq, wav_data, 'ex%d' % i))
      expected_inputs += self._ExampleToInputs(
          examples[-1],
          truncated_length)
    self.assertEqual(expected_num_inputs, len(expected_inputs))
    return examples, expected_inputs

  def _ValidateProvideBatchTFRecord(self,
                                    truncated_length,
                                    batch_size,
                                    lengths,
                                    expected_num_inputs):
    examples, expected_inputs = self._CreateExamplesAndExpectedInputs(
        truncated_length, lengths, expected_num_inputs)

    with tempfile.NamedTemporaryFile() as temp_tfr:
      with tf.python_io.TFRecordWriter(temp_tfr.name) as writer:
        for ex in examples:
          writer.write(ex.SerializeToString())

      self._ValidateProvideBatch(
          temp_tfr.name,
          truncated_length,
          batch_size,
          expected_inputs)

  def _ValidateProvideBatchMemory(self,
                                  truncated_length,
                                  batch_size,
                                  lengths,
                                  expected_num_inputs):
    examples, expected_inputs = self._CreateExamplesAndExpectedInputs(
        truncated_length, lengths, expected_num_inputs)

    self._ValidateProvideBatch(
        [e.SerializeToString() for e in examples],
        truncated_length,
        batch_size,
        expected_inputs)

  def _ValidateProvideBatchPlaceholder(self,
                                       truncated_length,
                                       batch_size,
                                       lengths,
                                       expected_num_inputs):
    examples, expected_inputs = self._CreateExamplesAndExpectedInputs(
        truncated_length, lengths, expected_num_inputs)
    examples_ph = tf.placeholder(tf.string, [None])
    feed_dict = {examples_ph: [e.SerializeToString() for e in examples]}

    self._ValidateProvideBatch(
        examples_ph,
        truncated_length,
        batch_size,
        expected_inputs,
        feed_dict=feed_dict)

  def _ValidateProvideBatchBoth(self,
                                truncated_length,
                                batch_size,
                                lengths,
                                expected_num_inputs):
    self._ValidateProvideBatchTFRecord(
        truncated_length=truncated_length,
        batch_size=batch_size,
        lengths=lengths,
        expected_num_inputs=expected_num_inputs)
    self._ValidateProvideBatchMemory(
        truncated_length=truncated_length,
        batch_size=batch_size,
        lengths=lengths,
        expected_num_inputs=expected_num_inputs)
    self._ValidateProvideBatchPlaceholder(
        truncated_length=truncated_length,
        batch_size=batch_size,
        lengths=lengths,
        expected_num_inputs=expected_num_inputs)

  def testProvideBatchFullSeqs(self):
    self._ValidateProvideBatchBoth(
        truncated_length=0,
        batch_size=2,
        lengths=[10, 50, 100, 10, 50, 80],
        expected_num_inputs=6)

  def testProvideBatchTruncated(self):
    self._ValidateProvideBatchBoth(
        truncated_length=15,
        batch_size=2,
        lengths=[10, 50, 100, 10, 50, 80],
        expected_num_inputs=6)

  def testGeneratedShardedFilenamesCommaWithShard(self):
    filenames = data.generate_sharded_filenames('/foo/bar@3,/baz/qux@2')
    self.assertEqual(
        [
            '/foo/bar-00000-of-00003',
            '/foo/bar-00001-of-00003',
            '/foo/bar-00002-of-00003',
            '/baz/qux-00000-of-00002',
            '/baz/qux-00001-of-00002',
        ],
        filenames)

  def testGeneratedShardedFilenamesCommaWithoutShard(self):
    filenames = data.generate_sharded_filenames('/foo/bar,/baz/qux')
    self.assertEqual(
        [
            '/foo/bar',
            '/baz/qux',
        ],
        filenames)

  def testCombineTensorBatch(self):
    with tf.Graph().as_default():
      tensor = tf.constant([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
      lengths = tf.constant([3, 2])
      combined = data.combine_tensor_batch(
          tensor, lengths, max_length=5, batch_size=2)
      sess = tf.Session()
      np.testing.assert_equal([1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                              sess.run(combined))


if __name__ == '__main__':
  tf.test.main()
