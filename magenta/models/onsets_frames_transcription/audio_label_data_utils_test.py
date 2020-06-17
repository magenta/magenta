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

"""Test for audio_label_data_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.onsets_frames_transcription import audio_label_data_utils

from note_seq import audio_io
from note_seq import constants
from note_seq import testing_lib
from note_seq.protobuf import music_pb2

import numpy as np
import tensorflow.compat.v1 as tf

SAMPLE_RATE = 16000


class SplitAudioTest(tf.test.TestCase):

  def _CreateSyntheticSequence(self):
    seq = music_pb2.NoteSequence(total_time=10)
    testing_lib.add_track_to_sequence(seq, 0, [(50, 20, 0, 5)])
    testing_lib.add_track_to_sequence(seq, 0, [(50, 80, 5, 10)])
    return seq

  def _CreateSyntheticExample(self):
    sequence = self._CreateSyntheticSequence()
    wav_samples = np.zeros(9 * SAMPLE_RATE, np.float32)
    wav_data = audio_io.samples_to_wav_data(wav_samples, SAMPLE_RATE)
    return wav_data, sequence

  def testSplitAudioLabelData(self):
    wav_data, sequence = self._CreateSyntheticExample()
    records = audio_label_data_utils.process_record(
        wav_data, sequence, 'test', sample_rate=SAMPLE_RATE)

    for record in records:
      audio = record.features.feature['audio'].bytes_list.value[0]
      velocity_range = music_pb2.VelocityRange.FromString(
          record.features.feature['velocity_range'].bytes_list.value[0])
      note_sequence = music_pb2.NoteSequence.FromString(
          record.features.feature['sequence'].bytes_list.value[0])

      expected_samples = np.zeros(10 * SAMPLE_RATE)
      np.testing.assert_array_equal(
          expected_samples,
          audio_io.wav_data_to_samples(audio, sample_rate=SAMPLE_RATE))
      self.assertEqual(velocity_range.min, 20)
      self.assertEqual(velocity_range.max, 80)
      self.assertEqual(note_sequence.notes[0].velocity, 20)
      self.assertEqual(note_sequence.notes[0].end_time, 5.)
      self.assertEqual(note_sequence.notes[1].velocity, 80)
      self.assertEqual(note_sequence.notes[1].end_time, 10.)

  def testSplitMidi(self):
    sequence = music_pb2.NoteSequence()
    sequence.notes.add(pitch=60, start_time=1.0, end_time=2.9)
    sequence.notes.add(pitch=60, start_time=8.0, end_time=11.0)
    sequence.notes.add(pitch=60, start_time=14.0, end_time=17.0)
    sequence.notes.add(pitch=60, start_time=20.0, end_time=23.0)
    sequence.total_time = 25.

    sample_rate = 160
    samples = np.zeros(sample_rate * int(sequence.total_time))
    splits = audio_label_data_utils.find_split_points(
        sequence, samples, sample_rate, 0, 3)

    self.assertEqual(splits, [0., 3., 6., 9., 12., 15., 18., 21., 24., 25.])

    samples[int(8.5 * sample_rate)] = 1
    samples[int(8.5 * sample_rate) + 1] = -1
    splits = audio_label_data_utils.find_split_points(
        sequence, samples, sample_rate, 0, 3)

    self.assertEqual(splits, [
        0.0, 3.0, 6.0, 8.50625, 11.50625, 14.50625, 17.50625, 20.50625,
        23.50625, 25.
    ])


class MixSequencesTest(tf.test.TestCase):

  def testMixSequences(self):
    sample_rate = 10

    sequence1 = music_pb2.NoteSequence()
    sequence1.notes.add(pitch=60, start_time=0.5, end_time=1.0, velocity=90)
    sequence1.notes.add(pitch=62, start_time=1.0, end_time=2.0, velocity=90)
    sequence1.total_time = 2.0

    samples1 = np.linspace(0, 1, int(sample_rate * sequence1.total_time))

    sequence2 = music_pb2.NoteSequence()
    sequence2.notes.add(pitch=64, start_time=0.5, end_time=1.0, velocity=90)
    sequence2.total_time = 1.0

    samples2 = np.linspace(0, 1, int(sample_rate * sequence2.total_time))

    mixed_samples, mixed_sequence = audio_label_data_utils.mix_sequences(
        [samples1, samples2], sample_rate, [sequence1, sequence2])

    expected_sequence = music_pb2.NoteSequence()
    expected_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    expected_sequence.notes.add(
        pitch=60, start_time=0.5, end_time=1.0, velocity=127)
    expected_sequence.notes.add(
        pitch=62, start_time=1.0, end_time=2.0, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=0.5, end_time=1.0, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=1.5, end_time=2.0, velocity=127)
    expected_sequence.total_time = 2.0

    self.assertProtoEquals(expected_sequence, mixed_sequence)

    expected_samples = np.concatenate([samples2, samples2]) * .5 + samples1 * .5
    np.testing.assert_array_equal(expected_samples, mixed_samples)

  def testMixSequencesLongerNoteSequence(self):
    sample_rate = 10

    sequence1 = music_pb2.NoteSequence()
    sequence1.notes.add(pitch=60, start_time=0.5, end_time=1.0, velocity=90)
    sequence1.notes.add(pitch=62, start_time=1.0, end_time=2.0, velocity=90)
    sequence1.total_time = 2.0

    # samples1 will be .1 seconds shorter than sequence1
    samples1 = np.linspace(0, 1, int(sample_rate * (sequence1.total_time - .1)))

    sequence2 = music_pb2.NoteSequence()
    sequence2.notes.add(pitch=64, start_time=0.5, end_time=1.0, velocity=90)
    sequence2.total_time = 1.0

    samples2 = np.linspace(0, 1, int(sample_rate * sequence2.total_time))

    mixed_samples, mixed_sequence = audio_label_data_utils.mix_sequences(
        [samples1, samples2], sample_rate, [sequence1, sequence2])

    expected_sequence = music_pb2.NoteSequence()
    expected_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    expected_sequence.notes.add(
        pitch=60, start_time=0.5, end_time=1.0, velocity=127)
    expected_sequence.notes.add(
        pitch=62, start_time=1.0, end_time=2.0, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=0.5, end_time=1.0, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=1.5, end_time=2.0, velocity=127)
    expected_sequence.total_time = 2.0

    self.assertProtoEquals(expected_sequence, mixed_sequence)

    # We expect samples1 to have 2 samples of padding and samples2 to be
    # repeated 1 time fully and once with a single sample.
    expected_samples = (
        np.concatenate([samples2, samples2, [samples2[0]]]) * .5 +
        np.concatenate([samples1, [0, 0]]) * .5)
    np.testing.assert_array_equal(expected_samples, mixed_samples)

  def testMixSequencesWithSustain(self):
    sample_rate = 10

    sequence1 = music_pb2.NoteSequence()
    sequence1.notes.add(pitch=60, start_time=0.5, end_time=0.6, velocity=90)
    sequence1.notes.add(pitch=62, start_time=1.0, end_time=2.0, velocity=90)
    sequence1.total_time = 2.0
    testing_lib.add_control_changes_to_sequence(
        sequence1, 0, [(0.0, 64, 127), (1.0, 64, 0)])

    samples1 = np.linspace(0, 1, int(sample_rate * sequence1.total_time))

    sequence2 = music_pb2.NoteSequence()
    sequence2.notes.add(pitch=64, start_time=0.5, end_time=0.6, velocity=90)
    sequence2.total_time = 1.0
    testing_lib.add_control_changes_to_sequence(
        sequence2, 0, [(0.0, 64, 127), (0.9, 64, 0)])

    samples2 = np.linspace(0, 1, int(sample_rate * sequence2.total_time))

    mixed_samples, mixed_sequence = audio_label_data_utils.mix_sequences(
        [samples1, samples2], sample_rate, [sequence1, sequence2])

    expected_sequence = music_pb2.NoteSequence()
    expected_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    expected_sequence.notes.add(
        pitch=60, start_time=0.5, end_time=1.0, velocity=127)
    expected_sequence.notes.add(
        pitch=62, start_time=1.0, end_time=2.0, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=0.5, end_time=0.9, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=1.5, end_time=1.9, velocity=127)
    expected_sequence.total_time = 2.0

    self.assertProtoEquals(expected_sequence, mixed_sequence)

    expected_samples = np.concatenate([samples2, samples2]) * .5 + samples1 * .5
    np.testing.assert_array_equal(expected_samples, mixed_samples)

  def testMixSequencesTotalTime(self):
    sample_rate = 10

    sequence1 = music_pb2.NoteSequence()
    sequence1.notes.add(pitch=60, start_time=0.5, end_time=1.0, velocity=90)
    sequence1.notes.add(pitch=62, start_time=1.0, end_time=1.5, velocity=90)
    sequence1.total_time = 1.5

    samples1 = np.linspace(0, 1, int(sample_rate * 2))

    sequence2 = music_pb2.NoteSequence()
    sequence2.notes.add(pitch=64, start_time=0.5, end_time=0.9, velocity=90)
    sequence2.total_time = 0.9

    samples2 = np.linspace(0, 1, int(sample_rate * 1))

    mixed_samples, mixed_sequence = audio_label_data_utils.mix_sequences(
        [samples1, samples2], sample_rate, [sequence1, sequence2])

    expected_sequence = music_pb2.NoteSequence()
    expected_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    expected_sequence.notes.add(
        pitch=60, start_time=0.5, end_time=1.0, velocity=127)
    expected_sequence.notes.add(
        pitch=62, start_time=1.0, end_time=1.5, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=0.5, end_time=0.9, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=1.5, end_time=1.9, velocity=127)

    # Expected time is 1.9 because the sequences are repeated according to the
    # length of their associated audio. So sequence1 is not repeated at all
    # (audio is 2 seconds) and sequence2 is repeated once after shifting all the
    # notes by the audio length of 1 second. The final total_time is left as is
    # after the last repeat, so it ends up being 1 + .9 seconds.
    expected_sequence.total_time = 1.9

    self.assertProtoEquals(expected_sequence, mixed_sequence)

    expected_samples = np.concatenate([samples2, samples2]) * .5 + samples1 * .5
    np.testing.assert_array_equal(expected_samples, mixed_samples)

  def testMixSequencesNormalize(self):
    sample_rate = 10

    sequence1 = music_pb2.NoteSequence()
    sequence1.notes.add(pitch=60, start_time=0.5, end_time=1.0, velocity=32)
    sequence1.notes.add(pitch=62, start_time=1.0, end_time=2.0, velocity=64)
    sequence1.total_time = 2.0

    samples1 = np.linspace(0, .5, int(sample_rate * sequence1.total_time))

    sequence2 = music_pb2.NoteSequence()
    sequence2.notes.add(pitch=64, start_time=0.5, end_time=1.0, velocity=90)
    sequence2.total_time = 1.0

    samples2 = np.linspace(0, 1, int(sample_rate * sequence2.total_time))

    mixed_samples, mixed_sequence = audio_label_data_utils.mix_sequences(
        [samples1, samples2], sample_rate, [sequence1, sequence2])

    expected_sequence = music_pb2.NoteSequence()
    expected_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    expected_sequence.notes.add(
        pitch=60, start_time=0.5, end_time=1.0, velocity=63)
    expected_sequence.notes.add(
        pitch=62, start_time=1.0, end_time=2.0, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=0.5, end_time=1.0, velocity=127)
    expected_sequence.notes.add(
        pitch=64, start_time=1.5, end_time=2.0, velocity=127)
    expected_sequence.total_time = 2.0

    self.assertProtoEquals(expected_sequence, mixed_sequence)

    samples1_normalized = samples1 * 2  # previous max was .5
    expected_samples = (np.concatenate([samples2, samples2]) * .5 +
                        samples1_normalized * .5)
    np.testing.assert_array_equal(expected_samples, mixed_samples)


if __name__ == '__main__':
  tf.test.main()
