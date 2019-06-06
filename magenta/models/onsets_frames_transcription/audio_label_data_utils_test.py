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

"""Test for audio_label_data_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.onsets_frames_transcription import audio_label_data_utils

from magenta.music import audio_io
from magenta.music import testing_lib
from magenta.protobuf import music_pb2

import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000


class SplitAudioTest(tf.test.TestCase):

  def _CreateSyntheticSequence(self):
    seq = music_pb2.NoteSequence(total_time=10)
    testing_lib.add_track_to_sequence(seq, 0, [(50, 20, 0, 5)])
    testing_lib.add_track_to_sequence(seq, 0, [(50, 80, 5, 10)])
    return seq

  def _CreateSyntheticExample(self):
    sequence = self._CreateSyntheticSequence()
    wav_samples = np.zeros(2 * SAMPLE_RATE, np.float32)
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

      self.assertEqual(
          np.all(
              audio_io.wav_data_to_samples(audio, sample_rate=SAMPLE_RATE) ==
              np.zeros(2 * SAMPLE_RATE)), True)
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


if __name__ == '__main__':
  tf.test.main()
