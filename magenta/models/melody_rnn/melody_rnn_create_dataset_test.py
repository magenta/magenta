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

"""Tests for melody_rnn_create_dataset."""

import magenta
from magenta.models.melody_rnn import melody_rnn_model
from magenta.models.melody_rnn import melody_rnn_pipeline
from magenta.pipelines import melody_pipelines
from magenta.pipelines import note_sequence_pipelines
from magenta.protobuf import music_pb2
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class MelodyRNNPipelineTest(tf.test.TestCase):

  def setUp(self):
    self.config = melody_rnn_model.MelodyRnnConfig(
        None,
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.MelodyOneHotEncoding(0, 127)),
        tf.contrib.training.HParams(),
        min_note=0,
        max_note=127,
        transpose_to_key=0)

  def testMelodyRNNPipeline(self):
    note_sequence = magenta.common.testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 120}""")
    magenta.music.testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 0.00, 2.0), (11, 55, 2.1, 5.0), (40, 45, 5.1, 8.0),
         (55, 120, 8.1, 11.0), (53, 99, 11.1, 14.1)])

    quantizer = note_sequence_pipelines.Quantizer(steps_per_quarter=4)
    melody_extractor = melody_pipelines.MelodyExtractor(
        min_bars=7, min_unique_pitches=5, gap_bars=1.0,
        ignore_polyphonic_notes=False)
    one_hot_encoding = magenta.music.OneHotEventSequenceEncoderDecoder(
        magenta.music.MelodyOneHotEncoding(
            self.config.min_note, self.config.max_note))
    quantized = quantizer.transform(note_sequence)[0]
    melody = melody_extractor.transform(quantized)[0]
    melody.squash(
        self.config.min_note,
        self.config.max_note,
        self.config.transpose_to_key)
    one_hot = one_hot_encoding.encode(melody)
    expected_result = {'training_melodies': [one_hot], 'eval_melodies': []}

    pipeline_inst = melody_rnn_pipeline.get_pipeline(
        self.config, eval_ratio=0.0)
    result = pipeline_inst.transform(note_sequence)
    self.assertEqual(expected_result, result)


if __name__ == '__main__':
  tf.test.main()
