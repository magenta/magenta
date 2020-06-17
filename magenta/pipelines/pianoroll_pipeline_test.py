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

"""Tests for pianoroll_pipeline."""

from absl.testing import absltest
from magenta.pipelines import pianoroll_pipeline
from note_seq import sequences_lib
from note_seq import testing_lib as music_testing_lib
from note_seq.protobuf import music_pb2


class PianorollPipelineTest(absltest.TestCase):

  def setUp(self):
    super(PianorollPipelineTest, self).setUp()
    self.note_sequence = music_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        tempos: {
          qpm: 60
        }
        ticks_per_quarter: 220
        """)

  def testExtractPianorollSequences(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = pianoroll_pipeline.extract_pianoroll_sequences(
        quantized_sequence)
    self.assertLen(seqs, 1)

    seqs, _ = pianoroll_pipeline.extract_pianoroll_sequences(
        quantized_sequence, min_steps_discard=2, max_steps_discard=5)
    self.assertLen(seqs, 1)

    self.note_sequence.notes[0].end_time = 1.0
    self.note_sequence.total_time = 1.0
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    seqs, _ = pianoroll_pipeline.extract_pianoroll_sequences(
        quantized_sequence, min_steps_discard=3, max_steps_discard=5)
    self.assertEmpty(seqs)

    self.note_sequence.notes[0].end_time = 10.0
    self.note_sequence.total_time = 10.0
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    seqs, _ = pianoroll_pipeline.extract_pianoroll_sequences(
        quantized_sequence, min_steps_discard=3, max_steps_discard=5)
    self.assertEmpty(seqs)

  def testExtractPianorollMultiProgram(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)])
    self.note_sequence.notes[0].program = 2
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = pianoroll_pipeline.extract_pianoroll_sequences(
        quantized_sequence)
    self.assertEmpty(seqs)

  def testExtractNonZeroStart(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = pianoroll_pipeline.extract_pianoroll_sequences(
        quantized_sequence, start_step=4, min_steps_discard=1)
    self.assertEmpty(seqs)
    seqs, _ = pianoroll_pipeline.extract_pianoroll_sequences(
        quantized_sequence, start_step=0, min_steps_discard=1)
    self.assertLen(seqs, 1)


if __name__ == '__main__':
  absltest.main()
