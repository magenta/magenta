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

"""Tests for note_sequence_pipelines."""

from absl.testing import absltest
from magenta.common import testing_lib as common_testing_lib
from magenta.pipelines import note_sequence_pipelines
from note_seq import sequences_lib
from note_seq import testing_lib
from note_seq.protobuf import music_pb2


class PipelineUnitsCommonTest(absltest.TestCase):

  def _unit_transform_test(self, unit, input_instance,
                           expected_outputs):
    outputs = unit.transform(input_instance)
    self.assertIsInstance(outputs, list)
    common_testing_lib.assert_set_equality(self, expected_outputs, outputs)
    self.assertEqual(unit.input_type, type(input_instance))
    if outputs:
      self.assertEqual(unit.output_type, type(outputs[0]))

  def testSplitter(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    expected_sequences = sequences_lib.split_note_sequence(note_sequence, 1.0)

    unit = note_sequence_pipelines.Splitter(1.0)
    self._unit_transform_test(unit, note_sequence, expected_sequences)

  def testTimeChangeSplitter(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          time: 2.0
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    expected_sequences = sequences_lib.split_note_sequence_on_time_changes(
        note_sequence)

    unit = note_sequence_pipelines.TimeChangeSplitter()
    self._unit_transform_test(unit, note_sequence, expected_sequences)

  def testQuantizer(self):
    steps_per_quarter = 4
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    expected_quantized_sequence = sequences_lib.quantize_note_sequence(
        note_sequence, steps_per_quarter)

    unit = note_sequence_pipelines.Quantizer(steps_per_quarter)
    self._unit_transform_test(unit, note_sequence,
                              [expected_quantized_sequence])

  def testSustainPipeline(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50), (55, 120, 4.0, 4.01)])
    testing_lib.add_control_changes_to_sequence(
        note_sequence, 0,
        [(0.0, 64, 127), (0.75, 64, 0), (2.0, 64, 127), (3.0, 64, 0),
         (3.75, 64, 127), (4.5, 64, 127), (4.8, 64, 0), (4.9, 64, 127),
         (6.0, 64, 0)])
    expected_sequence = sequences_lib.apply_sustain_control_changes(
        note_sequence)

    unit = note_sequence_pipelines.SustainPipeline()
    self._unit_transform_test(unit, note_sequence, [expected_sequence])

  def testStretchPipeline(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          time: 1.0
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50), (55, 120, 4.0, 4.01)])

    expected_sequences = [
        sequences_lib.stretch_note_sequence(note_sequence, 0.5),
        sequences_lib.stretch_note_sequence(note_sequence, 1.0),
        sequences_lib.stretch_note_sequence(note_sequence, 1.5)]

    unit = note_sequence_pipelines.StretchPipeline(
        stretch_factors=[0.5, 1.0, 1.5])
    self._unit_transform_test(unit, note_sequence, expected_sequences)

  def testTranspositionPipeline(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    tp = note_sequence_pipelines.TranspositionPipeline(range(0, 2))
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 1.0, 4.0)])
    testing_lib.add_track_to_sequence(
        note_sequence, 1,
        [(36, 100, 2.0, 2.01)],
        is_drum=True)
    transposed = tp.transform(note_sequence)
    self.assertLen(transposed, 2)
    self.assertLen(transposed[0].notes, 2)
    self.assertLen(transposed[1].notes, 2)
    self.assertEqual(12, transposed[0].notes[0].pitch)
    self.assertEqual(13, transposed[1].notes[0].pitch)
    self.assertEqual(36, transposed[0].notes[1].pitch)
    self.assertEqual(36, transposed[1].notes[1].pitch)

  def testTranspositionPipelineOutOfRangeNotes(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    tp = note_sequence_pipelines.TranspositionPipeline(
        range(-1, 2), min_pitch=0, max_pitch=12)
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(10, 100, 1.0, 2.0), (12, 100, 2.0, 4.0), (13, 100, 4.0, 5.0)])
    transposed = tp.transform(note_sequence)
    self.assertLen(transposed, 1)
    self.assertLen(transposed[0].notes, 3)
    self.assertEqual(9, transposed[0].notes[0].pitch)
    self.assertEqual(11, transposed[0].notes[1].pitch)
    self.assertEqual(12, transposed[0].notes[2].pitch)


if __name__ == '__main__':
  absltest.main()
