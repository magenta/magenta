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

"""Tests for chord_pipelines."""

from absl.testing import absltest
from magenta.common import testing_lib as common_testing_lib
from magenta.pipelines import chord_pipelines
from magenta.pipelines import melody_pipelines
from note_seq import chords_lib
from note_seq import constants
from note_seq import sequences_lib
from note_seq import testing_lib as music_testing_lib
from note_seq.protobuf import music_pb2

NO_CHORD = constants.NO_CHORD


class ChordPipelinesTest(absltest.TestCase):

  def setUp(self):
    self.steps_per_quarter = 1
    self.note_sequence = music_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4
        }
        tempos: {
          qpm: 60
        }
        """)

  def _unit_transform_test(self, unit, input_instance,
                           expected_outputs):
    outputs = unit.transform(input_instance)
    self.assertIsInstance(outputs, list)
    common_testing_lib.assert_set_equality(self, expected_outputs, outputs)
    self.assertEqual(unit.input_type, type(input_instance))
    if outputs:
      self.assertEqual(unit.output_type, type(outputs[0]))

  def testChordsExtractor(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    music_testing_lib.add_chords_to_sequence(
        note_sequence, [('C', 2), ('Am', 4), ('F', 5)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        note_sequence, steps_per_quarter=1)
    quantized_sequence.total_quantized_steps = 8
    expected_events = [[NO_CHORD, NO_CHORD, 'C', 'C', 'Am', 'F', 'F', 'F']]
    expected_chord_progressions = []
    for events_list in expected_events:
      chords = chords_lib.ChordProgression(
          events_list, steps_per_quarter=1, steps_per_bar=4)
      expected_chord_progressions.append(chords)
    unit = chord_pipelines.ChordsExtractor(all_transpositions=False)
    self._unit_transform_test(unit, quantized_sequence,
                              expected_chord_progressions)

  def testExtractChords(self):
    music_testing_lib.add_chords_to_sequence(
        self.note_sequence, [('C', 2), ('G7', 6), ('F', 8)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    quantized_sequence.total_quantized_steps = 10
    chord_progressions, _ = chord_pipelines.extract_chords(quantized_sequence)
    expected = [[NO_CHORD, NO_CHORD, 'C', 'C', 'C', 'C', 'G7', 'G7', 'F', 'F']]
    self.assertEqual(expected, [list(chords) for chords in chord_progressions])

  def testExtractChordsAllTranspositions(self):
    music_testing_lib.add_chords_to_sequence(
        self.note_sequence, [('C', 1)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    quantized_sequence.total_quantized_steps = 2
    chord_progressions, _ = chord_pipelines.extract_chords(
        quantized_sequence, all_transpositions=True)
    expected = list(zip([NO_CHORD] * 12, ['Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                                          'C', 'Db', 'D', 'Eb', 'E', 'F']))
    self.assertEqual(expected, [tuple(chords) for chords in chord_progressions])

  def testExtractChordsForMelodies(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    music_testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('C', 2), ('G7', 6), ('Cmaj7', 33)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, _ = chord_pipelines.extract_chords_for_melodies(
        quantized_sequence, melodies)
    expected = [[NO_CHORD, NO_CHORD, 'C', 'C', 'C', 'C',
                 'G7', 'G7', 'G7', 'G7', 'G7'],
                [NO_CHORD, NO_CHORD, 'C', 'C', 'C', 'C', 'G7', 'G7'],
                ['G7', 'Cmaj7', 'Cmaj7', 'Cmaj7', 'Cmaj7']]
    self.assertEqual(expected, [list(chords) for chords in chord_progressions])

  def testExtractChordsForMelodiesCoincidentChords(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    music_testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('C', 2), ('G7', 6), ('E13', 8), ('Cmaj7', 8)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, stats = chord_pipelines.extract_chords_for_melodies(
        quantized_sequence, melodies)
    expected = [[NO_CHORD, NO_CHORD, 'C', 'C', 'C', 'C', 'G7', 'G7'],
                ['Cmaj7', 'Cmaj7', 'Cmaj7', 'Cmaj7', 'Cmaj7']]
    stats_dict = dict((stat.name, stat) for stat in stats)
    self.assertIsNone(chord_progressions[0])
    self.assertEqual(expected,
                     [list(chords) for chords in chord_progressions[1:]])
    self.assertEqual(stats_dict['coincident_chords'].count, 1)


if __name__ == '__main__':
  absltest.main()
