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

"""Tests for melody_pipelines."""

from absl.testing import absltest
from magenta.common import testing_lib as common_testing_lib
from magenta.pipelines import melody_pipelines
from note_seq import constants
from note_seq import melodies_lib
from note_seq import sequences_lib
from note_seq import testing_lib as music_testing_lib
from note_seq.protobuf import music_pb2

NOTE_OFF = constants.MELODY_NOTE_OFF
NO_EVENT = constants.MELODY_NO_EVENT


class MelodyPipelinesTest(absltest.TestCase):

  def setUp(self):
    self.steps_per_quarter = 4
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

  def testMelodyExtractor(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    music_testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7)])
    music_testing_lib.add_track_to_sequence(
        note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        note_sequence, steps_per_quarter=1)
    expected_events = [
        [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 11],
        [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14, NO_EVENT]]
    expected_melodies = []
    for events_list in expected_events:
      melody = melodies_lib.Melody(
          events_list, steps_per_quarter=1, steps_per_bar=4)
      expected_melodies.append(melody)
    unit = melody_pipelines.MelodyExtractor(
        min_bars=1, min_unique_pitches=1, gap_bars=1)
    self._unit_transform_test(unit, quantized_sequence, expected_melodies)

  def testExtractMelodiesSimple(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 9)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 9,
        [(13, 100, 2, 4), (15, 25, 6, 8)],
        is_drum=True)

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 11],
                [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT, NO_EVENT]]
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True)

    self.assertLen(melodies, 2)
    self.assertIsInstance(melodies[0], melodies_lib.Melody)
    self.assertIsInstance(melodies[1], melodies_lib.Melody)

    melodies = sorted([list(melody) for melody in melodies])
    self.assertEqual(expected, melodies)

  def testExtractMultipleMelodiesFromSameTrack(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 11,
                 NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT],
                [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT],
                [NO_EVENT, 50, 52, NO_EVENT, NO_EVENT]]
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    melodies = sorted([list(melody) for melody in melodies])
    self.assertEqual(expected, melodies)

  def testExtractMelodiesMelodyTooShort(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 7)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 2,
        [(12, 127, 2, 4), (14, 50, 6, 9)])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT],
                [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT, NO_EVENT]]
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=2, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    melodies = [list(melody) for melody in melodies]
    self.assertEqual(expected, melodies)

  def testExtractMelodiesPadEnd(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 7)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 2,
        [(12, 127, 2, 4), (14, 50, 6, 9)])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NOTE_OFF],
                [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT],
                [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT, NO_EVENT, NOTE_OFF, NO_EVENT, NO_EVENT]]
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True, pad_end=True)
    melodies = [list(melody) for melody in melodies]
    self.assertEqual(expected, melodies)

  def testExtractMelodiesMelodyTooLong(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 15)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 18)])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14] +
                [NO_EVENT] * 7,
                [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14] +
                [NO_EVENT] * 7]
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, max_steps_truncate=14,
        max_steps_discard=18, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    melodies = [list(melody) for melody in melodies]
    self.assertEqual(expected, melodies)

  def testExtractMelodiesMelodyTooLongWithPad(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 15)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 18)])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT]]
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, max_steps_truncate=14,
        max_steps_discard=18, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True, pad_end=True)
    melodies = [list(melody) for melody in melodies]
    self.assertEqual(expected, melodies)

  def testExtractMelodiesTooFewPitches(self):
    # Test that extract_melodies discards melodies with too few pitches where
    # pitches are equivalent by octave.
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0, 1), (13, 100, 1, 2), (18, 100, 2, 3),
         (24, 100, 3, 4), (25, 100, 4, 5)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 100, 0, 1), (13, 100, 1, 2), (18, 100, 2, 3),
         (25, 100, 3, 4), (26, 100, 4, 5)])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    expected = [[12, 13, 18, 25, 26]]
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=1, min_unique_pitches=4,
        ignore_polyphonic_notes=True)
    melodies = [list(melody) for melody in melodies]
    self.assertEqual(expected, melodies)

  def testExtractMelodiesLateStart(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 102, 103), (13, 100, 104, 106)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 100, 100, 101), (13, 100, 102, 105)])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    expected = [[NO_EVENT, NO_EVENT, 12, NOTE_OFF, 13, NO_EVENT],
                [12, NOTE_OFF, 13, NO_EVENT, NO_EVENT]]
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    melodies = sorted([list(melody) for melody in melodies])
    self.assertEqual(expected, melodies)

  def testExtractMelodiesStatistics(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7), (10, 100, 8, 10), (9, 100, 11, 14),
         (8, 100, 16, 40), (7, 100, 41, 42)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 2, 8)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 2,
        [(12, 127, 0, 1)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 3,
        [(12, 127, 2, 4), (12, 50, 6, 8)])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    _, stats = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=False)

    stats_dict = dict((stat.name, stat) for stat in stats)
    self.assertEqual(stats_dict['polyphonic_tracks_discarded'].count, 1)
    self.assertEqual(stats_dict['melodies_discarded_too_short'].count, 1)
    self.assertEqual(stats_dict['melodies_discarded_too_few_pitches'].count, 1)
    self.assertEqual(
        stats_dict['melody_lengths_in_bars'].counters,
        {float('-inf'): 0, 0: 0, 1: 0, 2: 0, 10: 1, 20: 0, 30: 0, 40: 0, 50: 0,
         100: 0, 200: 0, 500: 0})


if __name__ == '__main__':
  absltest.main()
