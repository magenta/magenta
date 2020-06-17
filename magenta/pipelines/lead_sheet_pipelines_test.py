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

"""Tests for lead_sheet_pipelines."""

from absl.testing import absltest
from magenta.common import testing_lib as common_testing_lib
from magenta.pipelines import chord_pipelines
from magenta.pipelines import lead_sheet_pipelines
from magenta.pipelines import melody_pipelines
from note_seq import chords_lib
from note_seq import constants
from note_seq import lead_sheets_lib
from note_seq import melodies_lib
from note_seq import sequences_lib
from note_seq import testing_lib as music_testing_lib
from note_seq.protobuf import music_pb2

NOTE_OFF = constants.MELODY_NOTE_OFF
NO_EVENT = constants.MELODY_NO_EVENT
NO_CHORD = constants.NO_CHORD


class LeadSheetPipelinesTest(absltest.TestCase):

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

  def testLeadSheetExtractor(self):
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
    music_testing_lib.add_chords_to_sequence(
        note_sequence,
        [('Cm7', 2), ('F9', 4), ('G7b9', 6)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        note_sequence, steps_per_quarter=1)
    expected_melody_events = [
        [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 11],
        [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14, NO_EVENT]]
    expected_chord_events = [
        [NO_CHORD, NO_CHORD, 'Cm7', 'Cm7', 'F9', 'F9', 'G7b9'],
        [NO_CHORD, NO_CHORD, 'Cm7', 'Cm7', 'F9', 'F9', 'G7b9', 'G7b9']]
    expected_lead_sheets = []
    for melody_events, chord_events in zip(expected_melody_events,
                                           expected_chord_events):
      melody = melodies_lib.Melody(
          melody_events, steps_per_quarter=1, steps_per_bar=4)
      chords = chords_lib.ChordProgression(
          chord_events, steps_per_quarter=1, steps_per_bar=4)
      lead_sheet = lead_sheets_lib.LeadSheet(melody, chords)
      expected_lead_sheets.append(lead_sheet)
    unit = lead_sheet_pipelines.LeadSheetExtractor(
        min_bars=1, min_unique_pitches=1, gap_bars=1, all_transpositions=False)
    self._unit_transform_test(unit, quantized_sequence, expected_lead_sheets)

  def testExtractLeadSheetFragments(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, .5, 1), (11, 1, 1.5, 2.75)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, .5, 1), (14, 50, 1.5, 2),
         (50, 100, 8.25, 9.25), (52, 100, 8.5, 9.25)])
    music_testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('C', .5), ('G7', 1.5), ('Cmaj7', 8.25)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    lead_sheets, _ = lead_sheet_pipelines.extract_lead_sheet_fragments(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True, require_chords=True)
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, _ = chord_pipelines.extract_chords_for_melodies(
        quantized_sequence, melodies)
    self.assertEqual(list(melodies),
                     list(lead_sheet.melody for lead_sheet in lead_sheets))
    self.assertEqual(list(chord_progressions),
                     list(lead_sheet.chords for lead_sheet in lead_sheets))

  def testExtractLeadSheetFragmentsCoincidentChords(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    music_testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('C', 2), ('G7', 6), ('Cmaj7', 33), ('F', 33)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    lead_sheets, _ = lead_sheet_pipelines.extract_lead_sheet_fragments(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True, require_chords=True)
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, _ = chord_pipelines.extract_chords_for_melodies(
        quantized_sequence, melodies)
    # Last lead sheet should be rejected for coincident chords.
    self.assertEqual(list(melodies[:2]),
                     list(lead_sheet.melody for lead_sheet in lead_sheets))
    self.assertEqual(list(chord_progressions[:2]),
                     list(lead_sheet.chords for lead_sheet in lead_sheets))

  def testExtractLeadSheetFragmentsNoChords(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    music_testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('C', 2), ('G7', 6), (NO_CHORD, 10)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    lead_sheets, stats = lead_sheet_pipelines.extract_lead_sheet_fragments(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True, require_chords=True)
    melodies, _ = melody_pipelines.extract_melodies(
        quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, _ = chord_pipelines.extract_chords_for_melodies(
        quantized_sequence, melodies)
    stats_dict = dict((stat.name, stat) for stat in stats)
    # Last lead sheet should be rejected for having no chords.
    self.assertEqual(list(melodies[:2]),
                     list(lead_sheet.melody for lead_sheet in lead_sheets))
    self.assertEqual(list(chord_progressions[:2]),
                     list(lead_sheet.chords for lead_sheet in lead_sheets))
    self.assertEqual(stats_dict['empty_chord_progressions'].count, 1)

if __name__ == '__main__':
  absltest.main()
