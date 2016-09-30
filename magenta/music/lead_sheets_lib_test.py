# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lead_sheets."""

# internal imports
import tensorflow as tf

from magenta.music import chords_lib
from magenta.music import constants
from magenta.music import lead_sheets_lib
from magenta.music import melodies_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib

NOTE_OFF = constants.MELODY_NOTE_OFF
NO_EVENT = constants.MELODY_NO_EVENT
NO_CHORD = constants.NO_CHORD


class LeadSheetsLibTest(tf.test.TestCase):

  def setUp(self):
    self.quantized_sequence = sequences_lib.QuantizedSequence()
    self.quantized_sequence.qpm = 60.0
    self.quantized_sequence.steps_per_quarter = 4

  def testTranspose(self):
    # LeadSheet transposition should agree with melody & chords transpositions.
    melody_events = [12 * 5 + 4, NO_EVENT, 12 * 5 + 5,
                     NOTE_OFF, 12 * 6, NO_EVENT]
    chord_events = [NO_CHORD, 'C', 'F', 'Dm', 'D', 'G']
    lead_sheet = lead_sheets_lib.LeadSheet()
    lead_sheet.from_event_list(zip(melody_events, chord_events))
    lead_sheet.transpose(transpose_amount=-5, min_note=12 * 5, max_note=12 * 7)
    expected_melody = melodies_lib.MonophonicMelody()
    expected_melody.from_event_list(melody_events[:])
    expected_melody.transpose(
        transpose_amount=-5, min_note=12 * 5, max_note=12 * 7)
    expected_chords = chords_lib.ChordProgression()
    expected_chords.from_event_list(chord_events[:])
    expected_chords.transpose(transpose_amount=-5)
    self.assertEqual(expected_melody, lead_sheet.melody)
    self.assertEqual(expected_chords, lead_sheet.chords)

  def testSquash(self):
    # LeadSheet squash should agree with melody squash & chords transpose.
    melody_events = [12 * 5, NO_EVENT, 12 * 5 + 2,
                     NOTE_OFF, 12 * 6 + 4, NO_EVENT]
    chord_events = ['C', 'Am', 'Dm', 'G', 'C', NO_CHORD]
    lead_sheet = lead_sheets_lib.LeadSheet()
    lead_sheet.from_event_list(zip(melody_events, chord_events))
    lead_sheet.squash(min_note=12 * 5, max_note=12 * 6, transpose_to_key=0)
    expected_melody = melodies_lib.MonophonicMelody()
    expected_melody.from_event_list(melody_events[:])
    transpose_amount = expected_melody.squash(
        min_note=12 * 5, max_note=12 * 6, transpose_to_key=0)
    expected_chords = chords_lib.ChordProgression()
    expected_chords.from_event_list(chord_events[:])
    expected_chords.transpose(transpose_amount=transpose_amount)
    self.assertEqual(expected_melody, lead_sheet.melody)
    self.assertEqual(expected_chords, lead_sheet.chords)

  def testExtractLeadSheetFragments(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence,
        [('C', 2), ('G7', 6), ('Cmaj7', 33)])
    lead_sheets, _ = lead_sheets_lib.extract_lead_sheet_fragments(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True, require_chords=True)
    melodies, _ = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, _ = chords_lib.extract_chords_for_melodies(
        self.quantized_sequence, melodies)
    self.assertEqual(list(melodies),
                     list(lead_sheet.melody for lead_sheet in lead_sheets))
    self.assertEqual(list(chord_progressions),
                     list(lead_sheet.chords for lead_sheet in lead_sheets))

  def testExtractLeadSheetFragmentsCoincidentChords(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence,
        [('C', 2), ('G7', 6), ('Cmaj7', 33), ('F', 33)])
    lead_sheets, _ = lead_sheets_lib.extract_lead_sheet_fragments(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True, require_chords=True)
    melodies, _ = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, _ = chords_lib.extract_chords_for_melodies(
        self.quantized_sequence, melodies)
    # Last lead sheet should be rejected for coincident chords.
    self.assertEqual(list(melodies[:2]),
                     list(lead_sheet.melody for lead_sheet in lead_sheets))
    self.assertEqual(list(chord_progressions[:2]),
                     list(lead_sheet.chords for lead_sheet in lead_sheets))

  def testExtractLeadSheetFragmentsNoChords(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence,
        [('C', 2), ('G7', 6), (NO_CHORD, 10)])
    lead_sheets, stats = lead_sheets_lib.extract_lead_sheet_fragments(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True, require_chords=True)
    melodies, _ = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, _ = chords_lib.extract_chords_for_melodies(
        self.quantized_sequence, melodies)
    stats_dict = dict([(stat.name, stat) for stat in stats])
    # Last lead sheet should be rejected for having no chords.
    self.assertEqual(list(melodies[:2]),
                     list(lead_sheet.melody for lead_sheet in lead_sheets))
    self.assertEqual(list(chord_progressions[:2]),
                     list(lead_sheet.chords for lead_sheet in lead_sheets))
    self.assertEqual(stats_dict['empty_chord_progressions'].count, 1)

  def testSetLength(self):
    # Setting LeadSheet length should agree with setting length on melody and
    # chords separately.
    melody_events = [60]
    chord_events = ['C7']
    lead_sheet = lead_sheets_lib.LeadSheet()
    lead_sheet.from_event_list(zip(melody_events, chord_events), start_step=9)
    lead_sheet.set_length(5)
    expected_melody = melodies_lib.MonophonicMelody()
    expected_melody.from_event_list(melody_events[:], start_step=9)
    expected_melody.set_length(5)
    expected_chords = chords_lib.ChordProgression()
    expected_chords.from_event_list(chord_events[:], start_step=9)
    expected_chords.set_length(5)
    self.assertEquals(expected_melody, lead_sheet.melody)
    self.assertEquals(expected_chords, lead_sheet.chords)
    self.assertEquals(9, lead_sheet.start_step)
    self.assertEquals(14, lead_sheet.end_step)

  def testToSequence(self):
    # Sequence produced from lead sheet should contain notes from melody
    # sequence and chords from chord sequence as text annotations.
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list([NO_EVENT, 1, NO_EVENT, NOTE_OFF, NO_EVENT, 2, 3,
                            NOTE_OFF, NO_EVENT])
    chords = chords_lib.ChordProgression()
    chords.from_event_list([NO_CHORD, 'A', 'A', 'C#m', 'C#m', 'D', 'B', 'B',
                            'B'])
    lead_sheet = lead_sheets_lib.LeadSheet()
    lead_sheet.from_melody_and_chords(melody, chords)
    sequence = lead_sheet.to_sequence(
        velocity=10,
        instrument=1,
        sequence_start_time=2,
        qpm=60.0)

    melody_sequence = melody.to_sequence(
        velocity=10,
        instrument=1,
        sequence_start_time=2,
        qpm=60.0)
    chords_sequence = chords.to_sequence(
        sequence_start_time=2,
        qpm=60.0)
    self.assertEquals(melody_sequence.ticks_per_quarter,
                      sequence.ticks_per_quarter)
    self.assertProtoEquals(melody_sequence.tempos, sequence.tempos)
    self.assertEquals(melody_sequence.total_time, sequence.total_time)
    self.assertProtoEquals(melody_sequence.notes, sequence.notes)
    self.assertProtoEquals(chords_sequence.text_annotations,
                           sequence.text_annotations)


if __name__ == '__main__':
  tf.test.main()
