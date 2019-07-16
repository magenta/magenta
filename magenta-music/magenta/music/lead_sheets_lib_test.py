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

"""Tests for lead_sheets."""

import copy

from magenta.music import chords_lib
from magenta.music import constants
from magenta.music import lead_sheets_lib
from magenta.music import melodies_lib
from magenta.music import testing_lib as music_testing_lib
from magenta.music.protobuf import music_pb2
import tensorflow as tf

NOTE_OFF = constants.MELODY_NOTE_OFF
NO_EVENT = constants.MELODY_NO_EVENT
NO_CHORD = constants.NO_CHORD


class LeadSheetsLibTest(tf.test.TestCase):

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

  def testTranspose(self):
    # LeadSheet transposition should agree with melody & chords transpositions.
    melody_events = [12 * 5 + 4, NO_EVENT, 12 * 5 + 5,
                     NOTE_OFF, 12 * 6, NO_EVENT]
    chord_events = [NO_CHORD, 'C', 'F', 'Dm', 'D', 'G']
    melody = melodies_lib.Melody(melody_events)
    chords = chords_lib.ChordProgression(chord_events)
    expected_melody = copy.deepcopy(melody)
    expected_chords = copy.deepcopy(chords)
    lead_sheet = lead_sheets_lib.LeadSheet(melody, chords)
    lead_sheet.transpose(transpose_amount=-5, min_note=12 * 5, max_note=12 * 7)
    expected_melody.transpose(
        transpose_amount=-5, min_note=12 * 5, max_note=12 * 7)
    expected_chords.transpose(transpose_amount=-5)
    self.assertEqual(expected_melody, lead_sheet.melody)
    self.assertEqual(expected_chords, lead_sheet.chords)

  def testSquash(self):
    # LeadSheet squash should agree with melody squash & chords transpose.
    melody_events = [12 * 5, NO_EVENT, 12 * 5 + 2,
                     NOTE_OFF, 12 * 6 + 4, NO_EVENT]
    chord_events = ['C', 'Am', 'Dm', 'G', 'C', NO_CHORD]
    melody = melodies_lib.Melody(melody_events)
    chords = chords_lib.ChordProgression(chord_events)
    expected_melody = copy.deepcopy(melody)
    expected_chords = copy.deepcopy(chords)
    lead_sheet = lead_sheets_lib.LeadSheet(melody, chords)
    lead_sheet.squash(min_note=12 * 5, max_note=12 * 6, transpose_to_key=0)
    transpose_amount = expected_melody.squash(
        min_note=12 * 5, max_note=12 * 6, transpose_to_key=0)
    expected_chords.transpose(transpose_amount=transpose_amount)
    self.assertEqual(expected_melody, lead_sheet.melody)
    self.assertEqual(expected_chords, lead_sheet.chords)

  def testSetLength(self):
    # Setting LeadSheet length should agree with setting length on melody and
    # chords separately.
    melody_events = [60]
    chord_events = ['C7']
    melody = melodies_lib.Melody(melody_events, start_step=9)
    chords = chords_lib.ChordProgression(chord_events, start_step=9)
    expected_melody = copy.deepcopy(melody)
    expected_chords = copy.deepcopy(chords)
    lead_sheet = lead_sheets_lib.LeadSheet(melody, chords)
    lead_sheet.set_length(5)
    expected_melody.set_length(5)
    expected_chords.set_length(5)
    self.assertEqual(expected_melody, lead_sheet.melody)
    self.assertEqual(expected_chords, lead_sheet.chords)
    self.assertEqual(9, lead_sheet.start_step)
    self.assertEqual(14, lead_sheet.end_step)
    self.assertListEqual([9, 10, 11, 12, 13], lead_sheet.steps)

  def testToSequence(self):
    # Sequence produced from lead sheet should contain notes from melody
    # sequence and chords from chord sequence as text annotations.
    melody = melodies_lib.Melody(
        [NO_EVENT, 1, NO_EVENT, NOTE_OFF, NO_EVENT, 2, 3, NOTE_OFF, NO_EVENT])
    chords = chords_lib.ChordProgression(
        [NO_CHORD, 'A', 'A', 'C#m', 'C#m', 'D', 'B', 'B', 'B'])
    lead_sheet = lead_sheets_lib.LeadSheet(melody, chords)

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

    self.assertEqual(melody_sequence.ticks_per_quarter,
                     sequence.ticks_per_quarter)
    self.assertProtoEquals(melody_sequence.tempos, sequence.tempos)
    self.assertEqual(melody_sequence.total_time, sequence.total_time)
    self.assertProtoEquals(melody_sequence.notes, sequence.notes)
    self.assertProtoEquals(chords_sequence.text_annotations,
                           sequence.text_annotations)


if __name__ == '__main__':
  tf.test.main()
