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

"""Tests for chords_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from magenta.music import chord_symbols_lib
from magenta.music import chords_lib
from magenta.music import constants
from magenta.music import melodies_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.music.protobuf import music_pb2
import tensorflow as tf

NO_CHORD = constants.NO_CHORD


class ChordsLibTest(tf.test.TestCase):

  def setUp(self):
    self.steps_per_quarter = 1
    self.note_sequence = testing_lib.parse_test_proto(
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
    # Transpose ChordProgression with basic triads.
    events = ['Cm', 'F', 'Bb', 'Eb']
    chords = chords_lib.ChordProgression(events)
    chords.transpose(transpose_amount=7)
    expected = ['Gm', 'C', 'F', 'Bb']
    self.assertEqual(expected, list(chords))

    # Transpose ChordProgression with more complex chords.
    events = ['Esus2', 'B13', 'A7/B', 'F#dim']
    chords = chords_lib.ChordProgression(events)
    chords.transpose(transpose_amount=-2)
    expected = ['Dsus2', 'A13', 'G7/A', 'Edim']
    self.assertEqual(expected, list(chords))

    # Transpose ChordProgression containing NO_CHORD.
    events = ['C', 'Bb', NO_CHORD, 'F', 'C']
    chords = chords_lib.ChordProgression(events)
    chords.transpose(transpose_amount=4)
    expected = ['E', 'D', NO_CHORD, 'A', 'E']
    self.assertEqual(expected, list(chords))

  def testTransposeUnknownChordSymbol(self):
    # Attempt to transpose ChordProgression with unknown chord symbol.
    events = ['Cm', 'G7', 'P#13', 'F']
    chords = chords_lib.ChordProgression(events)
    with self.assertRaises(chord_symbols_lib.ChordSymbolError):
      chords.transpose(transpose_amount=-4)

  def testFromQuantizedNoteSequence(self):
    testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('Am', 4), ('D7', 8), ('G13', 12), ('Csus', 14)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    chords = chords_lib.ChordProgression()
    chords.from_quantized_sequence(
        quantized_sequence, start_step=0, end_step=16)
    expected = [NO_CHORD, NO_CHORD, NO_CHORD, NO_CHORD,
                'Am', 'Am', 'Am', 'Am', 'D7', 'D7', 'D7', 'D7',
                'G13', 'G13', 'Csus', 'Csus']
    self.assertEqual(expected, list(chords))

  def testFromQuantizedNoteSequenceWithinSingleChord(self):
    testing_lib.add_chords_to_sequence(
        self.note_sequence, [('F', 0), ('Gm', 8)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    chords = chords_lib.ChordProgression()
    chords.from_quantized_sequence(
        quantized_sequence, start_step=4, end_step=6)
    expected = ['F'] * 2
    self.assertEqual(expected, list(chords))

  def testFromQuantizedNoteSequenceWithNoChords(self):
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    chords = chords_lib.ChordProgression()
    chords.from_quantized_sequence(
        quantized_sequence, start_step=0, end_step=16)
    expected = [NO_CHORD] * 16
    self.assertEqual(expected, list(chords))

  def testFromQuantizedNoteSequenceWithCoincidentChords(self):
    testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('Am', 4), ('D7', 8), ('G13', 12), ('Csus', 12)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    chords = chords_lib.ChordProgression()
    with self.assertRaises(chords_lib.CoincidentChordsError):
      chords.from_quantized_sequence(
          quantized_sequence, start_step=0, end_step=16)

  def testToSequence(self):
    chords = chords_lib.ChordProgression(
        [NO_CHORD, 'C7', 'C7', 'C7', 'C7', 'Am7b5', 'F6', 'F6', NO_CHORD])
    sequence = chords.to_sequence(sequence_start_time=2, qpm=60.0)

    self.assertProtoEquals(
        'ticks_per_quarter: 220 '
        'tempos < qpm: 60.0 > '
        'text_annotations < '
        '  text: "C7" time: 2.25 annotation_type: CHORD_SYMBOL '
        '> '
        'text_annotations < '
        '  text: "Am7b5" time: 3.25 annotation_type: CHORD_SYMBOL '
        '> '
        'text_annotations < '
        '  text: "F6" time: 3.5 annotation_type: CHORD_SYMBOL '
        '> '
        'text_annotations < '
        '  text: "N.C." time: 4.0 annotation_type: CHORD_SYMBOL '
        '> ',
        sequence)

  def testEventListChordsWithMelodies(self):
    note_sequence = music_pb2.NoteSequence(ticks_per_quarter=220)
    note_sequence.tempos.add(qpm=60.0)
    testing_lib.add_chords_to_sequence(
        note_sequence, [('N.C.', 0), ('C', 2), ('G7', 6)])
    note_sequence.total_time = 8.0

    melodies = [
        melodies_lib.Melody([60, -2, -2, -1],
                            start_step=0, steps_per_quarter=1, steps_per_bar=4),
        melodies_lib.Melody([62, -2, -2, -1],
                            start_step=4, steps_per_quarter=1, steps_per_bar=4),
    ]

    quantized_sequence = sequences_lib.quantize_note_sequence(
        note_sequence, steps_per_quarter=1)
    chords = chords_lib.event_list_chords(quantized_sequence, melodies)

    expected_chords = [
        [NO_CHORD, NO_CHORD, 'C', 'C'],
        ['C', 'C', 'G7', 'G7']
    ]

    self.assertEqual(expected_chords, chords)

  def testAddChordsToSequence(self):
    note_sequence = music_pb2.NoteSequence(ticks_per_quarter=220)
    note_sequence.tempos.add(qpm=60.0)
    testing_lib.add_chords_to_sequence(
        note_sequence, [('N.C.', 0), ('C', 2), ('G7', 6)])
    note_sequence.total_time = 8.0

    expected_sequence = copy.deepcopy(note_sequence)
    del note_sequence.text_annotations[:]

    chords = [NO_CHORD, 'C', 'C', 'G7']
    chord_times = [0.0, 2.0, 4.0, 6.0]
    chords_lib.add_chords_to_sequence(note_sequence, chords, chord_times)

    self.assertEqual(expected_sequence, note_sequence)


if __name__ == '__main__':
  tf.test.main()
