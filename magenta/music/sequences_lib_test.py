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
"""Tests for sequences_lib."""

import copy

# internal imports
import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2


class SequencesLibTest(tf.test.TestCase):

  def setUp(self):
    self.steps_per_quarter = 4
    self.note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    self.expected_quantized_sequence = sequences_lib.QuantizedSequence()
    self.expected_quantized_sequence.qpm = 60.0
    self.expected_quantized_sequence.steps_per_quarter = self.steps_per_quarter

  def testExtractSubsequence(self):
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    expected_subsequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        expected_subsequence, 0,
        [(40, 45, 2.50, 3.50), (55, 120, 4.0, 4.01)])
    expected_subsequence.total_time = 4.75

    subsequence = sequences_lib.extract_subsequence(sequence, 2.5, 4.75)
    self.assertProtoEquals(expected_subsequence, subsequence)

  def testEq(self):
    left_hand = sequences_lib.QuantizedSequence()
    left_hand.qpm = 123.0
    left_hand.steps_per_quarter = 7
    left_hand.time_signature = sequences_lib.QuantizedSequence.TimeSignature(
        numerator=7, denominator=8)
    testing_lib.add_quantized_track_to_sequence(
        left_hand, 0,
        [(12, 100, 0, 40), (11, 100, 1, 2)])
    testing_lib.add_quantized_track_to_sequence(
        left_hand, 2,
        [(55, 100, 4, 6), (14, 120, 4, 10)])
    testing_lib.add_quantized_track_to_sequence(
        left_hand, 3,
        [(1, 10, 0, 6), (2, 50, 20, 21), (0, 101, 17, 21)])
    testing_lib.add_quantized_chords_to_sequence(
        left_hand, [('Cmaj7', 1), ('G9', 2)])
    right_hand = sequences_lib.QuantizedSequence()
    right_hand.qpm = 123.0
    right_hand.steps_per_quarter = 7
    right_hand.time_signature = sequences_lib.QuantizedSequence.TimeSignature(
        numerator=7, denominator=8)
    testing_lib.add_quantized_track_to_sequence(
        right_hand, 0,
        [(11, 100, 1, 2), (12, 100, 0, 40)])
    testing_lib.add_quantized_track_to_sequence(
        right_hand, 2,
        [(14, 120, 4, 10), (55, 100, 4, 6)])
    testing_lib.add_quantized_track_to_sequence(
        right_hand, 3,
        [(0, 101, 17, 21), (2, 50, 20, 21), (1, 10, 0, 6)])
    testing_lib.add_quantized_chords_to_sequence(
        right_hand, [('G9', 2), ('Cmaj7', 1)])
    self.assertEqual(left_hand, right_hand)

  def testNotEq(self):
    left_hand = sequences_lib.QuantizedSequence()
    left_hand.bpm = 123.0
    left_hand.steps_per_beat = 7
    left_hand.time_signature = sequences_lib.QuantizedSequence.TimeSignature(
        numerator=7, denominator=8)
    testing_lib.add_quantized_track_to_sequence(
        left_hand, 0,
        [(12, 100, 0, 40), (11, 100, 1, 2)])
    testing_lib.add_quantized_track_to_sequence(
        left_hand, 2,
        [(55, 100, 4, 6), (15, 120, 4, 10)])
    testing_lib.add_quantized_track_to_sequence(
        left_hand, 3,
        [(1, 10, 0, 6), (2, 50, 20, 21), (0, 101, 17, 21)])
    testing_lib.add_quantized_chords_to_sequence(
        left_hand, [('Cmaj7', 1), ('G9', 2)])
    right_hand = sequences_lib.QuantizedSequence()
    right_hand.bpm = 123.0
    right_hand.steps_per_beat = 7
    right_hand.time_signature = sequences_lib.QuantizedSequence.TimeSignature(
        numerator=7, denominator=8)
    testing_lib.add_quantized_track_to_sequence(
        right_hand, 0,
        [(11, 100, 1, 2), (12, 100, 0, 40)])
    testing_lib.add_quantized_track_to_sequence(
        right_hand, 2,
        [(14, 120, 4, 10), (55, 100, 4, 6)])
    testing_lib.add_quantized_track_to_sequence(
        right_hand, 3,
        [(0, 101, 17, 21), (2, 50, 20, 21), (1, 10, 0, 6)])
    testing_lib.add_quantized_chords_to_sequence(
        right_hand, [('G9', 2), ('C7', 1)])
    self.assertNotEqual(left_hand, right_hand)

  def testFromNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('B7', 0.22), ('Em9', 4.0)])
    testing_lib.add_quantized_track_to_sequence(
        self.expected_quantized_sequence, 0,
        [(12, 100, 0, 40), (11, 55, 1, 2), (40, 45, 10, 14),
         (55, 120, 16, 17), (52, 99, 19, 20)])
    testing_lib.add_quantized_chords_to_sequence(
        self.expected_quantized_sequence,
        [('B7', 1), ('Em9', 16)])
    quantized = sequences_lib.QuantizedSequence()
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)
    self.assertEqual(self.expected_quantized_sequence, quantized)

  def testFromNoteSequence_TimeSignatureChange(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    del self.note_sequence.time_signatures[:]
    quantized = sequences_lib.QuantizedSequence()
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)

    # Single time signature.
    self.note_sequence.time_signatures.add(numerator=4, denominator=4, time=0)
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)

    # Multiple time signatures with no change.
    self.note_sequence.time_signatures.add(numerator=4, denominator=4, time=1)
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)

    # Time signature change.
    self.note_sequence.time_signatures.add(numerator=2, denominator=4, time=2)
    with self.assertRaises(sequences_lib.MultipleTimeSignatureException):
      quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)

  def testRounding(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 100, 0.01, 0.24), (11, 100, 0.22, 0.55), (40, 100, 0.50, 0.75),
         (41, 100, 0.689, 1.18), (44, 100, 1.19, 1.69), (55, 100, 4.0, 4.01)])
    testing_lib.add_quantized_track_to_sequence(
        self.expected_quantized_sequence, 1,
        [(12, 100, 0, 1), (11, 100, 1, 2), (40, 100, 2, 3),
         (41, 100, 3, 5), (44, 100, 5, 7), (55, 100, 16, 17)])
    quantized = sequences_lib.QuantizedSequence()
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)
    self.assertEqual(self.expected_quantized_sequence, quantized)

  def testMultiTrack(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 1.0, 4.0), (19, 100, 0.95, 3.0)])
    testing_lib.add_track_to_sequence(
        self.note_sequence, 3,
        [(12, 100, 1.0, 4.0), (19, 100, 2.0, 5.0)])
    testing_lib.add_track_to_sequence(
        self.note_sequence, 7,
        [(12, 100, 1.0, 5.0), (19, 100, 2.0, 4.0), (24, 100, 3.0, 3.5)])
    testing_lib.add_quantized_track_to_sequence(
        self.expected_quantized_sequence, 0,
        [(12, 100, 4, 16), (19, 100, 4, 12)])
    testing_lib.add_quantized_track_to_sequence(
        self.expected_quantized_sequence, 3,
        [(12, 100, 4, 16), (19, 100, 8, 20)])
    testing_lib.add_quantized_track_to_sequence(
        self.expected_quantized_sequence, 7,
        [(12, 100, 4, 20), (19, 100, 8, 16), (24, 100, 12, 14)])
    quantized = sequences_lib.QuantizedSequence()
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)
    self.assertEqual(self.expected_quantized_sequence, quantized)

  def testStepsPerBar(self):
    quantized = sequences_lib.QuantizedSequence()
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)
    self.assertEqual(16, quantized.steps_per_bar())

    self.note_sequence.time_signatures[0].numerator = 6
    self.note_sequence.time_signatures[0].denominator = 8
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)
    self.assertEqual(12.0, quantized.steps_per_bar())

  def testFilterDrums(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 1.0, 4.0), (19, 100, 0.95, 3.0)])
    testing_lib.add_track_to_sequence(
        self.note_sequence, 3,
        [(12, 100, 1.0, 4.0), (19, 100, 2.0, 5.0)])

    # Make instrument 0 a drum.
    for note in self.note_sequence.notes:
      if note.instrument == 0:
        note.is_drum = True

    testing_lib.add_quantized_track_to_sequence(
        self.expected_quantized_sequence, 3,
        [(12, 100, 4, 16), (19, 100, 8, 20)])

    quantized = sequences_lib.QuantizedSequence()
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)
    self.assertEqual(self.expected_quantized_sequence, quantized)

  def testDeepcopy(self):
    quantized = sequences_lib.QuantizedSequence()
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    quantized.from_note_sequence(self.note_sequence, self.steps_per_quarter)

    quantized_copy = copy.deepcopy(quantized)
    self.assertEqual(quantized, quantized_copy)

    testing_lib.add_quantized_track_to_sequence(
        quantized, 1,
        [(12, 100, 4, 20), (19, 100, 8, 16), (24, 100, 12, 14)])

    self.assertNotEqual(quantized, quantized_copy)


if __name__ == '__main__':
  tf.test.main()
