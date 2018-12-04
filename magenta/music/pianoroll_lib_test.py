# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Tests for pianoroll_lib."""

import copy

import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib

from magenta.music import pianoroll_lib

from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2


class PianorollLibTest(tf.test.TestCase):

  def setUp(self):
    self.maxDiff = None  # pylint:disable=invalid-name

    self.note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        tempos: {
          qpm: 60
        }
        ticks_per_quarter: 220
        """)

  def testFromQuantizedNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(20, 100, 0.0, 4.0), (24, 100, 0.0, 1.0), (26, 100, 0.0, 3.0),
         (110, 100, 1.0, 2.0), (24, 100, 2.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    pianoroll_seq = list(pianoroll_lib.PianorollSequence(quantized_sequence))

    expected_pianoroll_seq = [
        (3, 5),
        (5,),
        (3, 5),
        (3,),
    ]
    self.assertEqual(expected_pianoroll_seq, pianoroll_seq)

  def testFromQuantizedNoteSequence_SplitRepeats(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(0, 100, 0.0, 2.0), (0, 100, 2.0, 4.0), (1, 100, 0.0, 2.0),
         (2, 100, 2.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    pianoroll_seq = list(pianoroll_lib.PianorollSequence(
        quantized_sequence, min_pitch=0, split_repeats=True))

    expected_pianoroll_seq = [
        (0, 1),
        (1,),
        (0, 2),
        (0, 2),
    ]
    self.assertEqual(expected_pianoroll_seq, pianoroll_seq)

  def testFromEventsList_ShiftRange(self):
    pianoroll_seq = list(pianoroll_lib.PianorollSequence(
        events_list=[(0, 1), (2, 3), (4, 5), (6,)], steps_per_quarter=1,
        min_pitch=1, max_pitch=4, shift_range=True))

    expected_pianoroll_seq = [
        (0,),
        (1, 2),
        (3,),
        (),
    ]
    self.assertEqual(expected_pianoroll_seq, pianoroll_seq)

  def testToSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 0.0, 1.0),
         (67, 100, 3.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    pianoroll_seq = pianoroll_lib.PianorollSequence(quantized_sequence)
    pianoroll_seq_ns = pianoroll_seq.to_sequence(qpm=60.0)

    # Make comparison easier
    pianoroll_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, pianoroll_seq_ns)

  def testToSequenceWithBaseNoteSequence(self):
    pianoroll_seq = pianoroll_lib.PianorollSequence(
        steps_per_quarter=1, start_step=1)

    pianoroll_events = [(39, 43), (39, 43)]
    for event in pianoroll_events:
      pianoroll_seq.append(event)

    base_seq = copy.deepcopy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        base_seq, 0, [(60, 100, 0.0, 1.0)])

    pianoroll_seq_ns = pianoroll_seq.to_sequence(
        qpm=60.0, base_note_sequence=base_seq)

    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 1.0), (60, 100, 1.0, 3.0), (64, 100, 1.0, 3.0)])

    # Make comparison easier
    pianoroll_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, pianoroll_seq_ns)

  def testSetLengthAddSteps(self):
    pianoroll_seq = pianoroll_lib.PianorollSequence(steps_per_quarter=1)
    pianoroll_seq.append((0))

    self.assertEqual(1, pianoroll_seq.num_steps)
    self.assertListEqual([0], pianoroll_seq.steps)

    pianoroll_seq.set_length(5)

    self.assertEqual(5, pianoroll_seq.num_steps)
    self.assertListEqual([0, 1, 2, 3, 4], pianoroll_seq.steps)

    self.assertEqual([(0), (), (), (), ()], list(pianoroll_seq))

    # Add 5 more steps.
    pianoroll_seq.set_length(10)

    self.assertEqual(10, pianoroll_seq.num_steps)
    self.assertListEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pianoroll_seq.steps)

    self.assertEqual([(0)] + [()] * 9, list(pianoroll_seq))

  def testSetLengthRemoveSteps(self):
    pianoroll_seq = pianoroll_lib.PianorollSequence(steps_per_quarter=1)

    pianoroll_events = [(), (2, 4), (2, 4), (2,), (5,)]
    for event in pianoroll_events:
      pianoroll_seq.append(event)

    pianoroll_seq.set_length(2)

    self.assertEqual([(), (2, 4)], list(pianoroll_seq))

    pianoroll_seq.set_length(1)
    self.assertEqual([()], list(pianoroll_seq))

    pianoroll_seq.set_length(0)
    self.assertEqual([], list(pianoroll_seq))

  def testExtractPianorollSequences(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = pianoroll_lib.extract_pianoroll_sequences(quantized_sequence)
    self.assertEqual(1, len(seqs))

    seqs, _ = pianoroll_lib.extract_pianoroll_sequences(
        quantized_sequence, min_steps_discard=2, max_steps_discard=5)
    self.assertEqual(1, len(seqs))

    self.note_sequence.notes[0].end_time = 1.0
    self.note_sequence.total_time = 1.0
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    seqs, _ = pianoroll_lib.extract_pianoroll_sequences(
        quantized_sequence, min_steps_discard=3, max_steps_discard=5)
    self.assertEqual(0, len(seqs))

    self.note_sequence.notes[0].end_time = 10.0
    self.note_sequence.total_time = 10.0
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    seqs, _ = pianoroll_lib.extract_pianoroll_sequences(
        quantized_sequence, min_steps_discard=3, max_steps_discard=5)
    self.assertEqual(0, len(seqs))

  def testExtractPianorollMultiProgram(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)])
    self.note_sequence.notes[0].program = 2
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = pianoroll_lib.extract_pianoroll_sequences(quantized_sequence)
    self.assertEqual(0, len(seqs))

  def testExtractNonZeroStart(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = pianoroll_lib.extract_pianoroll_sequences(
        quantized_sequence, start_step=4, min_steps_discard=1)
    self.assertEqual(0, len(seqs))
    seqs, _ = pianoroll_lib.extract_pianoroll_sequences(
        quantized_sequence, start_step=0, min_steps_discard=1)
    self.assertEqual(1, len(seqs))


if __name__ == '__main__':
  tf.test.main()
