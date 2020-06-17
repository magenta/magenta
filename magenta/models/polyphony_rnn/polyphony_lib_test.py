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

"""Tests for polyphony_lib."""

import copy

from magenta.common import testing_lib as common_testing_lib
from magenta.models.polyphony_rnn import polyphony_lib
from note_seq import sequences_lib
from note_seq import testing_lib
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class PolyphonyLibTest(tf.test.TestCase):

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
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    poly_seq = list(polyphony_lib.PolyphonicSequence(quantized_sequence))

    pe = polyphony_lib.PolyphonicEvent
    expected_poly_seq = [
        pe(pe.START, None),
        # step 0
        pe(pe.NEW_NOTE, 64),
        pe(pe.NEW_NOTE, 60),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.NEW_NOTE, 67),
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.STEP_END, None),
        # step 2
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.STEP_END, None),
        # step 3
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(expected_poly_seq, poly_seq)

  def testToSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    poly_seq = polyphony_lib.PolyphonicSequence(quantized_sequence)
    poly_seq_ns = poly_seq.to_sequence(qpm=60.0)

    # Make comparison easier
    poly_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, poly_seq_ns)

  def testToSequenceWithContinuedNotesNotStarted(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.NEW_NOTE, 64),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.CONTINUED_NOTE, 67),  # Was not started, should be ignored.
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    for event in poly_events:
      poly_seq.append(event)

    poly_seq_ns = poly_seq.to_sequence(qpm=60.0)

    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 2.0), (64, 100, 0.0, 2.0)])

    # Make comparison easier
    poly_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, poly_seq_ns)

  def testToSequenceWithExtraEndEvents(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.END, None),  # END event before end. Should be ignored.
        pe(pe.NEW_NOTE, 64),
        pe(pe.END, None),  # END event before end. Should be ignored.
        pe(pe.STEP_END, None),
        pe(pe.END, None),  # END event before end. Should be ignored.
        # step 1
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.END, None),  # END event before end. Should be ignored.
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.END, None),  # END event before end. Should be ignored.
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    for event in poly_events:
      poly_seq.append(event)

    poly_seq_ns = poly_seq.to_sequence(qpm=60.0)

    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 2.0), (64, 100, 0.0, 2.0)])

    # Make comparison easier
    poly_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, poly_seq_ns)

  def testToSequenceWithUnfinishedSequence(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.NEW_NOTE, 64),
        # missing STEP_END and END events at end of sequence.
    ]
    for event in poly_events:
      poly_seq.append(event)

    with self.assertRaises(ValueError):
      poly_seq.to_sequence(qpm=60.0)

  def testToSequenceWithRepeatedNotes(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.NEW_NOTE, 64),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.NEW_NOTE, 60),
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    for event in poly_events:
      poly_seq.append(event)

    poly_seq_ns = poly_seq.to_sequence(qpm=60.0)

    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 1.0), (64, 100, 0.0, 2.0), (60, 100, 1.0, 2.0)])

    # Make comparison easier
    poly_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, poly_seq_ns)

  def testToSequenceWithBaseNoteSequence(self):
    poly_seq = polyphony_lib.PolyphonicSequence(
        steps_per_quarter=1, start_step=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.NEW_NOTE, 64),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    for event in poly_events:
      poly_seq.append(event)

    base_seq = copy.deepcopy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        base_seq, 0, [(60, 100, 0.0, 1.0)])

    poly_seq_ns = poly_seq.to_sequence(qpm=60.0, base_note_sequence=base_seq)

    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 1.0), (60, 100, 1.0, 3.0), (64, 100, 1.0, 3.0)])

    # Make comparison easier
    poly_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, poly_seq_ns)

  def testToSequenceWithEmptySteps(self):
    poly_seq = polyphony_lib.PolyphonicSequence(
        steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    for event in poly_events:
      poly_seq.append(event)

    poly_seq_ns = poly_seq.to_sequence(qpm=60.0)

    self.note_sequence.total_time = 2

    self.assertEqual(self.note_sequence, poly_seq_ns)

  def testSetLengthAddSteps(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)
    poly_seq.set_length(5)

    self.assertEqual(5, poly_seq.num_steps)
    self.assertListEqual([0, 0, 1, 2, 3, 4, 5], poly_seq.steps)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        pe(pe.START, None),

        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

    # Add 5 more steps to make sure END is managed properly.
    poly_seq.set_length(10)

    self.assertEqual(10, poly_seq.num_steps)
    self.assertListEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], poly_seq.steps)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        pe(pe.START, None),

        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

  def testSetLengthAddStepsToSequenceWithoutEnd(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    # Construct a list with one silence step and no END.
    pe = polyphony_lib.PolyphonicEvent
    poly_seq.append(pe(pe.STEP_END, None))

    poly_seq.set_length(2)
    poly_events = [
        pe(pe.START, None),

        pe(pe.STEP_END, None),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

  def testSetLengthAddStepsToSequenceWithUnfinishedStep(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    # Construct a list with one note and no STEP_END or END.
    pe = polyphony_lib.PolyphonicEvent
    poly_seq.append(pe(pe.NEW_NOTE, 60))

    poly_seq.set_length(2)
    poly_events = [
        pe(pe.START, None),

        pe(pe.NEW_NOTE, 60),
        pe(pe.STEP_END, None),

        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

  def testSetLengthRemoveSteps(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.NEW_NOTE, 64),
        pe(pe.STEP_END, None),
        # step 2
        pe(pe.NEW_NOTE, 67),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    for event in poly_events:
      poly_seq.append(event)

    poly_seq.set_length(2)
    poly_events = [
        pe(pe.START, None),
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.NEW_NOTE, 64),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

    poly_seq.set_length(1)
    poly_events = [
        pe(pe.START, None),
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

    poly_seq.set_length(0)
    poly_events = [
        pe(pe.START, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

  def testSetLengthRemoveStepsFromSequenceWithoutEnd(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    # Construct a list with two silence steps and no END.
    pe = polyphony_lib.PolyphonicEvent
    poly_seq.append(pe(pe.STEP_END, None))
    poly_seq.append(pe(pe.STEP_END, None))

    poly_seq.set_length(1)
    poly_events = [
        pe(pe.START, None),

        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

  def testSetLengthRemoveStepsFromSequenceWithUnfinishedStep(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    # Construct a list with a silence step, a new note, and no STEP_END or END.
    pe = polyphony_lib.PolyphonicEvent
    poly_seq.append(pe(pe.STEP_END, None))
    poly_seq.append(pe(pe.NEW_NOTE, 60))

    poly_seq.set_length(1)
    poly_events = [
        pe(pe.START, None),

        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    self.assertEqual(poly_events, list(poly_seq))

  def testNumSteps(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.NEW_NOTE, 64),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    for event in poly_events:
      poly_seq.append(event)

    self.assertEqual(2, poly_seq.num_steps)
    self.assertListEqual([0, 0, 0, 0, 1, 1, 1, 2], poly_seq.steps)

  def testNumStepsIncompleteStep(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.NEW_NOTE, 64),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.STEP_END, None),
        # incomplete step. should not be counted.
        pe(pe.NEW_NOTE, 72),

    ]
    for event in poly_events:
      poly_seq.append(event)

    self.assertEqual(2, poly_seq.num_steps)
    self.assertListEqual([0, 0, 0, 0, 1, 1, 1, 2], poly_seq.steps)

  def testSteps(self):
    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.NEW_NOTE, 64),
        pe(pe.STEP_END, None),
        # step 1
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]

    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)
    for event in poly_events:
      poly_seq.append(event)
    self.assertListEqual([0, 0, 0, 0, 1, 1, 1, 2], poly_seq.steps)

    poly_seq = polyphony_lib.PolyphonicSequence(
        steps_per_quarter=1, start_step=2)
    for event in poly_events:
      poly_seq.append(event)
    self.assertListEqual([2, 2, 2, 2, 3, 3, 3, 4], poly_seq.steps)

  def testTrimTrailingEndEvents(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.STEP_END, None),

        pe(pe.END, None),
        pe(pe.END, None),
    ]
    for event in poly_events:
      poly_seq.append(event)

    poly_seq.trim_trailing_end_events()

    poly_events_expected = [
        pe(pe.START, None),
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.STEP_END, None),
    ]

    self.assertEqual(poly_events_expected, list(poly_seq))

  def testExtractPolyphonicSequences(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = polyphony_lib.extract_polyphonic_sequences(quantized_sequence)
    self.assertEqual(1, len(seqs))

    seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_sequence, min_steps_discard=2, max_steps_discard=5)
    self.assertEqual(1, len(seqs))

    self.note_sequence.notes[0].end_time = 1.0
    self.note_sequence.total_time = 1.0
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_sequence, min_steps_discard=3, max_steps_discard=5)
    self.assertEqual(0, len(seqs))

    self.note_sequence.notes[0].end_time = 10.0
    self.note_sequence.total_time = 10.0
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_sequence, min_steps_discard=3, max_steps_discard=5)
    self.assertEqual(0, len(seqs))

  def testExtractPolyphonicMultiProgram(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)])
    self.note_sequence.notes[0].program = 2
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = polyphony_lib.extract_polyphonic_sequences(quantized_sequence)
    self.assertEqual(0, len(seqs))

  def testExtractNonZeroStart(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_sequence, start_step=4, min_steps_discard=1)
    self.assertEqual(0, len(seqs))
    seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_sequence, start_step=0, min_steps_discard=1)
    self.assertEqual(1, len(seqs))


if __name__ == '__main__':
  tf.test.main()
