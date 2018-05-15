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
    self.maxDiff = None

    self.steps_per_quarter = 4
    self.note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")

  def testTrimNoteSequence(self):
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

    subsequence = sequences_lib.trim_note_sequence(sequence, 2.5, 4.75)
    self.assertProtoEquals(expected_subsequence, subsequence)

  def testExtractSubsequence(self):
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.5), ('G7', 3.0), ('F', 4.8)])
    testing_lib.add_control_changes_to_sequence(
        sequence, 0,
        [(0.0, 64, 127), (2.0, 64, 0), (4.0, 64, 127), (5.0, 64, 0)])
    testing_lib.add_control_changes_to_sequence(
        sequence, 1, [(2.0, 64, 127)])
    expected_subsequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        expected_subsequence, 0,
        [(40, 45, 0.0, 1.0), (55, 120, 1.5, 1.51)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence, [('C', 0.0), ('G7', 0.5)])
    testing_lib.add_control_changes_to_sequence(
        expected_subsequence, 0, [(0.0, 64, 0), (1.5, 64, 127)])
    testing_lib.add_control_changes_to_sequence(
        expected_subsequence, 1, [(0.0, 64, 127)])
    expected_subsequence.control_changes.sort(key=lambda cc: cc.time)
    expected_subsequence.total_time = 1.51
    expected_subsequence.subsequence_info.start_time_offset = 2.5
    expected_subsequence.subsequence_info.end_time_offset = 5.99

    subsequence = sequences_lib.extract_subsequence(sequence, 2.5, 4.75)
    self.assertProtoEquals(expected_subsequence, subsequence)

  def testExtractSubsequencePastEnd(self):
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.5), ('G7', 3.0), ('F', 18.0)])

    with self.assertRaises(ValueError):
      sequences_lib.extract_subsequence(sequence, 15.0, 16.0)

  def testSplitNoteSequenceWithHopSize(self):
    # Tests splitting a NoteSequence at regular hop size, truncating notes.
    sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 8.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.0), ('G7', 2.0), ('F', 4.0)])

    expected_subsequence_1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_1, 0,
        [(12, 100, 0.01, 3.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.0)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_1, [('C', 1.0), ('G7', 2.0)])
    expected_subsequence_1.total_time = 3.0
    expected_subsequence_1.subsequence_info.end_time_offset = 5.0

    expected_subsequence_2 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_2, 0,
        [(55, 120, 1.0, 1.01), (52, 99, 1.75, 2.0)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_2, [('G7', 0.0), ('F', 1.0)])
    expected_subsequence_2.total_time = 2.0
    expected_subsequence_2.subsequence_info.start_time_offset = 3.0
    expected_subsequence_2.subsequence_info.end_time_offset = 3.0

    expected_subsequence_3 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_chords_to_sequence(
        expected_subsequence_3, [('F', 0.0)])
    expected_subsequence_3.total_time = 0.0
    expected_subsequence_3.subsequence_info.start_time_offset = 6.0
    expected_subsequence_3.subsequence_info.end_time_offset = 2.0

    subsequences = sequences_lib.split_note_sequence(
        sequence, hop_size_seconds=3.0)
    self.assertEquals(3, len(subsequences))
    self.assertProtoEquals(expected_subsequence_1, subsequences[0])
    self.assertProtoEquals(expected_subsequence_2, subsequences[1])
    self.assertProtoEquals(expected_subsequence_3, subsequences[2])

  def testSplitNoteSequenceAtTimes(self):
    # Tests splitting a NoteSequence at specified times, truncating notes.
    sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 8.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.0), ('G7', 2.0), ('F', 4.0)])

    expected_subsequence_1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_1, 0,
        [(12, 100, 0.01, 3.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.0)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_1, [('C', 1.0), ('G7', 2.0)])
    expected_subsequence_1.total_time = 3.0
    expected_subsequence_1.subsequence_info.end_time_offset = 5.0

    expected_subsequence_2 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_chords_to_sequence(
        expected_subsequence_2, [('G7', 0.0)])
    expected_subsequence_2.total_time = 0.0
    expected_subsequence_2.subsequence_info.start_time_offset = 3.0
    expected_subsequence_2.subsequence_info.end_time_offset = 5.0

    expected_subsequence_3 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_3, 0,
        [(55, 120, 0.0, 0.01), (52, 99, 0.75, 1.0)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_3, [('F', 0.0)])
    expected_subsequence_3.total_time = 1.0
    expected_subsequence_3.subsequence_info.start_time_offset = 4.0
    expected_subsequence_3.subsequence_info.end_time_offset = 3.0

    subsequences = sequences_lib.split_note_sequence(
        sequence, hop_size_seconds=[3.0, 4.0])
    self.assertEquals(3, len(subsequences))
    self.assertProtoEquals(expected_subsequence_1, subsequences[0])
    self.assertProtoEquals(expected_subsequence_2, subsequences[1])
    self.assertProtoEquals(expected_subsequence_3, subsequences[2])

  def testSplitNoteSequenceSkipSplitsInsideNotes(self):
    # Tests splitting a NoteSequence at regular hop size, skipping splits that
    # would have occurred inside a note.
    sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 3.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 0.0), ('G7', 3.0), ('F', 4.5)])

    expected_subsequence_1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_1, 0,
        [(12, 100, 0.01, 3.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_1, [('C', 0.0), ('G7', 3.0)])
    expected_subsequence_1.total_time = 3.50
    expected_subsequence_1.subsequence_info.end_time_offset = 1.5

    expected_subsequence_2 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_2, 0,
        [(55, 120, 0.0, 0.01), (52, 99, 0.75, 1.0)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_2, [('G7', 0.0), ('F', 0.5)])
    expected_subsequence_2.total_time = 1.0
    expected_subsequence_2.subsequence_info.start_time_offset = 4.0

    subsequences = sequences_lib.split_note_sequence(
        sequence, hop_size_seconds=2.0, skip_splits_inside_notes=True)
    self.assertEquals(2, len(subsequences))
    self.assertProtoEquals(expected_subsequence_1, subsequences[0])
    self.assertProtoEquals(expected_subsequence_2, subsequences[1])

  def testSplitNoteSequenceNoTimeChanges(self):
    # Tests splitting a NoteSequence on time changes for a NoteSequence that has
    # no time changes (time signature and tempo changes).
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.5), ('G7', 3.0), ('F', 4.8)])

    expected_subsequence = music_pb2.NoteSequence()
    expected_subsequence.CopyFrom(sequence)
    expected_subsequence.subsequence_info.start_time_offset = 0.0
    expected_subsequence.subsequence_info.end_time_offset = 0.0

    subsequences = sequences_lib.split_note_sequence_on_time_changes(sequence)
    self.assertEquals(1, len(subsequences))
    self.assertProtoEquals(expected_subsequence, subsequences[0])

  def testSplitNoteSequenceDuplicateTimeChanges(self):
    # Tests splitting a NoteSequence on time changes for a NoteSequence that has
    # duplicate time changes.
    sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        time_signatures: {
          time: 2.0
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.5), ('G7', 3.0), ('F', 4.8)])

    expected_subsequence = music_pb2.NoteSequence()
    expected_subsequence.CopyFrom(sequence)
    expected_subsequence.subsequence_info.start_time_offset = 0.0
    expected_subsequence.subsequence_info.end_time_offset = 0.0

    subsequences = sequences_lib.split_note_sequence_on_time_changes(sequence)
    self.assertEquals(1, len(subsequences))
    self.assertProtoEquals(expected_subsequence, subsequences[0])

  def testSplitNoteSequenceCoincidentTimeChanges(self):
    # Tests splitting a NoteSequence on time changes for a NoteSequence that has
    # two time changes occurring simultaneously.
    sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        time_signatures: {
          time: 2.0
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 60}
        tempos: {
          time: 2.0
          qpm: 80}""")
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.5), ('G7', 3.0), ('F', 4.8)])

    expected_subsequence_1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_1, 0,
        [(12, 100, 0.01, 2.0), (11, 55, 0.22, 0.50)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_1, [('C', 1.5)])
    expected_subsequence_1.total_time = 2.0
    expected_subsequence_1.subsequence_info.end_time_offset = 8.0

    expected_subsequence_2 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 80}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_2, 0,
        [(40, 45, 0.50, 1.50), (55, 120, 2.0, 2.01), (52, 99, 2.75, 3.0)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_2, [('C', 0.0), ('G7', 1.0), ('F', 2.8)])
    expected_subsequence_2.total_time = 3.0
    expected_subsequence_2.subsequence_info.start_time_offset = 2.0
    expected_subsequence_2.subsequence_info.end_time_offset = 5.0

    subsequences = sequences_lib.split_note_sequence_on_time_changes(sequence)
    self.assertEquals(2, len(subsequences))
    self.assertProtoEquals(expected_subsequence_1, subsequences[0])
    self.assertProtoEquals(expected_subsequence_2, subsequences[1])

  def testSplitNoteSequenceMultipleTimeChangesSkipSplitsInsideNotes(self):
    # Tests splitting a NoteSequence on time changes skipping splits that occur
    # inside notes.
    sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        time_signatures: {
          time: 2.0
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 60}
        tempos: {
          time: 4.25
          qpm: 80}""")
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 3.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.5), ('G7', 3.0), ('F', 4.8)])

    expected_subsequence_1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        time_signatures: {
          time: 2.0
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_1, 0,
        [(12, 100, 0.01, 3.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_1, [('C', 1.5), ('G7', 3.0)])
    expected_subsequence_1.total_time = 4.01
    expected_subsequence_1.subsequence_info.end_time_offset = 0.99

    expected_subsequence_2 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 80}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_2, 0, [(52, 99, 0.5, 0.75)])
    testing_lib.add_chords_to_sequence(expected_subsequence_2, [
        ('G7', 0.0), ('F', 0.55)])
    expected_subsequence_2.total_time = 0.75
    expected_subsequence_2.subsequence_info.start_time_offset = 4.25

    subsequences = sequences_lib.split_note_sequence_on_time_changes(
        sequence, skip_splits_inside_notes=True)
    self.assertEquals(2, len(subsequences))
    self.assertProtoEquals(expected_subsequence_1, subsequences[0])
    self.assertProtoEquals(expected_subsequence_2, subsequences[1])

  def testSplitNoteSequenceMultipleTimeChanges(self):
    # Tests splitting a NoteSequence on time changes, truncating notes on splits
    # that occur inside notes.
    sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        time_signatures: {
          time: 2.0
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 60}
        tempos: {
          time: 4.25
          qpm: 80}""")
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.5), ('G7', 3.0), ('F', 4.8)])

    expected_subsequence_1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_1, 0,
        [(12, 100, 0.01, 2.0), (11, 55, 0.22, 0.50)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_1, [('C', 1.5)])
    expected_subsequence_1.total_time = 2.0
    expected_subsequence_1.subsequence_info.end_time_offset = 8.0

    expected_subsequence_2 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_2, 0,
        [(40, 45, 0.50, 1.50), (55, 120, 2.0, 2.01)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_2, [('C', 0.0), ('G7', 1.0)])
    expected_subsequence_2.total_time = 2.01
    expected_subsequence_2.subsequence_info.start_time_offset = 2.0
    expected_subsequence_2.subsequence_info.end_time_offset = 5.99

    expected_subsequence_3 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 80}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_3, 0,
        [(52, 99, 0.5, 0.75)])
    testing_lib.add_chords_to_sequence(
        expected_subsequence_3, [('G7', 0.0), ('F', 0.55)])
    expected_subsequence_3.total_time = 0.75
    expected_subsequence_3.subsequence_info.start_time_offset = 4.25
    expected_subsequence_3.subsequence_info.end_time_offset = 5.0

    subsequences = sequences_lib.split_note_sequence_on_time_changes(sequence)
    self.assertEquals(3, len(subsequences))
    self.assertProtoEquals(expected_subsequence_1, subsequences[0])
    self.assertProtoEquals(expected_subsequence_2, subsequences[1])
    self.assertProtoEquals(expected_subsequence_3, subsequences[2])

  def testSplitNoteSequenceWithStatelessEvents(self):
    # Tests splitting a NoteSequence at specified times with stateless events.
    sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 8.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_beats_to_sequence(sequence, [1.0, 2.0, 4.0])

    expected_subsequence_1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_1, 0,
        [(12, 100, 0.01, 3.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.0)])
    testing_lib.add_beats_to_sequence(expected_subsequence_1, [1.0, 2.0])
    expected_subsequence_1.total_time = 3.0
    expected_subsequence_1.subsequence_info.end_time_offset = 5.0

    expected_subsequence_2 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    expected_subsequence_2.total_time = 0.0
    expected_subsequence_2.subsequence_info.start_time_offset = 3.0
    expected_subsequence_2.subsequence_info.end_time_offset = 5.0

    expected_subsequence_3 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        expected_subsequence_3, 0,
        [(55, 120, 0.0, 0.01), (52, 99, 0.75, 1.0)])
    testing_lib.add_beats_to_sequence(expected_subsequence_3, [0.0])
    expected_subsequence_3.total_time = 1.0
    expected_subsequence_3.subsequence_info.start_time_offset = 4.0
    expected_subsequence_3.subsequence_info.end_time_offset = 3.0

    subsequences = sequences_lib.split_note_sequence(
        sequence, hop_size_seconds=[3.0, 4.0])
    self.assertEquals(3, len(subsequences))
    self.assertProtoEquals(expected_subsequence_1, subsequences[0])
    self.assertProtoEquals(expected_subsequence_2, subsequences[1])
    self.assertProtoEquals(expected_subsequence_3, subsequences[2])

  def testQuantizeNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('B7', 0.22), ('Em9', 4.0)])
    testing_lib.add_control_changes_to_sequence(
        self.note_sequence, 0,
        [(2.0, 64, 127), (4.0, 64, 0)])

    expected_quantized_sequence = copy.deepcopy(self.note_sequence)
    expected_quantized_sequence.quantization_info.steps_per_quarter = (
        self.steps_per_quarter)
    testing_lib.add_quantized_steps_to_sequence(
        expected_quantized_sequence,
        [(0, 40), (1, 2), (10, 14), (16, 17), (19, 20)])
    testing_lib.add_quantized_chord_steps_to_sequence(
        expected_quantized_sequence, [1, 16])
    testing_lib.add_quantized_control_steps_to_sequence(
        expected_quantized_sequence, [8, 16])

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=self.steps_per_quarter)

    self.assertProtoEquals(expected_quantized_sequence, quantized_sequence)

  def testQuantizeNoteSequenceAbsolute(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        self.note_sequence,
        [('B7', 0.22), ('Em9', 4.0)])
    testing_lib.add_control_changes_to_sequence(
        self.note_sequence, 0,
        [(2.0, 64, 127), (4.0, 64, 0)])

    expected_quantized_sequence = copy.deepcopy(self.note_sequence)
    expected_quantized_sequence.quantization_info.steps_per_second = 4
    testing_lib.add_quantized_steps_to_sequence(
        expected_quantized_sequence,
        [(0, 40), (1, 2), (10, 14), (16, 17), (19, 20)])
    testing_lib.add_quantized_chord_steps_to_sequence(
        expected_quantized_sequence, [1, 16])
    testing_lib.add_quantized_control_steps_to_sequence(
        expected_quantized_sequence, [8, 16])

    quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
        self.note_sequence, steps_per_second=4)

    self.assertProtoEquals(expected_quantized_sequence, quantized_sequence)

  def testAssertIsQuantizedNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])

    relative_quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=self.steps_per_quarter)
    absolute_quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
        self.note_sequence, steps_per_second=4)

    sequences_lib.assert_is_quantized_sequence(relative_quantized_sequence)
    sequences_lib.assert_is_quantized_sequence(absolute_quantized_sequence)
    with self.assertRaises(sequences_lib.QuantizationStatusException):
      sequences_lib.assert_is_quantized_sequence(self.note_sequence)

  def testAssertIsRelativeQuantizedNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])

    relative_quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=self.steps_per_quarter)
    absolute_quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
        self.note_sequence, steps_per_second=4)

    sequences_lib.assert_is_relative_quantized_sequence(
        relative_quantized_sequence)
    with self.assertRaises(sequences_lib.QuantizationStatusException):
      sequences_lib.assert_is_relative_quantized_sequence(
          absolute_quantized_sequence)
    with self.assertRaises(sequences_lib.QuantizationStatusException):
      sequences_lib.assert_is_relative_quantized_sequence(self.note_sequence)

  def testQuantizeNoteSequence_TimeSignatureChange(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    del self.note_sequence.time_signatures[:]
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # Single time signature.
    self.note_sequence.time_signatures.add(numerator=4, denominator=4, time=0)
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # Multiple time signatures with no change.
    self.note_sequence.time_signatures.add(numerator=4, denominator=4, time=1)
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # Time signature change.
    self.note_sequence.time_signatures.add(numerator=2, denominator=4, time=2)
    with self.assertRaises(sequences_lib.MultipleTimeSignatureException):
      sequences_lib.quantize_note_sequence(
          self.note_sequence, self.steps_per_quarter)

  def testQuantizeNoteSequence_ImplicitTimeSignatureChange(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    del self.note_sequence.time_signatures[:]

    # No time signature.
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # Implicit time signature change.
    self.note_sequence.time_signatures.add(numerator=2, denominator=4, time=2)
    with self.assertRaises(sequences_lib.MultipleTimeSignatureException):
      sequences_lib.quantize_note_sequence(
          self.note_sequence, self.steps_per_quarter)

  def testQuantizeNoteSequence_NoImplicitTimeSignatureChangeOutOfOrder(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    del self.note_sequence.time_signatures[:]

    # No time signature.
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # No implicit time signature change, but time signatures are added out of
    # order.
    self.note_sequence.time_signatures.add(numerator=2, denominator=4, time=2)
    self.note_sequence.time_signatures.add(numerator=2, denominator=4, time=0)
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

  def testStepsPerQuarterToStepsPerSecond(self):
    self.assertEqual(
        4.0, sequences_lib.steps_per_quarter_to_steps_per_second(4, 60.0))

  def testQuantizeToStep(self):
    self.assertEqual(
        32, sequences_lib.quantize_to_step(8.0001, 4))
    self.assertEqual(
        34, sequences_lib.quantize_to_step(8.4999, 4))
    self.assertEqual(
        33, sequences_lib.quantize_to_step(8.4999, 4, quantize_cutoff=1.0))

  def testFromNoteSequence_TempoChange(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    del self.note_sequence.tempos[:]

    # No tempos.
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # Single tempo.
    self.note_sequence.tempos.add(qpm=60, time=0)
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # Multiple tempos with no change.
    self.note_sequence.tempos.add(qpm=60, time=1)
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # Tempo change.
    self.note_sequence.tempos.add(qpm=120, time=2)
    with self.assertRaises(sequences_lib.MultipleTempoException):
      sequences_lib.quantize_note_sequence(
          self.note_sequence, self.steps_per_quarter)

  def testFromNoteSequence_ImplicitTempoChange(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    del self.note_sequence.tempos[:]

    # No tempo.
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # Implicit tempo change.
    self.note_sequence.tempos.add(qpm=60, time=2)
    with self.assertRaises(sequences_lib.MultipleTempoException):
      sequences_lib.quantize_note_sequence(
          self.note_sequence, self.steps_per_quarter)

  def testFromNoteSequence_NoImplicitTempoChangeOutOfOrder(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    del self.note_sequence.tempos[:]

    # No tempo.
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    # No implicit tempo change, but tempos are added out of order.
    self.note_sequence.tempos.add(qpm=60, time=2)
    self.note_sequence.tempos.add(qpm=60, time=0)
    sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

  def testRounding(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 100, 0.01, 0.24), (11, 100, 0.22, 0.55), (40, 100, 0.50, 0.75),
         (41, 100, 0.689, 1.18), (44, 100, 1.19, 1.69), (55, 100, 4.0, 4.01)])

    expected_quantized_sequence = copy.deepcopy(self.note_sequence)
    expected_quantized_sequence.quantization_info.steps_per_quarter = (
        self.steps_per_quarter)
    testing_lib.add_quantized_steps_to_sequence(
        expected_quantized_sequence,
        [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (16, 17)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    self.assertProtoEquals(expected_quantized_sequence, quantized_sequence)

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

    expected_quantized_sequence = copy.deepcopy(self.note_sequence)
    expected_quantized_sequence.quantization_info.steps_per_quarter = (
        self.steps_per_quarter)
    testing_lib.add_quantized_steps_to_sequence(
        expected_quantized_sequence,
        [(4, 16), (4, 12), (4, 16), (8, 20), (4, 20), (8, 16), (12, 14)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    self.assertProtoEquals(expected_quantized_sequence, quantized_sequence)

  def testStepsPerBar(self):
    qns = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    self.assertEqual(16, sequences_lib.steps_per_bar_in_quantized_sequence(qns))

    self.note_sequence.time_signatures[0].numerator = 6
    self.note_sequence.time_signatures[0].denominator = 8
    qns = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    self.assertEqual(12.0,
                     sequences_lib.steps_per_bar_in_quantized_sequence(qns))

  def testStretchNoteSequence(self):
    expected_stretched_sequence = copy.deepcopy(self.note_sequence)
    expected_stretched_sequence.tempos[0].qpm = 40

    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.0, 10.0), (11, 55, 0.2, 0.5), (40, 45, 2.5, 3.5)])
    testing_lib.add_track_to_sequence(
        expected_stretched_sequence, 0,
        [(12, 100, 0.0, 15.0), (11, 55, 0.3, 0.75), (40, 45, 3.75, 5.25)])

    testing_lib.add_chords_to_sequence(
        self.note_sequence, [('B7', 0.5), ('Em9', 2.0)])
    testing_lib.add_chords_to_sequence(
        expected_stretched_sequence, [('B7', 0.75), ('Em9', 3.0)])

    stretched_sequence = sequences_lib.stretch_note_sequence(
        self.note_sequence, stretch_factor=1.5)
    self.assertProtoEquals(expected_stretched_sequence, stretched_sequence)

  def testApplySustainControlChanges(self):
    """Verify sustain controls extend notes until the end of the control."""
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_control_changes_to_sequence(
        sequence, 0,
        [(0.0, 64, 127), (0.75, 64, 0), (2.0, 64, 127), (3.0, 64, 0),
         (3.75, 64, 127), (4.5, 64, 127), (4.8, 64, 0), (4.9, 64, 127),
         (6.0, 64, 0)])
    testing_lib.add_track_to_sequence(
        sequence, 1,
        [(12, 100, 0.01, 10.0), (52, 99, 4.75, 5.0)])
    expected_sequence = copy.copy(sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50), (55, 120, 4.0, 4.01)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(11, 55, 0.22, 0.75), (40, 45, 2.50, 3.50), (55, 120, 4.0, 4.8)])

    sus_sequence = sequences_lib.apply_sustain_control_changes(sequence)
    self.assertProtoEquals(expected_sequence, sus_sequence)

  def testApplySustainControlChangesWithRepeatedNotes(self):
    """Verify that sustain control handles repeated notes correctly.

    For example, a single pitch played before sustain:
    x-- x-- x--
    After sustain:
    x---x---x--

    Notes should be extended until either the end of the sustain control or the
    beginning of another note of the same pitch.
    """
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_control_changes_to_sequence(
        sequence, 0,
        [(1.0, 64, 127), (4.0, 64, 0)])
    expected_sequence = copy.copy(sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.25, 1.50), (60, 100, 1.25, 1.50), (72, 100, 2.00, 3.50),
         (60, 100, 2.0, 3.00), (60, 100, 3.50, 4.50)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(60, 100, 0.25, 1.25), (60, 100, 1.25, 2.00), (72, 100, 2.00, 4.00),
         (60, 100, 2.0, 3.50), (60, 100, 3.50, 4.50)])

    sus_sequence = sequences_lib.apply_sustain_control_changes(sequence)
    self.assertProtoEquals(expected_sequence, sus_sequence)

  def testApplySustainControlChangesWithRepeatedNotesBeforeSustain(self):
    """Repeated notes before sustain can overlap and should not be modified.

    Once a repeat happens within the sustain, any active notes should end
    before the next one starts.

    This is kind of an edge case because a note overlapping a note of the same
    pitch may not make sense, but apply_sustain_control_changes tries not to
    modify events that happen outside of a sustain.
    """
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_control_changes_to_sequence(
        sequence, 0,
        [(1.0, 64, 127), (4.0, 64, 0)])
    expected_sequence = copy.copy(sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.25, 1.50), (60, 100, .50, 1.50), (60, 100, 1.25, 2.0)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(60, 100, 0.25, 1.25), (60, 100, 0.50, 1.25), (60, 100, 1.25, 4.00)])

    sus_sequence = sequences_lib.apply_sustain_control_changes(sequence)
    self.assertProtoEquals(expected_sequence, sus_sequence)

  def testApplySustainControlChangesSimultaneousOnOff(self):
    """Test sustain on and off events happening at the same time.

    The off event should be processed last, so this should be a no-op.
    """
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_control_changes_to_sequence(
        sequence, 0, [(1.0, 64, 127), (1.0, 64, 0)])
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.50, 1.50), (60, 100, 2.0, 3.0)])

    sus_sequence = sequences_lib.apply_sustain_control_changes(sequence)
    self.assertProtoEquals(sequence, sus_sequence)

  def testApplySustainControlChangesExtendNotesToEnd(self):
    """Test sustain control extending the duration of the final note."""
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_control_changes_to_sequence(
        sequence, 0, [(1.0, 64, 127), (4.0, 64, 0)])
    expected_sequence = copy.copy(sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.50, 1.50), (72, 100, 2.0, 3.0)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(60, 100, 0.50, 4.00), (72, 100, 2.0, 4.0)])
    expected_sequence.total_time = 4.0

    sus_sequence = sequences_lib.apply_sustain_control_changes(sequence)
    self.assertProtoEquals(expected_sequence, sus_sequence)

  def testApplySustainControlChangesExtraneousSustain(self):
    """Test applying extraneous sustain control at the end of the sequence."""
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_control_changes_to_sequence(
        sequence, 0, [(4.0, 64, 127), (5.0, 64, 0)])
    expected_sequence = copy.copy(sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.50, 1.50), (72, 100, 2.0, 3.0)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(60, 100, 0.50, 1.50), (72, 100, 2.0, 3.0)])
    # The total_time field only takes *notes* into account, and should not be
    # affected by a sustain-on event beyond the last note.
    expected_sequence.total_time = 3.0

    sus_sequence = sequences_lib.apply_sustain_control_changes(sequence)
    self.assertProtoEquals(expected_sequence, sus_sequence)

  def testApplySustainControlChangesWithIdenticalNotes(self):
    """In the case of identical notes, one should be dropped.

    This is an edge case because in most cases, the same pitch should not sound
    twice at the same time on one instrument.
    """
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_control_changes_to_sequence(
        sequence, 0,
        [(1.0, 64, 127), (4.0, 64, 0)])
    expected_sequence = copy.copy(sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 2.00, 2.50), (60, 100, 2.00, 2.50)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(60, 100, 2.00, 4.00)])

    sus_sequence = sequences_lib.apply_sustain_control_changes(sequence)
    self.assertProtoEquals(expected_sequence, sus_sequence)

  def testInferDenseChordsForSequence(self):
    # Test non-quantized sequence.
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 1.0, 3.0), (64, 100, 1.0, 2.0), (67, 100, 1.0, 2.0),
         (65, 100, 2.0, 3.0), (69, 100, 2.0, 3.0),
         (62, 100, 3.0, 5.0), (65, 100, 3.0, 4.0), (69, 100, 3.0, 4.0)])
    expected_sequence = copy.copy(sequence)
    testing_lib.add_chords_to_sequence(
        expected_sequence, [('C', 1.0), ('F/C', 2.0), ('Dm', 3.0)])
    sequences_lib.infer_dense_chords_for_sequence(sequence)
    self.assertProtoEquals(expected_sequence, sequence)

    # Test quantized sequence.
    sequence = copy.copy(self.note_sequence)
    sequence.quantization_info.steps_per_quarter = 1
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 1.1, 3.0), (64, 100, 1.0, 1.9), (67, 100, 1.0, 2.0),
         (65, 100, 2.0, 3.2), (69, 100, 2.1, 3.1),
         (62, 100, 2.9, 4.8), (65, 100, 3.0, 4.0), (69, 100, 3.0, 4.1)])
    testing_lib.add_quantized_steps_to_sequence(
        sequence,
        [(1, 3), (1, 2), (1, 2), (2, 3), (2, 3), (3, 5), (3, 4), (3, 4)])
    expected_sequence = copy.copy(sequence)
    testing_lib.add_chords_to_sequence(
        expected_sequence, [('C', 1.0), ('F/C', 2.0), ('Dm', 3.0)])
    testing_lib.add_quantized_chord_steps_to_sequence(
        expected_sequence, [1, 2, 3])
    sequences_lib.infer_dense_chords_for_sequence(sequence)
    self.assertProtoEquals(expected_sequence, sequence)

  def testShiftSequenceTimes(self):
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 1.5), ('G7', 3.0), ('F', 4.8)])
    testing_lib.add_control_changes_to_sequence(
        sequence, 0,
        [(0.0, 64, 127), (2.0, 64, 0), (4.0, 64, 127), (5.0, 64, 0)])
    testing_lib.add_control_changes_to_sequence(
        sequence, 1, [(2.0, 64, 127)])
    testing_lib.add_pitch_bends_to_sequence(
        sequence, 1, 1, [(2.0, 100), (3.0, 0)])

    expected_sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(12, 100, 1.01, 11.0), (11, 55, 1.22, 1.50), (40, 45, 3.50, 4.50),
         (55, 120, 5.0, 5.01), (52, 99, 5.75, 6.0)])
    testing_lib.add_chords_to_sequence(
        expected_sequence, [('C', 2.5), ('G7', 4.0), ('F', 5.8)])
    testing_lib.add_control_changes_to_sequence(
        expected_sequence, 0,
        [(1.0, 64, 127), (3.0, 64, 0), (5.0, 64, 127), (6.0, 64, 0)])
    testing_lib.add_control_changes_to_sequence(
        expected_sequence, 1, [(3.0, 64, 127)])
    testing_lib.add_pitch_bends_to_sequence(
        expected_sequence, 1, 1, [(3.0, 100), (4.0, 0)])

    expected_sequence.time_signatures[0].time = 1
    expected_sequence.tempos[0].time = 1

    shifted_sequence = sequences_lib.shift_sequence_times(sequence, 1.0)
    self.assertProtoEquals(expected_sequence, shifted_sequence)

  def testConcatenateSequences(self):
    sequence1 = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence1, 0,
        [(60, 100, 0.0, 1.0), (72, 100, 0.5, 1.5)])
    sequence2 = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence2, 0,
        [(59, 100, 0.0, 1.0), (71, 100, 0.5, 1.5)])

    expected_sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(60, 100, 0.0, 1.0), (72, 100, 0.5, 1.5),
         (59, 100, 1.5, 2.5), (71, 100, 2.0, 3.0)])

    cat_seq = sequences_lib.concatenate_sequences([sequence1, sequence2])
    self.assertProtoEquals(expected_sequence, cat_seq)

  def testConcatenateSequencesWithSpecifiedDurations(self):
    sequence1 = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence1, 0, [(60, 100, 0.0, 1.0), (72, 100, 0.5, 1.5)])
    sequence2 = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence2, 0,
        [(59, 100, 0.0, 1.0)])
    sequence3 = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence3, 0,
        [(72, 100, 0.0, 1.0), (73, 100, 0.5, 1.5)])

    expected_sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(60, 100, 0.0, 1.0), (72, 100, 0.5, 1.5),
         (59, 100, 2.0, 3.0),
         (72, 100, 3.5, 4.5), (73, 100, 4.0, 5.0)])

    cat_seq = sequences_lib.concatenate_sequences(
        [sequence1, sequence2, sequence3],
        sequence_durations=[2, 1.5, 2])
    self.assertProtoEquals(expected_sequence, cat_seq)

  def testRemoveRedundantData(self):
    sequence = copy.copy(self.note_sequence)
    redundant_tempo = sequence.tempos.add()
    redundant_tempo.CopyFrom(sequence.tempos[0])
    redundant_tempo.time = 5.0
    sequence.sequence_metadata.composers.append('Foo')
    sequence.sequence_metadata.composers.append('Bar')
    sequence.sequence_metadata.composers.append('Foo')
    sequence.sequence_metadata.composers.append('Bar')
    sequence.sequence_metadata.genre.append('Classical')
    sequence.sequence_metadata.genre.append('Classical')

    fixed_sequence = sequences_lib.remove_redundant_data(sequence)

    expected_sequence = copy.copy(self.note_sequence)
    expected_sequence.sequence_metadata.composers.append('Foo')
    expected_sequence.sequence_metadata.composers.append('Bar')
    expected_sequence.sequence_metadata.genre.append('Classical')

    self.assertProtoEquals(expected_sequence, fixed_sequence)

  def testRemoveRedundantDataOutOfOrder(self):
    sequence = copy.copy(self.note_sequence)
    meaningful_tempo = sequence.tempos.add()
    meaningful_tempo.time = 5.0
    meaningful_tempo.qpm = 50
    redundant_tempo = sequence.tempos.add()
    redundant_tempo.CopyFrom(sequence.tempos[0])

    expected_sequence = copy.copy(self.note_sequence)
    expected_meaningful_tempo = expected_sequence.tempos.add()
    expected_meaningful_tempo.time = 5.0
    expected_meaningful_tempo.qpm = 50

    fixed_sequence = sequences_lib.remove_redundant_data(sequence)
    self.assertProtoEquals(expected_sequence, fixed_sequence)

  def testExpandSectionGroups(self):
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.0, 1.0), (72, 100, 1.0, 2.0),
         (59, 100, 2.0, 3.0), (71, 100, 3.0, 4.0)])
    sequence.section_annotations.add(time=0, section_id=0)
    sequence.section_annotations.add(time=1, section_id=1)
    sequence.section_annotations.add(time=2, section_id=2)
    sequence.section_annotations.add(time=3, section_id=3)

    # A((BC)2D)2
    sg = sequence.section_groups.add()
    sg.sections.add(section_id=0)
    sg.num_times = 1
    sg = sequence.section_groups.add()
    sg.sections.add(section_group=music_pb2.NoteSequence.SectionGroup(
        sections=[music_pb2.NoteSequence.Section(section_id=1),
                  music_pb2.NoteSequence.Section(section_id=2)],
        num_times=2))
    sg.sections.add(section_id=3)
    sg.num_times = 2

    expanded = sequences_lib.expand_section_groups(sequence)

    expected = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        expected, 0,
        [(60, 100, 0.0, 1.0),
         (72, 100, 1.0, 2.0),
         (59, 100, 2.0, 3.0),
         (72, 100, 3.0, 4.0),
         (59, 100, 4.0, 5.0),
         (71, 100, 5.0, 6.0),
         (72, 100, 6.0, 7.0),
         (59, 100, 7.0, 8.0),
         (72, 100, 8.0, 9.0),
         (59, 100, 9.0, 10.0),
         (71, 100, 10.0, 11.0)])
    expected.section_annotations.add(time=0, section_id=0)
    expected.section_annotations.add(time=1, section_id=1)
    expected.section_annotations.add(time=2, section_id=2)
    expected.section_annotations.add(time=3, section_id=1)
    expected.section_annotations.add(time=4, section_id=2)
    expected.section_annotations.add(time=5, section_id=3)
    expected.section_annotations.add(time=6, section_id=1)
    expected.section_annotations.add(time=7, section_id=2)
    expected.section_annotations.add(time=8, section_id=1)
    expected.section_annotations.add(time=9, section_id=2)
    expected.section_annotations.add(time=10, section_id=3)
    self.assertProtoEquals(expected, expanded)

  def testExpandWithoutSectionGroups(self):
    sequence = copy.copy(self.note_sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.0, 1.0), (72, 100, 1.0, 2.0),
         (59, 100, 2.0, 3.0), (71, 100, 3.0, 4.0)])
    sequence.section_annotations.add(time=0, section_id=0)
    sequence.section_annotations.add(time=1, section_id=1)
    sequence.section_annotations.add(time=2, section_id=2)
    sequence.section_annotations.add(time=3, section_id=3)

    expanded = sequences_lib.expand_section_groups(sequence)

    self.assertEqual(sequence, expanded)

if __name__ == '__main__':
  tf.test.main()
