# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for melody inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from magenta.music import melody_inference
from magenta.music import testing_lib
from magenta.protobuf import music_pb2


class MelodyInferenceTest(tf.test.TestCase):

  def testSequenceNoteFrames(self):
    sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.5, 2.0), (62, 100, 1.0, 1.25)])

    pitches, has_onsets, has_notes, event_times = (
        melody_inference.sequence_note_frames(sequence))

    expected_pitches = [60, 62]
    expected_has_onsets = [[0, 0], [1, 0], [0, 1], [0, 0]]
    expected_has_notes = [[0, 0], [1, 0], [1, 1], [1, 0]]
    expected_event_times = [0.5, 1.0, 1.25]

    self.assertEqual(expected_pitches, pitches)
    self.assertEqual(expected_has_onsets, has_onsets.tolist())
    self.assertEqual(expected_has_notes, has_notes.tolist())
    self.assertEqual(expected_event_times, event_times)

  def testMelodyInferenceEmptySequence(self):
    sequence = music_pb2.NoteSequence()
    melody_inference.infer_melody_for_sequence(sequence)
    expected_sequence = music_pb2.NoteSequence()
    self.assertEqual(expected_sequence, sequence)

  def testMelodyInferenceSingleNote(self):
    sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        sequence, 0, [(60, 100, 0.5, 1.0)])

    melody_inference.infer_melody_for_sequence(sequence)

    expected_sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        expected_sequence, 0, [(60, 100, 0.5, 1.0)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 1, [(60, 127, 0.5, 1.0)])

    self.assertEqual(expected_sequence, sequence)

  def testMelodyInferenceMonophonic(self):
    sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.5, 1.0), (62, 100, 1.0, 2.0), (64, 100, 2.0, 4.0)])

    melody_inference.infer_melody_for_sequence(sequence)

    expected_sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(60, 100, 0.5, 1.0), (62, 100, 1.0, 2.0), (64, 100, 2.0, 4.0)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 1,
        [(60, 127, 0.5, 1.0), (62, 127, 1.0, 2.0), (64, 127, 2.0, 4.0)])

    self.assertEqual(expected_sequence, sequence)

  def testMelodyInferencePolyphonic(self):
    sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        sequence, 0, [
            (36, 100, 0.0, 4.0), (64, 100, 0.0, 1.0), (67, 100, 0.0, 1.0),
            (65, 100, 1.0, 2.0), (69, 100, 1.0, 2.0),
            (67, 100, 2.0, 4.0), (71, 100, 2.0, 3.0),
            (72, 100, 3.0, 4.0)
        ])

    melody_inference.infer_melody_for_sequence(sequence)

    expected_sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        expected_sequence, 0, [
            (36, 100, 0.0, 4.0), (64, 100, 0.0, 1.0), (67, 100, 0.0, 1.0),
            (65, 100, 1.0, 2.0), (69, 100, 1.0, 2.0),
            (67, 100, 2.0, 4.0), (71, 100, 2.0, 3.0),
            (72, 100, 3.0, 4.0)
        ])
    testing_lib.add_track_to_sequence(
        expected_sequence, 1, [
            (67, 127, 0.0, 1.0), (69, 127, 1.0, 2.0),
            (71, 127, 2.0, 3.0), (72, 127, 3.0, 4.0)
        ])

    self.assertEqual(expected_sequence, sequence)


if __name__ == '__main__':
  tf.test.main()
