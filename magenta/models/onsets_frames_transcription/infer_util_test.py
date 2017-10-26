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

"""Tests for infer_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports

import numpy as np
import tensorflow as tf

from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import infer_util
from magenta.protobuf import music_pb2

DEFAULT_FRAMES_PER_SECOND = 16000 / 512


class InferUtilTest(tf.test.TestCase):

  def testSequenceToValuedIntervals(self):
    sequence = music_pb2.NoteSequence()
    sequence.notes.add(pitch=60, start_time=1.0, end_time=2.0)
    # Should be dropped because it is 0 duration.
    sequence.notes.add(pitch=60, start_time=3.0, end_time=3.0)

    intervals, pitches = infer_util.sequence_to_valued_intervals(
        sequence, min_duration_ms=0)
    np.testing.assert_array_equal([[1., 2.]], intervals)
    np.testing.assert_array_equal([60], pitches)

  def testPianorollToNoteSequence(self):
    # 100 frames of notes.
    frames = np.zeros((100, constants.MIDI_PITCHES), np.bool)
    # Activate key 39 for the middle 50 frames.
    frames[25:75, 39] = True
    sequence = infer_util.pianoroll_to_note_sequence(
        frames,
        frames_per_second=DEFAULT_FRAMES_PER_SECOND,
        min_duration_ms=0)
    self.assertEqual(1, len(sequence.notes))
    self.assertEqual(60, sequence.notes[0].pitch)
    self.assertEqual(25 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[0].start_time)
    self.assertEqual(75 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[0].end_time)

  def testPianorollToNoteSequenceWithOnsets(self):
    # 100 frames of notes and onsets.
    frames = np.zeros((100, constants.MIDI_PITCHES), np.bool)
    onsets = np.zeros((100, constants.MIDI_PITCHES), np.bool)
    # Activate key 39 for the middle 50 frames and last 10 frames.
    frames[25:75, 39] = True
    frames[90:100, 39] = True
    # Add an onset for the first occurrence.
    onsets[25, 39] = True
    # Add an onset for a note that doesn't have an active frame.
    onsets[80, 49] = True
    sequence = infer_util.pianoroll_to_note_sequence(
        frames,
        frames_per_second=DEFAULT_FRAMES_PER_SECOND,
        min_duration_ms=0,
        onset_predictions=onsets)
    self.assertEqual(2, len(sequence.notes))

    self.assertEqual(60, sequence.notes[0].pitch)
    self.assertEqual(25 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[0].start_time)
    self.assertEqual(75 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[0].end_time)

    self.assertEqual(70, sequence.notes[1].pitch)
    self.assertEqual(80 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[1].start_time)
    self.assertEqual(81 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[1].end_time)

  def testPianorollToNoteSequenceWithOnsetsOverlappingFrames(self):
    # 100 frames of notes and onsets.
    frames = np.zeros((100, constants.MIDI_PITCHES), np.bool)
    onsets = np.zeros((100, constants.MIDI_PITCHES), np.bool)
    # Activate key 39 for the middle 50 frames.
    frames[25:75, 39] = True
    # Add multiple onsets within those frames.
    onsets[25, 39] = True
    onsets[30, 39] = True
    # If an onset lasts for multiple frames, it should create only 1 note.
    onsets[35, 39] = True
    onsets[36, 39] = True
    sequence = infer_util.pianoroll_to_note_sequence(
        frames,
        frames_per_second=DEFAULT_FRAMES_PER_SECOND,
        min_duration_ms=0,
        onset_predictions=onsets)
    self.assertEqual(3, len(sequence.notes))

    self.assertEqual(60, sequence.notes[0].pitch)
    self.assertEqual(25 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[0].start_time)
    self.assertEqual(30 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[0].end_time)

    self.assertEqual(60, sequence.notes[1].pitch)
    self.assertEqual(30 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[1].start_time)
    self.assertEqual(35 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[1].end_time)

    self.assertEqual(60, sequence.notes[2].pitch)
    self.assertEqual(35 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[2].start_time)
    self.assertEqual(75 / DEFAULT_FRAMES_PER_SECOND,
                     sequence.notes[2].end_time)


if __name__ == '__main__':
  tf.test.main()
