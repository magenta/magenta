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
"""Tests for melodies_lib."""

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import sequences_lib


NOTE_OFF = melodies_lib.NOTE_OFF
NO_EVENT = melodies_lib.NO_EVENT


def add_quantized_track(quantized_sequence, instrument, notes):
  if instrument not in quantized_sequence.tracks:
    quantized_sequence.tracks[instrument] = []
  track = quantized_sequence.tracks[instrument]
  for pitch, velocity, start_step, end_step in notes:
    note = sequences_lib.Note(pitch=pitch,
                              velocity=velocity,
                              start=start_step,
                              end=end_step,
                              instrument=instrument,
                              program=0)
    track.append(note)


class MelodiesLibTest(tf.test.TestCase):

  def setUp(self):
    self.quantized_sequence = sequences_lib.QuantizedSequence()
    self.quantized_sequence.bpm = 60.0
    self.quantized_sequence.steps_per_beat = 4

  def testGetNoteHistogram(self):
    events = [NO_EVENT, NOTE_OFF, 12 * 2 + 1, 12 * 3, 12 * 5 + 11, 12 * 6 + 3,
              12 * 4 + 11]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    expected = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    self.assertEqual(expected, list(melody.get_note_histogram()))

    events = [0, 1, NO_EVENT, NOTE_OFF, 12 * 2 + 1, 12 * 3, 12 * 6 + 3,
              12 * 5 + 11, NO_EVENT, 12 * 4 + 11, 12 * 7 + 1]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    expected = [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    self.assertEqual(expected, list(melody.get_note_histogram()))

    melody = melodies_lib.MonophonicMelody()
    expected = [0] * 12
    self.assertEqual(expected, list(melody.get_note_histogram()))

  def testGetMajorKey(self):
    # D Major.
    events = [NO_EVENT, 12 * 2 + 2, 12 * 3 + 4, 12 * 5 + 1, 12 * 6 + 6,
              12 * 4 + 11, 12 * 3 + 9, 12 * 5 + 7, NOTE_OFF]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    self.assertEqual(2, melody.get_major_key())

    # C# Major with accidentals.
    events = [NO_EVENT, 12 * 2 + 1, 12 * 4 + 8, 12 * 5 + 5, 12 * 6 + 6,
              12 * 3 + 3, 12 * 2 + 11, 12 * 3 + 10, 12 * 5, 12 * 2 + 8,
              12 * 4 + 1, 12 * 3 + 5, 12 * 5 + 9, 12 * 4 + 3, NOTE_OFF]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    self.assertEqual(1, melody.get_major_key())

    # One note in C Major.
    events = [NO_EVENT, 12 * 2 + 11, NOTE_OFF]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    self.assertEqual(0, melody.get_major_key())

  def testSquash(self):
    # Melody in C, transposed to C, and squashed to 1 octave.
    events = [12 * 5, NO_EVENT, 12 * 5 + 2, NOTE_OFF, 12 * 6 + 4, NO_EVENT]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    melody.squash(min_note=12 * 5, max_note=12 * 6, transpose_to_key=0)
    expected = [12 * 5, NO_EVENT, 12 * 5 + 2, NOTE_OFF, 12 * 5 + 4, NO_EVENT]
    self.assertEqual(expected, list(melody))

    # Melody in D, transposed to C, and squashed to 1 octave.
    events = [12 * 5 + 2, 12 * 5 + 4, 12 * 6 + 7, 12 * 6 + 6, 12 * 5 + 1]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    melody.squash(min_note=12 * 5, max_note=12 * 6, transpose_to_key=0)
    expected = [12 * 5, 12 * 5 + 2, 12 * 5 + 5, 12 * 5 + 4, 12 * 5 + 11]
    self.assertEqual(expected, list(melody))

    # Melody in D, transposed to E, and squashed to 1 octave.
    events = [12 * 5 + 2, 12 * 5 + 4, 12 * 6 + 7, 12 * 6 + 6, 12 * 4 + 11]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    melody.squash(min_note=12 * 5, max_note=12 * 6, transpose_to_key=4)
    expected = [12 * 5 + 4, 12 * 5 + 6, 12 * 5 + 9, 12 * 5 + 8, 12 * 5 + 1]
    self.assertEqual(expected, list(melody))

  def testSquashCenterOctaves(self):
    # Move up an octave.
    events = [12 * 4, NO_EVENT, 12 * 4 + 2, NOTE_OFF, 12 * 4 + 4, NO_EVENT,
              12 * 4 + 5, 12 * 5 + 2, 12 * 4 - 1, NOTE_OFF]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    melody.squash(min_note=12 * 4, max_note=12 * 7, transpose_to_key=0)
    expected = [12 * 5, NO_EVENT, 12 * 5 + 2, NOTE_OFF, 12 * 5 + 4, NO_EVENT,
                12 * 5 + 5, 12 * 6 + 2, 12 * 5 - 1, NOTE_OFF]
    self.assertEqual(expected, list(melody))

    # Move down an octave.
    events = [12 * 6, NO_EVENT, 12 * 6 + 2, NOTE_OFF, 12 * 6 + 4, NO_EVENT,
              12 * 6 + 5, 12 * 7 + 2, 12 * 6 - 1, NOTE_OFF]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    melody.squash(min_note=12 * 4, max_note=12 * 7, transpose_to_key=0)
    expected = [12 * 5, NO_EVENT, 12 * 5 + 2, NOTE_OFF, 12 * 5 + 4, NO_EVENT,
                12 * 5 + 5, 12 * 6 + 2, 12 * 5 - 1, NOTE_OFF]
    self.assertEqual(expected, list(melody))

  def testSquashMaxNote(self):
    events = [12 * 5, 12 * 5 + 2, 12 * 5 + 4, 12 * 5 + 5, 12 * 5 + 11, 12 * 6,
              12 * 6 + 1]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    melody.squash(min_note=12 * 5, max_note=12 * 6, transpose_to_key=0)
    expected = [12 * 5, 12 * 5 + 2, 12 * 5 + 4, 12 * 5 + 5, 12 * 5 + 11, 12 * 5,
                12 * 5 + 1]
    self.assertEqual(expected, list(melody))

  def testSquashAllNotesOff(self):
    events = [NO_EVENT, NOTE_OFF, NO_EVENT, NO_EVENT]
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(events)
    melody.squash(min_note=12 * 4, max_note=12 * 7, transpose_to_key=0)
    self.assertEqual(events, list(melody))

  def testFromQuantizedSequence(self):
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 0, 40), (11, 55, 1, 2), (40, 45, 10, 14),
         (55, 120, 16, 17), (52, 99, 19, 20)])
    melody = melodies_lib.MonophonicMelody()
    melody.from_quantized_sequence(self.quantized_sequence,
                                   start_step=0, track=0)
    expected = [12, 11, NOTE_OFF, NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT,
                NO_EVENT, NO_EVENT, NO_EVENT, 40, NO_EVENT, NO_EVENT, NO_EVENT,
                NOTE_OFF, NO_EVENT, 55, NOTE_OFF, NO_EVENT, 52, NOTE_OFF]
    self.assertEqual(expected, list(melody))
    self.assertEqual(16, melody.steps_per_bar)

  def testFromNotesPolyphonic(self):
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 4, 16), (19, 100, 4, 12)])
    melody = melodies_lib.MonophonicMelody()
    with self.assertRaises(melodies_lib.PolyphonicMelodyException):
      melody.from_quantized_sequence(self.quantized_sequence,
                                     start_step=0, track=0,
                                     ignore_polyphonic_notes=False)
    self.assertFalse(list(melody))

  def testFromNotesPolyphonicWithIgnorePolyphonicNotes(self):
    add_quantized_track(
        self.quantized_sequence, 0,
        [(21, 100, 0, 8), (19, 100, 0, 12)])
    melody = melodies_lib.MonophonicMelody()
    melody.from_quantized_sequence(self.quantized_sequence,
                                   start_step=0, track=0,
                                   ignore_polyphonic_notes=True)
    expected = [21] + [-2] * 7 + [-1]
    self.assertEqual(expected, list(melody))

  def testFromNotesChord(self):
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 4, 5), (19, 100, 4, 5),
         (20, 100, 4, 5), (25, 100, 4, 5)])
    melody = melodies_lib.MonophonicMelody()
    with self.assertRaises(melodies_lib.PolyphonicMelodyException):
      melody.from_quantized_sequence(self.quantized_sequence,
                                     start_step=0, track=0,
                                     ignore_polyphonic_notes=False)
    self.assertFalse(list(melody))

  def testFromNotesTrimEmptyMeasures(self):
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 6, 7), (11, 100, 8, 9)])
    melody = melodies_lib.MonophonicMelody()
    melody.from_quantized_sequence(self.quantized_sequence,
                                   start_step=0, track=0,
                                   ignore_polyphonic_notes=False)
    expected = [NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT, 12,
                NOTE_OFF, 11, NOTE_OFF]
    self.assertEqual(expected, list(melody))

  def testFromNotesTimeOverlap(self):
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 4, 8), (11, 100, 13, 15),
         (13, 100, 8, 16)])
    melody = melodies_lib.MonophonicMelody()
    melody.from_quantized_sequence(self.quantized_sequence,
                                   start_step=0, track=0,
                                   ignore_polyphonic_notes=False)
    expected = [NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT, 12, NO_EVENT, NO_EVENT,
                NO_EVENT, 13, NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT, 11,
                NO_EVENT, NOTE_OFF]
    self.assertEqual(expected, list(melody))

  def testFromNotesStepsPerBar(self):
    self.quantized_sequence.time_signature = sequences_lib.TimeSignature(7, 8)
    self.quantized_sequence.steps_per_beat = 12
    self.quantized_sequence.tracks[0] = []
    melody = melodies_lib.MonophonicMelody()
    melody.from_quantized_sequence(self.quantized_sequence,
                                   start_step=0, track=0,
                                   ignore_polyphonic_notes=False)
    self.assertEqual(42, melody.steps_per_bar)

  def testFromNotesStartAndEndStep(self):
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 4, 8), (11, 100, 9, 10), (13, 100, 13, 15),
         (14, 100, 19, 20), (15, 100, 21, 27)])
    melody = melodies_lib.MonophonicMelody()
    melody.from_quantized_sequence(self.quantized_sequence,
                                   start_step=18, track=0,
                                   ignore_polyphonic_notes=False)
    expected = [NO_EVENT, NO_EVENT, NO_EVENT, 14, NOTE_OFF, 15, NO_EVENT,
                NO_EVENT, NO_EVENT, NO_EVENT, NO_EVENT, NOTE_OFF]
    self.assertEqual(expected, list(melody))
    self.assertEqual(32, melody.end_step)

  def testExtractMelodiesSimple(self):
    self.quantized_sequence.steps_per_beat = 1
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7)])
    add_quantized_track(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8)])
    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 11,
                 NOTE_OFF],
                [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT, NOTE_OFF]]
    melodies = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True)

    self.assertEqual(2, len(melodies))
    self.assertTrue(isinstance(melodies[0], melodies_lib.MonophonicMelody))
    self.assertTrue(isinstance(melodies[1], melodies_lib.MonophonicMelody))

    melodies = sorted([list(melody) for melody in melodies])
    self.assertEqual(expected, melodies)

  def testExtractMultipleMelodiesFromSameTrack(self):
    self.quantized_sequence.steps_per_beat = 1
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7)])
    add_quantized_track(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 36)])
    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 11,
                 NOTE_OFF],
                [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT, NOTE_OFF],
                [NO_EVENT, 50, 52, NO_EVENT, NOTE_OFF]]
    melodies = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    melodies = sorted([list(melody) for melody in melodies])
    self.assertEqual(expected, melodies)

  def testExtractMelodiesMelodyTooShort(self):
    self.quantized_sequence.steps_per_beat = 1
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 7)])
    add_quantized_track(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8)])
    expected = [[NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14,
                 NO_EVENT, NOTE_OFF]]
    melodies = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=2, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    melodies = [list(melody) for melody in melodies]
    self.assertEqual(expected, melodies)

  def testExtractMelodiesTooFewPitches(self):
    # Test that extract_melodies discards melodies with too few pitches where
    # pitches are equivalent by octave.
    self.quantized_sequence.steps_per_beat = 1
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 0, 1), (13, 100, 1, 2), (18, 100, 2, 3),
         (24, 100, 3, 4), (25, 100, 4, 5)])
    add_quantized_track(
        self.quantized_sequence, 1,
        [(12, 100, 0, 1), (13, 100, 1, 2), (18, 100, 2, 3),
         (25, 100, 3, 4), (26, 100, 4, 5)])
    expected = [[12, 13, 18, 25, 26, NOTE_OFF]]
    melodies = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=1, min_unique_pitches=4,
        ignore_polyphonic_notes=True)
    melodies = [list(melody) for melody in melodies]
    self.assertEqual(expected, melodies)

  def testExtractMelodiesLateStart(self):
    self.quantized_sequence.steps_per_beat = 1
    add_quantized_track(
        self.quantized_sequence, 0,
        [(12, 100, 102, 103), (13, 100, 104, 106)])
    add_quantized_track(
        self.quantized_sequence, 1,
        [(12, 100, 100, 101), (13, 100, 102, 104)])
    expected = [[NO_EVENT, NO_EVENT, 12, NOTE_OFF, 13, NO_EVENT, NOTE_OFF],
                [12, NOTE_OFF, 13, NO_EVENT, NOTE_OFF]]
    melodies = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=1, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    melodies = sorted([list(melody) for melody in melodies])
    self.assertEqual(expected, melodies)

if __name__ == '__main__':
  tf.test.main()
