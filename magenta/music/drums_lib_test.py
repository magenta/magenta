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
"""Tests for drums_lib."""

# internal imports
import tensorflow as tf

from magenta.music import drums_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib

DRUMS = lambda *args: frozenset(args)
NO_DRUMS = frozenset()


class DrumsLibTest(tf.test.TestCase):

  def setUp(self):
    self.quantized_sequence = sequences_lib.QuantizedSequence()
    self.quantized_sequence.qpm = 60.0
    self.quantized_sequence.steps_per_quarter = 4

  def testFromQuantizedSequence(self):
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 0, 40), (11, 55, 1, 2), (40, 45, 10, 14),
         (55, 120, 16, 17), (60, 100, 16, 22), (52, 99, 19, 20)],
        is_drum=True)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(self.quantized_sequence, start_step=0)
    expected = ([DRUMS(12), DRUMS(11), NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(40), NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(55, 60),
                 NO_DRUMS, NO_DRUMS, DRUMS(52)])
    self.assertEqual(expected, list(drums))
    self.assertEqual(16, drums.steps_per_bar)

  def testFromQuantizedSequenceMultipleTracks(self):
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 0, 40), (40, 45, 10, 14), (60, 100, 16, 22)],
        is_drum=True)
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(11, 55, 1, 2), (55, 120, 16, 17), (52, 99, 19, 20)],
        is_drum=True)
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 2,
        [(13, 100, 0, 40), (14, 45, 10, 14), (15, 100, 16, 22)])
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(self.quantized_sequence, start_step=0)
    expected = ([DRUMS(12), DRUMS(11), NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(40), NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(55, 60),
                 NO_DRUMS, NO_DRUMS, DRUMS(52)])
    self.assertEqual(expected, list(drums))
    self.assertEqual(16, drums.steps_per_bar)

  def testFromQuantizedSequenceNotCommonTimeSig(self):
    self.quantized_sequence.time_signature = (
        sequences_lib.QuantizedSequence.TimeSignature(numerator=7,
                                                      denominator=8))
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 0, 40), (11, 55, 1, 2), (40, 45, 10, 14),
         (30, 80, 10, 11), (55, 120, 16, 17), (52, 99, 19, 20)],
        is_drum=True)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(self.quantized_sequence, start_step=0)
    expected = ([DRUMS(12), DRUMS(11), NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(30, 40),
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(55),
                 NO_DRUMS, NO_DRUMS, DRUMS(52)])
    self.assertEqual(expected, list(drums))
    self.assertEqual(14, drums.steps_per_bar)

  def testFromNotesTrimEmptyMeasures(self):
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 6, 7), (11, 100, 8, 9)],
        is_drum=True)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(self.quantized_sequence, start_step=0)
    expected = [NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS,
                DRUMS(12), NO_DRUMS, DRUMS(11)]
    self.assertEqual(expected, list(drums))
    self.assertEqual(16, drums.steps_per_bar)

  def testFromNotesStepsPerBar(self):
    self.quantized_sequence.time_signature = (
        sequences_lib.QuantizedSequence.TimeSignature(numerator=7,
                                                      denominator=8))
    self.quantized_sequence.steps_per_quarter = 12
    self.quantized_sequence.tracks[0] = []
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(self.quantized_sequence, start_step=0)
    self.assertEqual(42, drums.steps_per_bar)

  def testFromNotesStartAndEndStep(self):
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 4, 8), (11, 100, 9, 10), (13, 100, 13, 15),
         (14, 100, 19, 20), (15, 100, 21, 27)],
        is_drum=True)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(self.quantized_sequence, start_step=18)
    expected = [NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(14), NO_DRUMS, DRUMS(15)]
    self.assertEqual(expected, list(drums))
    self.assertEqual(16, drums.start_step)
    self.assertEqual(22, drums.end_step)

  def testSetLength(self):
    events = [DRUMS(60)]
    drums = drums_lib.DrumTrack(events, start_step=9)
    drums.set_length(5)
    self.assertListEqual([DRUMS(60), NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS],
                         list(drums))
    self.assertEquals(9, drums.start_step)
    self.assertEquals(14, drums.end_step)

    drums = drums_lib.DrumTrack(events, start_step=9)
    drums.set_length(5, from_left=True)
    self.assertListEqual([NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(60)],
                         list(drums))
    self.assertEquals(5, drums.start_step)
    self.assertEquals(10, drums.end_step)

    events = [DRUMS(60), NO_DRUMS, NO_DRUMS, NO_DRUMS]
    drums = drums_lib.DrumTrack(events)
    drums.set_length(3)
    self.assertListEqual([DRUMS(60), NO_DRUMS, NO_DRUMS], list(drums))
    self.assertEquals(0, drums.start_step)
    self.assertEquals(3, drums.end_step)

    drums = drums_lib.DrumTrack(events)
    drums.set_length(3, from_left=True)
    self.assertListEqual([NO_DRUMS, NO_DRUMS, NO_DRUMS], list(drums))
    self.assertEquals(1, drums.start_step)
    self.assertEquals(4, drums.end_step)

  def testToSequenceSimple(self):
    drums = drums_lib.DrumTrack(
        [NO_DRUMS, DRUMS(1, 2), NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(2),
         DRUMS(3), NO_DRUMS, NO_DRUMS])
    sequence = drums.to_sequence(
        velocity=10,
        sequence_start_time=2,
        qpm=60.0)

    self.assertProtoEquals(
        'ticks_per_quarter: 220 '
        'tempos < qpm: 60.0 > '
        'total_time: 3.75 '
        'notes < '
        '  pitch: 1 velocity: 10 instrument: 9 start_time: 2.25 end_time: 2.5 '
        '  is_drum: true '
        '> '
        'notes < '
        '  pitch: 2 velocity: 10 instrument: 9 start_time: 2.25 end_time: 2.5 '
        '  is_drum: true '
        '> '
        'notes < '
        '  pitch: 2 velocity: 10 instrument: 9 start_time: 3.25 end_time: 3.5 '
        '  is_drum: true '
        '> '
        'notes < '
        '  pitch: 3 velocity: 10 instrument: 9 start_time: 3.5 end_time: 3.75 '
        '  is_drum: true '
        '> ',
        sequence)

  def testToSequenceEndsWithNonzeroStart(self):
    drums = drums_lib.DrumTrack([NO_DRUMS, DRUMS(1), NO_DRUMS], start_step=4)
    sequence = drums.to_sequence(
        velocity=100,
        sequence_start_time=0.5,
        qpm=60.0)

    self.assertProtoEquals(
        'ticks_per_quarter: 220 '
        'tempos < qpm: 60.0 > '
        'total_time: 2.0 '
        'notes < '
        '  pitch: 1 velocity: 100 instrument: 9 start_time: 1.75 end_time: 2.0 '
        '  is_drum: true '
        '> ',
        sequence)

  def testToSequenceEmpty(self):
    drums = drums_lib.DrumTrack()
    sequence = drums.to_sequence(
        velocity=10,
        sequence_start_time=2,
        qpm=60.0)

    self.assertProtoEquals(
        'ticks_per_quarter: 220 '
        'tempos < qpm: 60.0 > ',
        sequence)

  def testExtractDrumTracksSimple(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7)],
        is_drum=True)
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 9)],
        is_drum=True)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(11, 14)]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        self.quantized_sequence, min_bars=1, gap_bars=1)

    self.assertEqual(1, len(drum_tracks))
    self.assertTrue(isinstance(drum_tracks[0], drums_lib.DrumTrack))

    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)

  def testExtractMultipleDrumTracks(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)],
        is_drum=True)
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 37, 38)],
        is_drum=True)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(11, 14)],
                [NO_DRUMS, DRUMS(50), NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(52)]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        self.quantized_sequence, min_bars=1, gap_bars=2)
    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooShort(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 7)],
        is_drum=True)
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        self.quantized_sequence, min_bars=2, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual([], drum_tracks)

  def testExtractDrumTracksPadEnd(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 7)],
        is_drum=True)
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (15, 50, 6, 8)],
        is_drum=True)
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 2,
        [(12, 127, 2, 4), (16, 50, 8, 9)],
        is_drum=True)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(14, 15), NO_DRUMS, DRUMS(16), NO_DRUMS, NO_DRUMS,
                 NO_DRUMS]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        self.quantized_sequence, min_bars=1, gap_bars=1, pad_end=True)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooLongTruncate(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 15), (14, 50, 10, 15), (16, 100, 14, 19)],
        is_drum=True)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(14), NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(14), NO_DRUMS,
                 NO_DRUMS, NO_DRUMS]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        self.quantized_sequence, min_bars=1, max_steps_truncate=14, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooLongDiscard(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 15), (14, 50, 10, 15), (16, 100, 14, 19),
         (14, 100, 18, 19)],
        is_drum=True)
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        self.quantized_sequence, min_bars=1, max_steps_discard=18, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual([], drum_tracks)

  def testExtractDrumTracksLateStart(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 102, 103), (13, 100, 104, 106)],
        is_drum=True)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, DRUMS(13)]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        self.quantized_sequence, min_bars=1, gap_bars=1)
    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)


if __name__ == '__main__':
  tf.test.main()
