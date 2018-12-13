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

import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import drums_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2

DRUMS = lambda *args: frozenset(args)
NO_DRUMS = frozenset()


class DrumsLibTest(tf.test.TestCase):

  def setUp(self):
    self.steps_per_quarter = 4
    self.note_sequence = common_testing_lib.parse_test_proto(
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

  def testFromQuantizedNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.0, 10.0), (11, 55, 0.25, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.25), (60, 100, 4.0, 5.5), (52, 99, 4.75, 5.0)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(quantized_sequence, search_start_step=0)
    expected = ([DRUMS(12), DRUMS(11), NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(40), NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(55, 60),
                 NO_DRUMS, NO_DRUMS, DRUMS(52)])
    self.assertEqual(expected, list(drums))
    self.assertEqual(16, drums.steps_per_bar)

  def testFromQuantizedNoteSequenceMultipleTracks(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0, 10), (40, 45, 2.5, 3.5), (60, 100, 4, 5.5)],
        is_drum=True)
    testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(11, 55, .25, .5), (55, 120, 4, 4.25), (52, 99, 4.75, 5)],
        is_drum=True)
    testing_lib.add_track_to_sequence(
        self.note_sequence, 2,
        [(13, 100, 0, 10), (14, 45, 2.5, 3.5), (15, 100, 4, 5.5)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(quantized_sequence, search_start_step=0)
    expected = ([DRUMS(12), DRUMS(11), NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(40), NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(55, 60),
                 NO_DRUMS, NO_DRUMS, DRUMS(52)])
    self.assertEqual(expected, list(drums))
    self.assertEqual(16, drums.steps_per_bar)

  def testFromQuantizedNoteSequenceNotCommonTimeSig(self):
    self.note_sequence.time_signatures[0].numerator = 7
    self.note_sequence.time_signatures[0].denominator = 8

    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0, 10), (11, 55, .25, .5), (40, 45, 2.5, 3.5),
         (30, 80, 2.5, 2.75), (55, 120, 4, 4.25), (52, 99, 4.75, 5)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(quantized_sequence, search_start_step=0)
    expected = ([DRUMS(12), DRUMS(11), NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(30, 40),
                 NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(55),
                 NO_DRUMS, NO_DRUMS, DRUMS(52)])
    self.assertEqual(expected, list(drums))
    self.assertEqual(14, drums.steps_per_bar)

  def testFromNotesTrimEmptyMeasures(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 1.5, 1.75), (11, 100, 2, 2.25)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(quantized_sequence, search_start_step=0)
    expected = [NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS,
                DRUMS(12), NO_DRUMS, DRUMS(11)]
    self.assertEqual(expected, list(drums))
    self.assertEqual(16, drums.steps_per_bar)

  def testFromNotesStepsPerBar(self):
    self.note_sequence.time_signatures[0].numerator = 7
    self.note_sequence.time_signatures[0].denominator = 8

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=12)
    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(quantized_sequence, search_start_step=0)
    self.assertEqual(42, drums.steps_per_bar)

  def testFromNotesStartAndEndStep(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 1, 2), (11, 100, 2.25, 2.5), (13, 100, 3.25, 3.75),
         (14, 100, 8.75, 9), (15, 100, 9.25, 10.75)],
        is_drum=True)

    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, self.steps_per_quarter)

    drums = drums_lib.DrumTrack()
    drums.from_quantized_sequence(quantized_sequence, search_start_step=18)
    expected = [NO_DRUMS, DRUMS(14), NO_DRUMS, DRUMS(15)]
    self.assertEqual(expected, list(drums))
    self.assertEqual(34, drums.start_step)
    self.assertEqual(38, drums.end_step)

  def testSetLength(self):
    events = [DRUMS(60)]
    drums = drums_lib.DrumTrack(events, start_step=9)
    drums.set_length(5)
    self.assertListEqual([DRUMS(60), NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS],
                         list(drums))
    self.assertEqual(9, drums.start_step)
    self.assertEqual(14, drums.end_step)

    drums = drums_lib.DrumTrack(events, start_step=9)
    drums.set_length(5, from_left=True)
    self.assertListEqual([NO_DRUMS, NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(60)],
                         list(drums))
    self.assertEqual(5, drums.start_step)
    self.assertEqual(10, drums.end_step)

    events = [DRUMS(60), NO_DRUMS, NO_DRUMS, NO_DRUMS]
    drums = drums_lib.DrumTrack(events)
    drums.set_length(3)
    self.assertListEqual([DRUMS(60), NO_DRUMS, NO_DRUMS], list(drums))
    self.assertEqual(0, drums.start_step)
    self.assertEqual(3, drums.end_step)

    drums = drums_lib.DrumTrack(events)
    drums.set_length(3, from_left=True)
    self.assertListEqual([NO_DRUMS, NO_DRUMS, NO_DRUMS], list(drums))
    self.assertEqual(1, drums.start_step)
    self.assertEqual(4, drums.end_step)

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
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7)],
        is_drum=True)
    testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 9)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(11, 14)]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        quantized_sequence, min_bars=1, gap_bars=1)

    self.assertEqual(1, len(drum_tracks))
    self.assertTrue(isinstance(drum_tracks[0], drums_lib.DrumTrack))

    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)

  def testExtractMultipleDrumTracks(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)],
        is_drum=True)
    testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 37, 38)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(11, 14)],
                [NO_DRUMS, DRUMS(50), NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(52)]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        quantized_sequence, min_bars=1, gap_bars=2)
    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooShort(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 3, 4), (14, 50, 6, 7)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        quantized_sequence, min_bars=2, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual([], drum_tracks)

    del self.note_sequence.notes[:]
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 3, 4), (14, 50, 7, 8)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        quantized_sequence, min_bars=2, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual(
        [[NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
          DRUMS(14)]],
        drum_tracks)

  def testExtractDrumTracksPadEnd(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 7)],
        is_drum=True)
    testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (15, 50, 6, 8)],
        is_drum=True)
    testing_lib.add_track_to_sequence(
        self.note_sequence, 2,
        [(12, 127, 2, 4), (16, 50, 8, 9)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(14, 15), NO_DRUMS, DRUMS(16), NO_DRUMS, NO_DRUMS,
                 NO_DRUMS]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        quantized_sequence, min_bars=1, gap_bars=1, pad_end=True)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooLongTruncate(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 15), (14, 50, 10, 15), (16, 100, 14, 19)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(14), NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(14), NO_DRUMS,
                 NO_DRUMS, NO_DRUMS]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        quantized_sequence, min_bars=1, max_steps_truncate=14, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooLongDiscard(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 15), (14, 50, 10, 15), (16, 100, 14, 19),
         (14, 100, 18, 19)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        quantized_sequence, min_bars=1, max_steps_discard=18, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual([], drum_tracks)

  def testExtractDrumTracksLateStart(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 102, 103), (13, 100, 104, 106)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, DRUMS(13)]]
    drum_tracks, _ = drums_lib.extract_drum_tracks(
        quantized_sequence, min_bars=1, gap_bars=1)
    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)


if __name__ == '__main__':
  tf.test.main()
