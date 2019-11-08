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

"""Tests for drums_lib."""
from magenta.music import drums_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib as music_testing_lib
from magenta.music.protobuf import music_pb2
import tensorflow as tf

DRUMS = lambda *args: frozenset(args)
NO_DRUMS = frozenset()


class DrumsLibTest(tf.test.TestCase):

  def setUp(self):
    self.steps_per_quarter = 4
    self.note_sequence = music_testing_lib.parse_test_proto(
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
    music_testing_lib.add_track_to_sequence(
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
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0, 10), (40, 45, 2.5, 3.5), (60, 100, 4, 5.5)],
        is_drum=True)
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(11, 55, .25, .5), (55, 120, 4, 4.25), (52, 99, 4.75, 5)],
        is_drum=True)
    music_testing_lib.add_track_to_sequence(
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

    music_testing_lib.add_track_to_sequence(
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
    music_testing_lib.add_track_to_sequence(
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
    music_testing_lib.add_track_to_sequence(
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


if __name__ == '__main__':
  tf.test.main()
