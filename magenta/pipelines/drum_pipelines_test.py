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

"""Tests for drum_pipelines."""

from absl.testing import absltest
from magenta.common import testing_lib as common_testing_lib
from magenta.pipelines import drum_pipelines
from note_seq import drums_lib
from note_seq import sequences_lib
from note_seq import testing_lib as music_testing_lib
from note_seq.protobuf import music_pb2

DRUMS = lambda *args: frozenset(args)
NO_DRUMS = frozenset()


class DrumPipelinesTest(absltest.TestCase):

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

  def _unit_transform_test(self, unit, input_instance,
                           expected_outputs):
    outputs = unit.transform(input_instance)
    self.assertIsInstance(outputs, list)
    common_testing_lib.assert_set_equality(self, expected_outputs, outputs)
    self.assertEqual(unit.input_type, type(input_instance))
    if outputs:
      self.assertEqual(unit.output_type, type(outputs[0]))

  def testDrumsExtractor(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    music_testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7), (12, 1, 6, 8)],
        is_drum=True)
    music_testing_lib.add_track_to_sequence(
        note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        note_sequence, steps_per_quarter=1)
    expected_events = [
        [NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
         DRUMS(11, 12)]]
    expected_drum_tracks = []
    for events_list in expected_events:
      drums = drums_lib.DrumTrack(
          events_list, steps_per_quarter=1, steps_per_bar=4)
      expected_drum_tracks.append(drums)
    unit = drum_pipelines.DrumsExtractor(min_bars=1, gap_bars=1)
    self._unit_transform_test(unit, quantized_sequence, expected_drum_tracks)

  def testExtractDrumTracksSimple(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7)],
        is_drum=True)
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 9)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(11, 14)]]
    drum_tracks, _ = drum_pipelines.extract_drum_tracks(
        quantized_sequence, min_bars=1, gap_bars=1)

    self.assertLen(drum_tracks, 1)
    self.assertIsInstance(drum_tracks[0], drums_lib.DrumTrack)

    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)

  def testExtractMultipleDrumTracks(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)],
        is_drum=True)
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 37, 38)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(11, 14)],
                [NO_DRUMS, DRUMS(50), NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(52)]]
    drum_tracks, _ = drum_pipelines.extract_drum_tracks(
        quantized_sequence, min_bars=1, gap_bars=2)
    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooShort(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 3, 4), (14, 50, 6, 7)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    drum_tracks, _ = drum_pipelines.extract_drum_tracks(
        quantized_sequence, min_bars=2, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual([], drum_tracks)

    del self.note_sequence.notes[:]
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 3, 4), (14, 50, 7, 8)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    drum_tracks, _ = drum_pipelines.extract_drum_tracks(
        quantized_sequence, min_bars=2, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual(
        [[NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
          DRUMS(14)]],
        drum_tracks)

  def testExtractDrumTracksPadEnd(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 7)],
        is_drum=True)
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(12, 127, 2, 4), (15, 50, 6, 8)],
        is_drum=True)
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 2,
        [(12, 127, 2, 4), (16, 50, 8, 9)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(14, 15), NO_DRUMS, DRUMS(16), NO_DRUMS, NO_DRUMS,
                 NO_DRUMS]]
    drum_tracks, _ = drum_pipelines.extract_drum_tracks(
        quantized_sequence, min_bars=1, gap_bars=1, pad_end=True)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooLongTruncate(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 15), (14, 50, 10, 15), (16, 100, 14, 19)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, NO_DRUMS, NO_DRUMS,
                 DRUMS(14), NO_DRUMS, NO_DRUMS, NO_DRUMS, DRUMS(14), NO_DRUMS,
                 NO_DRUMS, NO_DRUMS]]
    drum_tracks, _ = drum_pipelines.extract_drum_tracks(
        quantized_sequence, min_bars=1, max_steps_truncate=14, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual(expected, drum_tracks)

  def testExtractDrumTracksTooLongDiscard(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 127, 2, 4), (14, 50, 6, 15), (14, 50, 10, 15), (16, 100, 14, 19),
         (14, 100, 18, 19)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    drum_tracks, _ = drum_pipelines.extract_drum_tracks(
        quantized_sequence, min_bars=1, max_steps_discard=18, gap_bars=1)
    drum_tracks = [list(drums) for drums in drum_tracks]
    self.assertEqual([], drum_tracks)

  def testExtractDrumTracksLateStart(self):
    music_testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 102, 103), (13, 100, 104, 106)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    expected = [[NO_DRUMS, NO_DRUMS, DRUMS(12), NO_DRUMS, DRUMS(13)]]
    drum_tracks, _ = drum_pipelines.extract_drum_tracks(
        quantized_sequence, min_bars=1, gap_bars=1)
    drum_tracks = sorted([list(drums) for drums in drum_tracks])
    self.assertEqual(expected, drum_tracks)


if __name__ == '__main__':
  absltest.main()
