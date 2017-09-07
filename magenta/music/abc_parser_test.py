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
"""Tests for abc_parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os.path

# internal imports

import six
import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import abc_parser
from magenta.music import midi_io
from magenta.protobuf import music_pb2

class AbcParserTest(tf.test.TestCase):

  def setUp(self):
    self.maxDiff = None

  def compareAccidentals(self, expected, accidentals):
    values = [v[1] for v in sorted(six.iteritems(accidentals))]
    self.assertEqual(expected, values)

  def compareAbc2midiNotes(self, expected, test):
    self.assertEqual(len(expected.notes), len(test.notes))

    # abc2midi adds a 1-tick delay to the start of every note, but we don't.
    tick_length = ((1 / (expected.tempos[0].qpm / 60)) /
                   expected.ticks_per_quarter)

    expected_adj = copy.deepcopy(expected)
    for note in expected_adj.notes:
      note.start_time -= tick_length

    for exp_note, test_note in zip(expected_adj.notes, test.notes):
      # For now, don't compare velocities.
      exp_note.velocity = test_note.velocity
      self.assertProtoEquals(exp_note, test_note)

  def testParseKeyBasic(self):
    # Most examples taken from
    # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key('C major')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.C, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key('A minor')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.A, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MINOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'C ionian')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.C, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'A aeolian')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.A, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MINOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'G Mixolydian')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.G, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MIXOLYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D dorian')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.DORIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'E phrygian')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.E, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.PHRYGIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'F Lydian')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.LYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'B Locrian')
    self.compareAccidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.B, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.LOCRIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'F# mixolydian')
    self.compareAccidentals([1, 0, 1, 1, 0, 1, 1], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F_SHARP, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MIXOLYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'F#Mix')
    self.compareAccidentals([1, 0, 1, 1, 0, 1, 1], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F_SHARP, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MIXOLYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'F#MIX')
    self.compareAccidentals([1, 0, 1, 1, 0, 1, 1], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F_SHARP, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MIXOLYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'Fm')
    self.compareAccidentals([-1, -1, 0, -1, -1, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MINOR, proto_mode)

  def testParseKeyExplicit(self):
    # Most examples taken from
    # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D exp _b _e ^f')
    self.compareAccidentals([0, -1, 0, 0, -1, 1, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

  def testParseKeyAccidentals(self):
    # Most examples taken from
    # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D Phr ^f')
    self.compareAccidentals([0, -1, 0, 0, -1, 1, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.PHRYGIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D maj =c')
    self.compareAccidentals([0, 0, 0, 0, 0, 1, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D =c')
    self.compareAccidentals([0, 0, 0, 0, 0, 1, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

  def testParseEnglishAbc(self):
    tunes = abc_parser.parse_tunebook_file(
        os.path.join(tf.resource_loader.get_data_files_path(),
                     'testdata/english.abc'))
    self.assertEqual(3, len(tunes))

    abc2midi_1 = midi_io.midi_file_to_sequence_proto(
        os.path.join(tf.resource_loader.get_data_files_path(),
                     'testdata/english1.mid'))

    del abc2midi_1.notes[10:]
    del tunes[0].notes[10:]
    self.compareAbc2midiNotes(abc2midi_1, tunes[0])

    expected_ns1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: ABC
          parser: MAGENTA_ABC
        }
        reference_number: 1
        sequence_metadata {
          title: "Dusty Miller, The; Binny's Jig"
          artist: "Trad."
          composers: "Trad."
        }
        time_signatures {
          numerator: 3
          denominator: 4
        }
        key_signatures {
          key: G
        }
        """)
    # We've already compared notes, now check the other attributes.
    del tunes[0].notes[:]
    self.assertProtoEquals(expected_ns1, tunes[0])

    expected_ns2 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: ABC
          parser: MAGENTA_ABC
        }
        reference_number: 2
        sequence_metadata {
          title: "Old Sir Simon the King"
          artist: "Trad."
          composers: "Trad."
        }
        time_signatures {
          numerator: 9
          denominator: 8
        }
        time_signatures {
          numerator: 12
          denominator: 8
        }
        time_signatures {
          numerator: 9
          denominator: 8
        }
        key_signatures {
          key: G
        }
        """)
    # TODO(fjord): add notes and times.
    del tunes[1].notes[:]
    self.assertProtoEquals(expected_ns2, tunes[1])

    expected_ns3 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: ABC
          parser: MAGENTA_ABC
        }
        reference_number: 3
        sequence_metadata {
          title: "William and Nancy; New Mown Hay; Legacy, The"
          artist: "Trad."
          composers: "Trad."
        }
        time_signatures {
          numerator: 6
          denominator: 8
        }
        key_signatures {
          key: G
        }
        """)
    # TODO(fjord): add notes and times.
    del tunes[2].notes[:]
    del tunes[2].text_annotations[:]
    self.assertProtoEquals(expected_ns3, tunes[2])

  def testParseOctaves(self):
    tunes = abc_parser.parse_tunebook("""X:1
        T:Test
        CC,',C,C'c
        """)
    self.assertEqual(1, len(tunes))

    expected_ns1 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: ABC
          parser: MAGENTA_ABC
        }
        reference_number: 1
        sequence_metadata {
          title: "Test"
        }
        notes {
          pitch: 60
          velocity: 90
          end_time: 0.25
        }
        notes {
          pitch: 48
          velocity: 90
          start_time: 0.25
          end_time: 0.5
        }
        notes {
          pitch: 48
          velocity: 90
          start_time: 0.5
          end_time: 0.75
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 0.75
          end_time: 1.0
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 1.0
          end_time: 1.25
        }
        """)
    # TODO(fjord): add timing
    self.assertProtoEquals(expected_ns1, tunes[0])

  def testParseTempos(self):
    # Examples from http://abcnotation.com/wiki/abc:standard:v2.1#qtempo
    tunes = abc_parser.parse_tunebook("""
        X:1
        L:1/4
        Q:60

        X:2
        L:1/4
        Q:C=100

        X:3
        Q:1/2=120

        X:4
        Q:1/4 3/8 1/4 3/8=40

        X:5
        Q:5/4=40

        X:6
        Q: "Allegro" 1/4=120

        X:7
        Q: 1/4=120 "Allegro"

        X:8
        Q: 3/8=50 "Slowly"

        X:9
        Q:"Andante"

        X:10
        Q:100  % define tempo using deprecated syntax
        % deprecated tempo syntax depends on unit note length. if it is
        % not defined, it is derived from the current meter.
        M:2/4  % define meter after tempo to verify that is supported.

        X:11
        Q:100  % define tempo using deprecated syntax
        % deprecated tempo syntax depends on unit note length.
        L:1/4  % define note length after tempo to verify that is supported.
        """)
    self.assertEqual(11, len(tunes))

    self.assertEqual(60, tunes[0].tempos[0].qpm)
    self.assertEqual(100, tunes[1].tempos[0].qpm)
    self.assertEqual(240, tunes[2].tempos[0].qpm)
    self.assertEqual(200, tunes[3].tempos[0].qpm)
    self.assertEqual(200, tunes[4].tempos[0].qpm)
    self.assertEqual(120, tunes[5].tempos[0].qpm)
    self.assertEqual(120, tunes[6].tempos[0].qpm)
    self.assertEqual(75, tunes[7].tempos[0].qpm)
    self.assertEqual(0, len(tunes[8].tempos))
    self.assertEqual(25, tunes[9].tempos[0].qpm)
    self.assertEqual(100, tunes[10].tempos[0].qpm)

if __name__ == '__main__':
  tf.test.main()
