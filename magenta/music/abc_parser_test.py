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

# internal imports

import six
import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import abc_parser
from magenta.protobuf import music_pb2

# Sample tunes taken from
# http://abcnotation.com/wiki/abc:standard:v2.1#sample_abc_tunes

ENGLISH_ABC = r"""%abc-2.1
H:This file contains some example English tunes
% note that the comments (like this one) are to highlight usages
%  and would not normally be included in such detail
O:England             % the origin of all tunes is England

X:1                   % tune no 1
T:Dusty Miller, The   % title
T:Binny's Jig         % an alternative title
C:Trad.               % traditional
R:DH                  % double hornpipe
M:3/4                 % meter
K:G                   % key
B>cd BAG|FA Ac BA|B>cd BAG|DG GB AG:|
Bdd gfg|aA Ac BA|Bdd gfa|gG GB AG:|
BG G/2G/2G BG|FA Ac BA|BG G/2G/2G BG|DG GB AG:|
W:Hey, the dusty miller, and his dusty coat;
W:He will win a shilling, or he spend a groat.
W:Dusty was the coat, dusty was the colour;
W:Dusty was the kiss, that I got frae the miller.

X:2
T:Old Sir Simon the King
C:Trad.
S:Offord MSS          % from Offord manuscript
N:see also Playford   % reference note
M:9/8
R:SJ                  % slip jig
N:originally in C     % transcription note
K:G
D|GFG GAG G2D|GFG GAG F2D|EFE EFE EFG|A2G F2E D2:|
D|GAG GAB d2D|GAG GAB c2D|[1 EFE EFE EFG|A2G F2E D2:|\ % no line-break in score
M:12/8                % change of meter
[2 E2E EFE E2E EFG|\  % no line-break in score
M:9/8                 % change of meter
A2G F2E D2|]

X:3
T:William and Nancy
T:New Mown Hay
T:Legacy, The
C:Trad.
O:England; Gloucs; Bledington % place of origin
B:Sussex Tune Book            % can be found in these books
B:Mally's Cotswold Morris vol.1 2
D:Morris On                   % can be heard on this record
P:(AB)2(AC)2A                 % play the parts in this order
M:6/8
K:G
[P:A] D|"G"G2G GBd|"C"e2e "G"dBG|"D7"A2d "G"BAG|"C"E2"D7"F "G"G2:|
[P:B] d|"G"e2d B2d|"C"gfe "G"d2d| "G"e2d    B2d|"C"gfe    "D7"d2c|
        "G"B2B Bcd|"C"e2e "G"dBG|"D7"A2d "G"BAG|"C"E2"D7"F "G"G2:|
% changes of meter, using inline fields
[T:Slows][M:4/4][L:1/4][P:C]"G"d2|"C"e2 "G"d2|B2 d2|"Em"gf "A7"e2|"D7"d2 "G"d2|\
       "C"e2 "G"d2|[M:3/8][L:1/8] "G"B2 d |[M:6/8] "C"gfe "D7"d2c|
        "G"B2B Bcd|"C"e2e "G"dBG|"D7"A2d "G"BAG|"C"E2"D7"F "G"G2:|
"""


class AbcParserTest(tf.test.TestCase):

  def setUp(self):
    self.maxDiff = None

  def compareAccidentals(self, expected, accidentals):
    values = [v[1] for v in sorted(six.iteritems(accidentals))]
    self.assertEqual(expected, values)

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
    tunes = abc_parser.parse_tunebook(ENGLISH_ABC)
    self.assertEqual(3, len(tunes))

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
    # TODO(fjord): add notes
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
        }
        notes {
          pitch: 48
          velocity: 90
        }
        notes {
          pitch: 48
          velocity: 90
        }
        notes {
          pitch: 72
          velocity: 90
        }
        notes {
          pitch: 72
          velocity: 90
        }
        """)
    # TODO(fjord): add timing
    self.assertProtoEquals(expected_ns1, tunes[0])

if __name__ == '__main__':
  tf.test.main()
