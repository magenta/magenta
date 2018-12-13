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

import six
import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import abc_parser
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2


class AbcParserTest(tf.test.TestCase):

  def setUp(self):
    self.maxDiff = None  # pylint:disable=invalid-name

  def compare_accidentals(self, expected, accidentals):
    values = [v[1] for v in sorted(six.iteritems(accidentals))]
    self.assertEqual(expected, values)

  def compare_proto_list(self, expected, test):
    self.assertEqual(len(expected), len(test))
    for e, t in zip(expected, test):
      self.assertProtoEquals(e, t)

  def compare_to_abc2midi_and_metadata(
      self, midi_path, expected_metadata, expected_expanded_metadata, test):
    """Compare parsing results to the abc2midi "reference" implementation."""
    # Compare section annotations and groups before expanding.
    self.compare_proto_list(expected_metadata.section_annotations,
                            test.section_annotations)
    self.compare_proto_list(expected_metadata.section_groups,
                            test.section_groups)

    expanded_test = sequences_lib.expand_section_groups(test)

    abc2midi = midi_io.midi_file_to_sequence_proto(
        os.path.join(tf.resource_loader.get_data_files_path(), midi_path))

    # abc2midi adds a 1-tick delay to the start of every note, but we don't.
    tick_length = ((1 / (abc2midi.tempos[0].qpm / 60)) /
                   abc2midi.ticks_per_quarter)

    for note in abc2midi.notes:
      # For now, don't compare velocities.
      note.velocity = 90
      note.start_time -= tick_length

    self.compare_proto_list(abc2midi.notes, expanded_test.notes)

    self.assertEqual(abc2midi.total_time, expanded_test.total_time)

    self.compare_proto_list(abc2midi.time_signatures,
                            expanded_test.time_signatures)

    # We've checked the notes and time signatures, now compare the rest of the
    # proto to the expected proto.
    expanded_test_copy = copy.deepcopy(expanded_test)
    del expanded_test_copy.notes[:]
    expanded_test_copy.ClearField('total_time')
    del expanded_test_copy.time_signatures[:]

    self.assertProtoEquals(expected_expanded_metadata, expanded_test_copy)

  def testParseKeyBasic(self):
    # Most examples taken from
    # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key('C major')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.C, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key('A minor')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.A, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MINOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'C ionian')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.C, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'A aeolian')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.A, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MINOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'G Mixolydian')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.G, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MIXOLYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D dorian')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.DORIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'E phrygian')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.E, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.PHRYGIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'F Lydian')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.LYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'B Locrian')
    self.compare_accidentals([0, 0, 0, 0, 0, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.B, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.LOCRIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'F# mixolydian')
    self.compare_accidentals([1, 0, 1, 1, 0, 1, 1], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F_SHARP, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MIXOLYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'F#Mix')
    self.compare_accidentals([1, 0, 1, 1, 0, 1, 1], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F_SHARP, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MIXOLYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'F#MIX')
    self.compare_accidentals([1, 0, 1, 1, 0, 1, 1], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F_SHARP, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MIXOLYDIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'Fm')
    self.compare_accidentals([-1, -1, 0, -1, -1, 0, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.F, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MINOR, proto_mode)

  def testParseKeyExplicit(self):
    # Most examples taken from
    # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D exp _b _e ^f')
    self.compare_accidentals([0, -1, 0, 0, -1, 1, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

  def testParseKeyAccidentals(self):
    # Most examples taken from
    # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D Phr ^f')
    self.compare_accidentals([0, -1, 0, 0, -1, 1, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.PHRYGIAN,
                     proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D maj =c')
    self.compare_accidentals([0, 0, 0, 0, 0, 1, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

    accidentals, proto_key, proto_mode = abc_parser.ABCTune.parse_key(
        'D =c')
    self.compare_accidentals([0, 0, 0, 0, 0, 1, 0], accidentals)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.D, proto_key)
    self.assertEqual(music_pb2.NoteSequence.KeySignature.MAJOR, proto_mode)

  def testParseEnglishAbc(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook_file(
        os.path.join(tf.resource_loader.get_data_files_path(),
                     'testdata/english.abc'))
    self.assertEqual(1, len(tunes))
    self.assertEqual(2, len(exceptions))
    self.assertTrue(isinstance(exceptions[0],
                               abc_parser.VariantEndingException))
    self.assertTrue(isinstance(exceptions[1], abc_parser.PartException))

    expected_metadata1 = common_testing_lib.parse_test_proto(
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
        key_signatures {
          key: G
        }
        section_annotations {
          time: 0.0
          section_id: 0
        }
        section_annotations {
          time: 6.0
          section_id: 1
        }
        section_annotations {
          time: 12.0
          section_id: 2
        }
        section_groups {
          sections {
            section_id: 0
          }
          num_times: 2
        }
        section_groups {
          sections {
            section_id: 1
          }
          num_times: 2
        }
        section_groups {
          sections {
            section_id: 2
          }
          num_times: 2
        }
        """)
    expected_expanded_metadata1 = common_testing_lib.parse_test_proto(
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
        key_signatures {
          key: G
        }
        section_annotations {
          time: 0.0
          section_id: 0
        }
        section_annotations {
          time: 6.0
          section_id: 0
        }
        section_annotations {
          time: 12.0
          section_id: 1
        }
        section_annotations {
          time: 18.0
          section_id: 1
        }
        section_annotations {
          time: 24.0
          section_id: 2
        }
        section_annotations {
          time: 30.0
          section_id: 2
        }
        """)
    self.compare_to_abc2midi_and_metadata(
        'testdata/english1.mid', expected_metadata1,
        expected_expanded_metadata1, tunes[1])

    # TODO(fjord): re-enable once we support variant endings.
    # expected_ns2_metadata = common_testing_lib.parse_test_proto(
    #     music_pb2.NoteSequence,
    #     """
    #     ticks_per_quarter: 220
    #     source_info: {
    #       source_type: SCORE_BASED
    #       encoding_type: ABC
    #       parser: MAGENTA_ABC
    #     }
    #     reference_number: 2
    #     sequence_metadata {
    #       title: "Old Sir Simon the King"
    #       artist: "Trad."
    #       composers: "Trad."
    #     }
    #     key_signatures {
    #       key: G
    #     }
    #     """)
    # self.compare_to_abc2midi_and_metadata(
    #     'testdata/english2.mid', expected_ns2_metadata, tunes[1])

    # TODO(fjord): re-enable once we support parts.
    # expected_ns3_metadata = common_testing_lib.parse_test_proto(
    #     music_pb2.NoteSequence,
    #     """
    #     ticks_per_quarter: 220
    #     source_info: {
    #       source_type: SCORE_BASED
    #       encoding_type: ABC
    #       parser: MAGENTA_ABC
    #     }
    #     reference_number: 3
    #     sequence_metadata {
    #       title: "William and Nancy; New Mown Hay; Legacy, The"
    #       artist: "Trad."
    #       composers: "Trad."
    #     }
    #     key_signatures {
    #       key: G
    #     }
    #     """)
    # # TODO(fjord): verify chord annotations
    # del tunes[3].text_annotations[:]
    # self.compare_to_abc2midi_and_metadata(
    #     'testdata/english3.mid', expected_ns3_metadata, tunes[3])

  def testParseOctaves(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""X:1
        T:Test
        CC,',C,C'c
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))

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
        total_time: 1.25
        """)
    self.assertProtoEquals(expected_ns1, tunes[1])

  def testParseTempos(self):
    # Examples from http://abcnotation.com/wiki/abc:standard:v2.1#qtempo
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
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
    self.assertEqual(0, len(exceptions))

    self.assertEqual(60, tunes[1].tempos[0].qpm)
    self.assertEqual(100, tunes[2].tempos[0].qpm)
    self.assertEqual(240, tunes[3].tempos[0].qpm)
    self.assertEqual(200, tunes[4].tempos[0].qpm)
    self.assertEqual(200, tunes[5].tempos[0].qpm)
    self.assertEqual(120, tunes[6].tempos[0].qpm)
    self.assertEqual(120, tunes[7].tempos[0].qpm)
    self.assertEqual(75, tunes[8].tempos[0].qpm)
    self.assertEqual(0, len(tunes[9].tempos))
    self.assertEqual(25, tunes[10].tempos[0].qpm)
    self.assertEqual(100, tunes[11].tempos[0].qpm)

  def testParseBrokenRhythm(self):
    # These tunes should be equivalent.
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        M:3/4
        T:Test
        B>cd B<cd

        X:2
        Q:1/4=120
        L:1/4
        M:3/4
        T:Test
        B3/2c/2d B/2c3/2d

        X:3
        Q:1/4=120
        L:1/4
        M:3/4
        T:Test
        B3/c/d B/c3/d
        """)
    self.assertEqual(3, len(tunes))
    self.assertEqual(0, len(exceptions))

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
        time_signatures {
          numerator: 3
          denominator: 4
        }
        tempos {
          qpm: 120
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 0.0
          end_time: 0.75
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 0.75
          end_time: 1.0
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 1.0
          end_time: 1.5
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 1.5
          end_time: 1.75
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 1.75
          end_time: 2.5
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 2.5
          end_time: 3.0
        }
        total_time: 3.0
        """)
    self.assertProtoEquals(expected_ns1, tunes[1])
    expected_ns2 = copy.deepcopy(expected_ns1)
    expected_ns2.reference_number = 2
    self.assertProtoEquals(expected_ns2, tunes[2])
    expected_ns2.reference_number = 3
    self.assertProtoEquals(expected_ns2, tunes[3])

  def testSlashDuration(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""X:1
        Q:1/4=120
        L:1/4
        T:Test
        CC/C//C///C////
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))

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
        tempos {
          qpm: 120
        }
        notes {
          pitch: 60
          velocity: 90
          start_time: 0.0
          end_time: 0.5
        }
        notes {
          pitch: 60
          velocity: 90
          start_time: 0.5
          end_time: 0.75
        }
        notes {
          pitch: 60
          velocity: 90
          start_time: 0.75
          end_time: 0.875
        }
        notes {
          pitch: 60
          velocity: 90
          start_time: 0.875
          end_time: 0.9375
        }
        notes {
          pitch: 60
          velocity: 90
          start_time: 0.9375
          end_time: 0.96875
        }
        total_time: 0.96875
        """)
    self.assertProtoEquals(expected_ns1, tunes[1])

  def testMultiVoice(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook_file(
        os.path.join(tf.resource_loader.get_data_files_path(),
                     'testdata/zocharti_loch.abc'))
    self.assertEqual(0, len(tunes))
    self.assertEqual(1, len(exceptions))
    self.assertTrue(isinstance(exceptions[0], abc_parser.MultiVoiceException))

  def testRepeats(self):
    # Several equivalent versions of the same tune.
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        Bcd ::[]|[]:: Bcd ::|

        X:2
        Q:1/4=120
        L:1/4
        T:Test
        Bcd :::: Bcd ::|

        X:3
        Q:1/4=120
        L:1/4
        T:Test
        |::Bcd ::|:: Bcd ::|

        % This version contains mismatched repeat symbols.
        X:4
        Q:1/4=120
        L:1/4
        T:Test
        |::Bcd ::|: Bcd ::|

        % This version is missing a repeat symbol at the end.
        X:5
        Q:1/4=120
        L:1/4
        T:Test
        |:: Bcd ::|: Bcd |

        % Ambiguous repeat that should go to the last repeat symbol.
        X:6
        Q:1/4=120
        L:1/4
        T:Test
        |:: Bcd ::| Bcd :|

        % Ambiguous repeat that should go to the last double bar.
        X:7
        Q:1/4=120
        L:1/4
        T:Test
        |:: Bcd ::| Bcd || Bcd :|

        % Ambiguous repeat that should go to the last double bar.
        X:8
        Q:1/4=120
        L:1/4
        T:Test
        || Bcd ::| Bcd || Bcd :|

        % Ensure double bar doesn't confuse declared repeat.
        X:9
        Q:1/4=120
        L:1/4
        T:Test
        |:: B || cd ::| Bcd || |: Bcd :|

        % Mismatched repeat at the very beginning.
        X:10
        Q:1/4=120
        L:1/4
        T:Test
        :| Bcd |:: Bcd ::|
        """)
    self.assertEqual(7, len(tunes))
    self.assertEqual(3, len(exceptions))
    self.assertTrue(isinstance(exceptions[0], abc_parser.RepeatParseException))
    self.assertTrue(isinstance(exceptions[1], abc_parser.RepeatParseException))
    self.assertTrue(isinstance(exceptions[2], abc_parser.RepeatParseException))
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
        tempos {
          qpm: 120
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 0.0
          end_time: 0.5
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 0.5
          end_time: 1.0
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 1.0
          end_time: 1.5
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 1.5
          end_time: 2.0
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 2.0
          end_time: 2.5
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 2.5
          end_time: 3.0
        }
        section_annotations {
          time: 0
          section_id: 0
        }
        section_annotations {
          time: 1.5
          section_id: 1
        }
        section_groups {
          sections {
            section_id: 0
          }
          num_times: 3
        }
        section_groups {
          sections {
            section_id: 1
          }
          num_times: 3
        }
        total_time: 3.0
        """)
    self.assertProtoEquals(expected_ns1, tunes[1])

    # Other versions are identical except for the reference number.
    expected_ns2 = copy.deepcopy(expected_ns1)
    expected_ns2.reference_number = 2
    self.assertProtoEquals(expected_ns2, tunes[2])

    expected_ns3 = copy.deepcopy(expected_ns1)
    expected_ns3.reference_number = 3
    self.assertProtoEquals(expected_ns3, tunes[3])

    # Also identical, except the last section is played only twice.
    expected_ns6 = copy.deepcopy(expected_ns1)
    expected_ns6.reference_number = 6
    expected_ns6.section_groups[-1].num_times = 2
    self.assertProtoEquals(expected_ns6, tunes[6])

    expected_ns7 = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: ABC
          parser: MAGENTA_ABC
        }
        reference_number: 7
        sequence_metadata {
          title: "Test"
        }
        tempos {
          qpm: 120
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 0.0
          end_time: 0.5
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 0.5
          end_time: 1.0
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 1.0
          end_time: 1.5
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 1.5
          end_time: 2.0
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 2.0
          end_time: 2.5
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 2.5
          end_time: 3.0
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 3.0
          end_time: 3.5
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 3.5
          end_time: 4.0
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 4.0
          end_time: 4.5
        }
        section_annotations {
          time: 0
          section_id: 0
        }
        section_annotations {
          time: 1.5
          section_id: 1
        }
        section_annotations {
          time: 3.0
          section_id: 2
        }
        section_groups {
          sections {
            section_id: 0
          }
          num_times: 3
        }
        section_groups {
          sections {
            section_id: 1
          }
          num_times: 1
        }
        section_groups {
          sections {
            section_id: 2
          }
          num_times: 2
        }
        total_time: 4.5
        """)
    self.assertProtoEquals(expected_ns7, tunes[7])

    expected_ns8 = copy.deepcopy(expected_ns7)
    expected_ns8.reference_number = 8
    self.assertProtoEquals(expected_ns8, tunes[8])

    expected_ns9 = copy.deepcopy(expected_ns7)
    expected_ns9.reference_number = 9
    self.assertProtoEquals(expected_ns9, tunes[9])

  def testInvalidCharacter(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        invalid notes!""")
    self.assertEqual(0, len(tunes))
    self.assertEqual(1, len(exceptions))
    self.assertTrue(isinstance(exceptions[0],
                               abc_parser.InvalidCharacterException))

  def testOneSidedRepeat(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        Bcd :| Bcd
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))
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
        tempos {
          qpm: 120
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 0.0
          end_time: 0.5
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 0.5
          end_time: 1.0
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 1.0
          end_time: 1.5
        }
        notes {
          pitch: 71
          velocity: 90
          start_time: 1.5
          end_time: 2.0
        }
        notes {
          pitch: 72
          velocity: 90
          start_time: 2.0
          end_time: 2.5
        }
        notes {
          pitch: 74
          velocity: 90
          start_time: 2.5
          end_time: 3.0
        }
        section_annotations {
          time: 0
          section_id: 0
        }
        section_annotations {
          time: 1.5
          section_id: 1
        }
        section_groups {
          sections {
            section_id: 0
          }
          num_times: 2
        }
        section_groups {
          sections {
            section_id: 1
          }
          num_times: 1
        }
        total_time: 3.0
        """)
    self.assertProtoEquals(expected_ns1, tunes[1])

  def testChords(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        [CEG]""")
    self.assertEqual(0, len(tunes))
    self.assertEqual(1, len(exceptions))
    self.assertTrue(isinstance(exceptions[0],
                               abc_parser.ChordException))

  def testChordAnnotations(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        "G"G
        % verify that an empty annotation doesn't cause problems.
        ""D
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))
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
        tempos {
          qpm: 120
        }
        notes {
          pitch: 67
          velocity: 90
          end_time: 0.5
        }
        notes {
          pitch: 62
          velocity: 90
          start_time: 0.5
          end_time: 1.0
        }
        text_annotations {
          text: "G"
          annotation_type: CHORD_SYMBOL
        }
        text_annotations {
          time: 0.5
        }
        total_time: 1.0
        """)
    self.assertProtoEquals(expected_ns1, tunes[1])

  def testNoteAccidentalsPerBar(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        GF^GGg|Gg
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))
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
        tempos {
          qpm: 120
        }
        notes {
          pitch: 67
          velocity: 90
          start_time: 0.0
          end_time: 0.5
        }
        notes {
          pitch: 65
          velocity: 90
          start_time: 0.5
          end_time: 1.0
        }
        notes {
          pitch: 68
          velocity: 90
          start_time: 1.0
          end_time: 1.5
        }
        notes {
          pitch: 68
          velocity: 90
          start_time: 1.5
          end_time: 2.0
        }
        notes {
          pitch: 80
          velocity: 90
          start_time: 2.0
          end_time: 2.5
        }
        notes {
          pitch: 67
          velocity: 90
          start_time: 2.5
          end_time: 3.0
        }
        notes {
          pitch: 79
          velocity: 90
          start_time: 3.0
          end_time: 3.5
        }
        total_time: 3.5
        """)
    self.assertProtoEquals(expected_ns1, tunes[1])

  def testDecorations(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        .a~bHcLdMeOfPgSATbucvd
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))
    self.assertEqual(11, len(tunes[1].notes))

  def testSlur(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        (ABC) ( a b c ) (c (d e f) g a)
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))
    self.assertEqual(12, len(tunes[1].notes))

  def testTie(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        abc-|cba c4-c4 C.-C
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))
    self.assertEqual(10, len(tunes[1].notes))

  def testTuplet(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook("""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        (3abc
        """)
    self.assertEqual(0, len(tunes))
    self.assertEqual(1, len(exceptions))
    self.assertTrue(isinstance(exceptions[0], abc_parser.TupletException))

  def testLineContinuation(self):
    tunes, exceptions = abc_parser.parse_abc_tunebook(r"""
        X:1
        Q:1/4=120
        L:1/4
        T:Test
        abc \
        cba|
        abc\
         cba|
        abc cba|
        cdef|\
        \
        cedf:|
        """)
    self.assertEqual(1, len(tunes))
    self.assertEqual(0, len(exceptions))
    self.assertEqual(26, len(tunes[1].notes))

if __name__ == '__main__':
  tf.test.main()
