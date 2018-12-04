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
"""Test to ensure correct import of MusicXML."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os.path
import tempfile
import zipfile

import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import musicxml_parser
from magenta.music import musicxml_reader
from magenta.protobuf import music_pb2

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class MusicXMLParserTest(tf.test.TestCase):
  """Class to test the MusicXML parser use cases.

  self.flute_scale_filename contains an F-major scale of 8 quarter notes each

  self.clarinet_scale_filename contains a F-major scale of 8 quarter notes
  each appearing as written pitch. This means the key is written as
  G-major but sounds as F-major. The MIDI pitch numbers must be transposed
  to be input into Magenta

  self.band_score_filename contains a number of instruments in written
  pitch. The score has two time signatures (6/8 and 2/4) and two sounding
  keys (Bb-major and Eb major). The file also contains chords and
  multiple voices (see Oboe part in measure 57), as well as dynamics,
  articulations, slurs, ties, hairpins, grace notes, tempo changes,
  and multiple barline types (double, repeat)

  self.compressed_filename contains the same content as
  self.flute_scale_filename, but compressed in MXL format

  self.rhythm_durations_filename contains a variety of rhythms (long, short,
  dotted, tuplet, and dotted tuplet) to test the computation of rhythmic
  ratios.

  self.atonal_transposition_filename contains a change of instrument
  from a non-transposing (Flute) to transposing (Bb Clarinet) in a score
  with no key / atonal. This ensures that transposition works properly when
  no key signature is found (Issue #355)

  self.st_anne_filename contains a 4-voice piece written in two parts.

  self.whole_measure_rest_forward_filename contains 4 measures:
  Measures 1 and 2 contain whole note rests in 4/4. The first is a <note>,
  the second uses a <forward>. The durations must match.
  Measures 3 and 4 contain whole note rests in 2/4. The first is a <note>,
  the second uses a <forward>. The durations must match.
  (Issue #674).

  self.meter_test_filename contains a different meter in each measure:
  - 1/4 through 7/4 inclusive
  - 1/8 through 12/8 inclusive
  - 2/2 through 4/2 inclusive
  - Common time and Cut time meters
  """

  def setUp(self):
    self.maxDiff = None   # pylint:disable=invalid-name

    self.steps_per_quarter = 4

    self.flute_scale_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/flute_scale.xml')

    self.clarinet_scale_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/clarinet_scale.xml')

    self.band_score_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/el_capitan.xml')

    self.compressed_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/flute_scale.mxl')

    self.multiple_rootfile_compressed_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/flute_scale_with_png.mxl')

    self.rhythm_durations_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/rhythm_durations.xml')

    self.st_anne_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/st_anne.xml')

    self.atonal_transposition_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/atonal_transposition_change.xml')

    self.chord_symbols_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/chord_symbols.xml')

    self.time_signature_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/st_anne.xml')

    self.unmetered_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/unmetered_example.xml')

    self.alternating_meter_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/alternating_meter.xml')

    self.mid_measure_meter_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/mid_measure_time_signature.xml')

    self.whole_measure_rest_forward_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/whole_measure_rest_forward.xml')

    self.meter_test_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/meter_test.xml')

  def check_musicxml_and_sequence(self, musicxml, sequence_proto):
    """Compares MusicXMLDocument object against a sequence proto.

    Args:
      musicxml: A MusicXMLDocument object.
      sequence_proto: A tensorflow.magenta.Sequence proto.
    """
    # Test time signature changes.
    self.assertEqual(len(musicxml.get_time_signatures()),
                     len(sequence_proto.time_signatures))
    for musicxml_time, sequence_time in zip(musicxml.get_time_signatures(),
                                            sequence_proto.time_signatures):
      self.assertEqual(musicxml_time.numerator, sequence_time.numerator)
      self.assertEqual(musicxml_time.denominator, sequence_time.denominator)
      self.assertAlmostEqual(musicxml_time.time_position, sequence_time.time)

    # Test key signature changes.
    self.assertEqual(len(musicxml.get_key_signatures()),
                     len(sequence_proto.key_signatures))
    for musicxml_key, sequence_key in zip(musicxml.get_key_signatures(),
                                          sequence_proto.key_signatures):

      if musicxml_key.mode == 'major':
        mode = 0
      elif musicxml_key.mode == 'minor':
        mode = 1

      # The Key enum in music.proto does NOT follow MIDI / MusicXML specs
      # Convert from MIDI / MusicXML key to music.proto key
      music_proto_keys = [11, 6, 1, 8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]
      key = music_proto_keys[musicxml_key.key + 7]
      self.assertEqual(key, sequence_key.key)
      self.assertEqual(mode, sequence_key.mode)
      self.assertAlmostEqual(musicxml_key.time_position, sequence_key.time)

    # Test tempos.
    musicxml_tempos = musicxml.get_tempos()
    self.assertEqual(len(musicxml_tempos),
                     len(sequence_proto.tempos))
    for musicxml_tempo, sequence_tempo in zip(
        musicxml_tempos, sequence_proto.tempos):
      self.assertAlmostEqual(musicxml_tempo.qpm, sequence_tempo.qpm)
      self.assertAlmostEqual(musicxml_tempo.time_position,
                             sequence_tempo.time)

    # Test parts/instruments.
    seq_parts = collections.defaultdict(list)
    for seq_note in sequence_proto.notes:
      seq_parts[seq_note.part].append(seq_note)

    self.assertEqual(len(musicxml.parts), len(seq_parts))
    for musicxml_part, seq_part_id in zip(
        musicxml.parts, sorted(seq_parts.keys())):

      seq_instrument_notes = seq_parts[seq_part_id]
      musicxml_notes = []
      for musicxml_measure in musicxml_part.measures:
        for musicxml_note in musicxml_measure.notes:
          if not musicxml_note.is_rest:
            musicxml_notes.append(musicxml_note)

      self.assertEqual(len(musicxml_notes), len(seq_instrument_notes))
      for musicxml_note, sequence_note in zip(musicxml_notes,
                                              seq_instrument_notes):
        self.assertEqual(musicxml_note.pitch[1], sequence_note.pitch)
        self.assertEqual(musicxml_note.velocity, sequence_note.velocity)
        self.assertAlmostEqual(musicxml_note.note_duration.time_position,
                               sequence_note.start_time)
        self.assertAlmostEqual(musicxml_note.note_duration.time_position
                               + musicxml_note.note_duration.seconds,
                               sequence_note.end_time)
        # Check that the duration specified in the MusicXML and the
        # duration float match to within +/- 1 (delta = 1)
        # Delta is used because duration in MusicXML is always an integer
        # For example, a 3:2 half note might have a durationfloat of 341.333
        # but would have the 1/3 distributed in the MusicXML as
        # 341.0, 341.0, 342.0.
        # Check that (3 * 341.333) = (341 + 341 + 342) is true by checking
        # that 341.0 and 342.0 are +/- 1 of 341.333
        self.assertAlmostEqual(
            musicxml_note.note_duration.duration,
            musicxml_note.state.divisions * 4
            * musicxml_note.note_duration.duration_float(),
            delta=1)

  def check_musicxml_to_sequence(self, filename):
    """Test the translation from MusicXML to Sequence proto."""
    source_musicxml = musicxml_parser.MusicXMLDocument(filename)
    sequence_proto = musicxml_reader.musicxml_to_sequence_proto(source_musicxml)
    self.check_musicxml_and_sequence(source_musicxml, sequence_proto)

  def check_fmajor_scale(self, filename, part_name):
    """Verify MusicXML scale file.

    Verify that it contains the correct pitches (sounding pitch) and durations.

    Args:
      filename: file to test.
      part_name: name of the part the sequence is expected to contain.
    """

    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        key_signatures {
          key: F
          time: 0
        }
        time_signatures {
          numerator: 4
          denominator: 4
        }
        tempos {
          qpm: 120.0
        }
        total_time: 4.0
        """)

    part_info = expected_ns.part_infos.add()
    part_info.name = part_name

    expected_pitches = [65, 67, 69, 70, 72, 74, 76, 77]
    time = 0
    for pitch in expected_pitches:
      note = expected_ns.notes.add()
      note.part = 0
      note.voice = 1
      note.pitch = pitch
      note.start_time = time
      time += .5
      note.end_time = time
      note.velocity = 64
      note.numerator = 1
      note.denominator = 4

    # Convert MusicXML to NoteSequence
    source_musicxml = musicxml_parser.MusicXMLDocument(filename)
    sequence_proto = musicxml_reader.musicxml_to_sequence_proto(source_musicxml)

    # Check equality
    self.assertProtoEquals(expected_ns, sequence_proto)

  def testsimplemusicxmltosequence(self):
    """Test the simple flute scale MusicXML file."""
    self.check_musicxml_to_sequence(self.flute_scale_filename)
    self.check_fmajor_scale(self.flute_scale_filename, 'Flute')

  def testcomplexmusicxmltosequence(self):
    """Test the complex band score MusicXML file."""
    self.check_musicxml_to_sequence(self.band_score_filename)

  def testtransposedxmltosequence(self):
    """Test the translation from transposed MusicXML to Sequence proto.

    Compare a transposed MusicXML file (clarinet) to an identical untransposed
    sequence (flute).
    """
    untransposed_musicxml = musicxml_parser.MusicXMLDocument(
        self.flute_scale_filename)
    transposed_musicxml = musicxml_parser.MusicXMLDocument(
        self.clarinet_scale_filename)
    untransposed_proto = musicxml_reader.musicxml_to_sequence_proto(
        untransposed_musicxml)
    self.check_musicxml_and_sequence(transposed_musicxml, untransposed_proto)
    self.check_fmajor_scale(self.clarinet_scale_filename, 'Clarinet in Bb')

  def testcompressedmxlunicodefilename(self):
    """Test an MXL file containing a unicode filename within its zip archive."""

    unicode_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        'testdata/unicode_filename.mxl')
    sequence = musicxml_reader.musicxml_file_to_sequence_proto(unicode_filename)
    self.assertEqual(len(sequence.notes), 8)

  def testcompressedxmltosequence(self):
    """Test the translation from compressed MusicXML to Sequence proto.

    Compare a compressed MusicXML file to an identical uncompressed sequence.
    """
    uncompressed_musicxml = musicxml_parser.MusicXMLDocument(
        self.flute_scale_filename)
    compressed_musicxml = musicxml_parser.MusicXMLDocument(
        self.compressed_filename)
    uncompressed_proto = musicxml_reader.musicxml_to_sequence_proto(
        uncompressed_musicxml)
    self.check_musicxml_and_sequence(compressed_musicxml, uncompressed_proto)
    self.check_fmajor_scale(self.flute_scale_filename, 'Flute')

  def testmultiplecompressedxmltosequence(self):
    """Test the translation from compressed MusicXML with multiple rootfiles.

    The example MXL file contains a MusicXML file of the Flute F Major scale,
    as well as the PNG rendering of the score contained within the single MXL
    file.
    """
    uncompressed_musicxml = musicxml_parser.MusicXMLDocument(
        self.flute_scale_filename)
    compressed_musicxml = musicxml_parser.MusicXMLDocument(
        self.multiple_rootfile_compressed_filename)
    uncompressed_proto = musicxml_reader.musicxml_to_sequence_proto(
        uncompressed_musicxml)
    self.check_musicxml_and_sequence(compressed_musicxml, uncompressed_proto)
    self.check_fmajor_scale(self.flute_scale_filename, 'Flute')

  def testrhythmdurationsxmltosequence(self):
    """Test the rhythm durations MusicXML file."""
    self.check_musicxml_to_sequence(self.rhythm_durations_filename)

  def testFluteScale(self):
    """Verify properties of the flute scale."""
    ns = musicxml_reader.musicxml_file_to_sequence_proto(
        self.flute_scale_filename)
    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        time_signatures: {
          numerator: 4
          denominator: 4
        }
        tempos: {
          qpm: 120
        }
        key_signatures: {
          key: F
        }
        source_info: {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        part_infos {
          part: 0
          name: "Flute"
        }
        total_time: 4.0
        """)
    expected_pitches = [65, 67, 69, 70, 72, 74, 76, 77]
    time = 0
    for pitch in expected_pitches:
      note = expected_ns.notes.add()
      note.part = 0
      note.voice = 1
      note.pitch = pitch
      note.start_time = time
      time += .5
      note.end_time = time
      note.velocity = 64
      note.numerator = 1
      note.denominator = 4
    self.assertProtoEquals(expected_ns, ns)

  def test_atonal_transposition(self):
    """Test that transposition works when changing instrument transposition.

    This can occur within a single part in a score where the score
    has no key signature / is atonal. Examples include changing from a
    non-transposing instrument to a transposing one (ex. Flute to Bb Clarinet)
    or vice versa, or changing among transposing instruments (ex. Bb Clarinet
    to Eb Alto Saxophone).
    """
    ns = musicxml_reader.musicxml_file_to_sequence_proto(
        self.atonal_transposition_filename)
    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        time_signatures: {
          numerator: 4
          denominator: 4
        }
        tempos: {
          qpm: 120
        }
        key_signatures: {
        }
        part_infos {
          part: 0
          name: "Flute"
        }
        source_info: {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        total_time: 4.0
        """)
    expected_pitches = [72, 74, 76, 77, 79, 77, 76, 74]
    time = 0
    for pitch in expected_pitches:
      note = expected_ns.notes.add()
      note.pitch = pitch
      note.start_time = time
      time += .5
      note.end_time = time
      note.velocity = 64
      note.numerator = 1
      note.denominator = 4
      note.voice = 1
    self.maxDiff = None
    self.assertProtoEquals(expected_ns, ns)

  def test_incomplete_measures(self):
    """Test that incomplete measures have the correct time signature.

    This can occur in pickup bars or incomplete measures. For example,
    if the time signature in the MusicXML is 4/4, but the measure only
    contains one quarter note, Magenta expects this pickup measure to have
    a time signature of 1/4.
    """
    ns = musicxml_reader.musicxml_file_to_sequence_proto(
        self.time_signature_filename)

    # One time signature per measure
    self.assertEqual(len(ns.time_signatures), 6)
    self.assertEqual(len(ns.key_signatures), 1)
    self.assertEqual(len(ns.notes), 112)

  def test_unmetered_music(self):
    """Test that time signatures are inserted for music without time signatures.

    MusicXML does not require the use of time signatures. Music without
    time signatures occur in medieval chant, cadenzas, and contemporary music.
    """
    ns = musicxml_reader.musicxml_file_to_sequence_proto(
        self.unmetered_filename)
    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        time_signatures: {
          numerator: 11
          denominator: 8
        }
        tempos: {
          qpm: 120
        }
        key_signatures: {
        }
        notes {
          pitch: 72
          velocity: 64
          end_time: 0.5
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 74
          velocity: 64
          start_time: 0.5
          end_time: 0.75
          numerator: 1
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 76
          velocity: 64
          start_time: 0.75
          end_time: 1.25
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 77
          velocity: 64
          start_time: 1.25
          end_time: 1.75
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 79
          velocity: 64
          start_time: 1.75
          end_time: 2.75
          numerator: 1
          denominator: 2
          voice: 1
        }
        part_infos {
          name: "Flute"
        }
        source_info: {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        total_time: 2.75
        """)
    self.maxDiff = None
    self.assertProtoEquals(expected_ns, ns)

  def test_st_anne(self):
    """Verify properties of the St. Anne file.

    The file contains 2 parts and 4 voices.
    """
    ns = musicxml_reader.musicxml_file_to_sequence_proto(
        self.st_anne_filename)
    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        time_signatures {
          numerator: 1
          denominator: 4
        }
        time_signatures {
          time: 0.5
          numerator: 4
          denominator: 4
        }
        time_signatures {
          time: 6.5
          numerator: 3
          denominator: 4
        }
        time_signatures {
          time: 8.0
          numerator: 1
          denominator: 4
        }
        time_signatures {
          time: 8.5
          numerator: 4
          denominator: 4
        }
        time_signatures {
          time: 14.5
          numerator: 3
          denominator: 4
        }
        tempos: {
          qpm: 120
        }
        key_signatures: {
          key: C
        }
        source_info: {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        part_infos {
          part: 0
          name: "Harpsichord"
        }
        part_infos {
          part: 1
          name: "Piano"
        }
        total_time: 16.0
        """)
    pitches_0_1 = [
        (67, .5),

        (64, .5),
        (69, .5),
        (67, .5),
        (72, .5),

        (72, .5),
        (71, .5),
        (72, .5),
        (67, .5),

        (72, .5),
        (67, .5),
        (69, .5),
        (66, .5),

        (67, 1.5),

        (71, .5),

        (72, .5),
        (69, .5),
        (74, .5),
        (71, .5),

        (72, .5),
        (69, .5),
        (71, .5),
        (67, .5),

        (69, .5),
        (72, .5),
        (74, .5),
        (71, .5),

        (72, 1.5),
    ]
    pitches_0_2 = [
        (60, .5),

        (60, .5),
        (60, .5),
        (60, .5),
        (64, .5),

        (62, .5),
        (62, .5),
        (64, .5),
        (64, .5),

        (64, .5),
        (64, .5),
        (64, .5),
        (62, .5),

        (62, 1.5),

        (62, .5),

        (64, .5),
        (60, .5),
        (65, .5),
        (62, .5),

        (64, .75),
        (62, .25),
        (59, .5),
        (60, .5),

        (65, .5),
        (64, .5),
        (62, .5),
        (62, .5),

        (64, 1.5),
    ]
    pitches_1_1 = [
        (52, .5),

        (55, .5),
        (57, .5),
        (60, .5),
        (60, .5),

        (57, .5),
        (55, .5),
        (55, .5),
        (60, .5),

        (60, .5),
        (59, .5),
        (57, .5),
        (57, .5),

        (59, 1.5),

        (55, .5),

        (55, .5),
        (57, .5),
        (57, .5),
        (55, .5),

        (55, .5),
        (57, .5),
        (56, .5),
        (55, .5),

        (53, .5),
        (55, .5),
        (57, .5),
        (55, .5),

        (55, 1.5),
    ]
    pitches_1_2 = [
        (48, .5),

        (48, .5),
        (53, .5),
        (52, .5),
        (57, .5),

        (53, .5),
        (55, .5),
        (48, .5),
        (48, .5),

        (45, .5),
        (52, .5),
        (48, .5),
        (50, .5),

        (43, 1.5),

        (55, .5),

        (48, .5),
        (53, .5),
        (50, .5),
        (55, .5),

        (48, .5),
        (53, .5),
        (52, .5),
        (52, .5),

        (50, .5),
        (48, .5),
        (53, .5),
        (55, .5),

        (48, 1.5),
    ]
    part_voice_instrument_program_pitches = [
        (0, 1, 1, 7, pitches_0_1),
        (0, 2, 1, 7, pitches_0_2),
        (1, 1, 2, 1, pitches_1_1),
        (1, 2, 2, 1, pitches_1_2),
    ]
    for part, voice, instrument, program, pitches in (
        part_voice_instrument_program_pitches):
      time = 0
      for pitch, duration in pitches:
        note = expected_ns.notes.add()
        note.part = part
        note.voice = voice
        note.pitch = pitch
        note.start_time = time
        time += duration
        note.end_time = time
        note.velocity = 64
        note.instrument = instrument
        note.program = program
        if duration == .5:
          note.numerator = 1
          note.denominator = 4
        if duration == .25:
          note.numerator = 1
          note.denominator = 8
        if duration == .75:
          note.numerator = 3
          note.denominator = 8
        if duration == 1.5:
          note.numerator = 3
          note.denominator = 4
    expected_ns.notes.sort(
        key=lambda note: (note.part, note.voice, note.start_time))
    ns.notes.sort(
        key=lambda note: (note.part, note.voice, note.start_time))
    self.assertProtoEquals(expected_ns, ns)

  def test_empty_part_name(self):
    """Verify that a part with an empty name can be parsed."""

    xml = br"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
      <!DOCTYPE score-partwise PUBLIC
          "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
          "http://www.musicxml.org/dtds/partwise.dtd">
      <score-partwise version="3.0">
        <part-list>
          <score-part id="P1">
            <part-name/>
          </score-part>
        </part-list>
        <part id="P1">
        </part>
      </score-partwise>
    """
    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(xml)
      temp_file.flush()
      ns = musicxml_reader.musicxml_file_to_sequence_proto(
          temp_file.name)

    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        key_signatures {
          key: C
          time: 0
        }
        tempos {
          qpm: 120.0
        }
        part_infos {
          part: 0
        }
        total_time: 0.0
        """)
    self.assertProtoEquals(expected_ns, ns)

  def test_empty_part_list(self):
    """Verify that a part without a corresponding score-part can be parsed."""

    xml = br"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
      <!DOCTYPE score-partwise PUBLIC
          "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
          "http://www.musicxml.org/dtds/partwise.dtd">
      <score-partwise version="3.0">
        <part id="P1">
        </part>
      </score-partwise>
    """
    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(xml)
      temp_file.flush()
      ns = musicxml_reader.musicxml_file_to_sequence_proto(
          temp_file.name)

    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        key_signatures {
          key: C
          time: 0
        }
        tempos {
          qpm: 120.0
        }
        part_infos {
          part: 0
        }
        total_time: 0.0
        """)
    self.assertProtoEquals(expected_ns, ns)

  def test_empty_doc(self):
    """Verify that an empty doc can be parsed."""

    xml = br"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
      <!DOCTYPE score-partwise PUBLIC
          "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
          "http://www.musicxml.org/dtds/partwise.dtd">
      <score-partwise version="3.0">
      </score-partwise>
    """
    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(xml)
      temp_file.flush()
      ns = musicxml_reader.musicxml_file_to_sequence_proto(
          temp_file.name)

    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        source_info: {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        key_signatures {
          key: C
          time: 0
        }
        tempos {
          qpm: 120.0
        }
        total_time: 0.0
        """)
    self.assertProtoEquals(expected_ns, ns)

  def test_chord_symbols(self):
    ns = musicxml_reader.musicxml_file_to_sequence_proto(
        self.chord_symbols_filename)
    chord_symbols = [(annotation.time, annotation.text)
                     for annotation in ns.text_annotations
                     if annotation.annotation_type == CHORD_SYMBOL]
    chord_symbols = list(sorted(chord_symbols, key=operator.itemgetter(0)))

    expected_beats_and_chords = [
        (0.0, 'N.C.'),
        (4.0, 'Cmaj7'),
        (12.0, 'F6(add9)'),
        (16.0, 'F#dim7/A'),
        (20.0, 'Bm7b5'),
        (24.0, 'E7(#9)'),
        (28.0, 'A7(add9)(no3)'),
        (32.0, 'Bbsus2'),
        (36.0, 'Am(maj7)'),
        (38.0, 'D13'),
        (40.0, 'E5'),
        (44.0, 'Caug')
    ]

    # Adjust for 120 QPM.
    expected_times_and_chords = [(beat / 2.0, chord)
                                 for beat, chord in expected_beats_and_chords]
    self.assertEqual(expected_times_and_chords, chord_symbols)

  def test_alternating_meter(self):
    with self.assertRaises(musicxml_parser.AlternatingTimeSignatureException):
      musicxml_parser.MusicXMLDocument(self.alternating_meter_filename)

  def test_mid_measure_meter_change(self):
    with self.assertRaises(musicxml_parser.MultipleTimeSignatureException):
      musicxml_parser.MusicXMLDocument(self.mid_measure_meter_filename)

  def test_unpitched_notes(self):
    with self.assertRaises(musicxml_parser.UnpitchedNoteException):
      musicxml_parser.MusicXMLDocument(os.path.join(
          tf.resource_loader.get_data_files_path(),
          'testdata/unpitched.xml'))
    with self.assertRaises(musicxml_reader.MusicXMLConversionError):
      musicxml_reader.musicxml_file_to_sequence_proto(os.path.join(
          tf.resource_loader.get_data_files_path(),
          'testdata/unpitched.xml'))

  def test_empty_archive(self):
    with tempfile.NamedTemporaryFile(suffix='.mxl') as temp_file:
      z = zipfile.ZipFile(temp_file.name, 'w')
      z.close()

      with self.assertRaises(musicxml_reader.MusicXMLConversionError):
        musicxml_reader.musicxml_file_to_sequence_proto(
            temp_file.name)

  def test_whole_measure_rest_forward(self):
    """Test that a whole measure rest can be encoded using <forward>.

    A whole measure rest is usually encoded as a <note> with a duration
    equal to that of a whole measure. An alternative encoding is to
    use the <forward> element to advance the time cursor to a duration
    equal to that of a whole measure. This implies a whole measure rest
    when there are no <note> elements in this measure.
    """
    ns = musicxml_reader.musicxml_file_to_sequence_proto(
        self.whole_measure_rest_forward_filename)
    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        time_signatures {
          numerator: 4
          denominator: 4
        }
        time_signatures {
          time: 6.0
          numerator: 2
          denominator: 4
        }
        key_signatures {
        }
        tempos {
          qpm: 120
        }
        notes {
          pitch: 72
          velocity: 64
          end_time: 2.0
          numerator: 1
          denominator: 1
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 4.0
          end_time: 6.0
          numerator: 1
          denominator: 1
          voice: 1
        }
        notes {
          pitch: 60
          velocity: 64
          start_time: 6.0
          end_time: 7.0
          numerator: 1
          denominator: 2
          voice: 1
        }
        notes {
          pitch: 60
          velocity: 64
          start_time: 8.0
          end_time: 9.0
          numerator: 1
          denominator: 2
          voice: 1
        }
        total_time: 9.0
        part_infos {
          name: "Flute"
        }
        source_info {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        """)
    self.assertProtoEquals(expected_ns, ns)

  def test_meter(self):
    """Test that meters are encoded properly.

    Musical meters are expressed as a ratio of beats to divisions.
    The MusicXML parser uses this ratio in lowest terms for timing
    purposes. However, the meters should be in the actual terms
    when appearing in a NoteSequence.
    """
    ns = musicxml_reader.musicxml_file_to_sequence_proto(
        self.meter_test_filename)
    expected_ns = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        ticks_per_quarter: 220
        time_signatures {
          numerator: 1
          denominator: 4
        }
        time_signatures {
          time: 0.5
          numerator: 2
          denominator: 4
        }
        time_signatures {
          time: 1.5
          numerator: 3
          denominator: 4
        }
        time_signatures {
          time: 3.0
          numerator: 4
          denominator: 4
        }
        time_signatures {
          time: 5.0
          numerator: 5
          denominator: 4
        }
        time_signatures {
          time: 7.5
          numerator: 6
          denominator: 4
        }
        time_signatures {
          time: 10.5
          numerator: 7
          denominator: 4
        }
        time_signatures {
          time: 14.0
          numerator: 1
          denominator: 8
        }
        time_signatures {
          time: 14.25
          numerator: 2
          denominator: 8
        }
        time_signatures {
          time: 14.75
          numerator: 3
          denominator: 8
        }
        time_signatures {
          time: 15.5
          numerator: 4
          denominator: 8
        }
        time_signatures {
          time: 16.5
          numerator: 5
          denominator: 8
        }
        time_signatures {
          time: 17.75
          numerator: 6
          denominator: 8
        }
        time_signatures {
          time: 19.25
          numerator: 7
          denominator: 8
        }
        time_signatures {
          time: 21.0
          numerator: 8
          denominator: 8
        }
        time_signatures {
          time: 23.0
          numerator: 9
          denominator: 8
        }
        time_signatures {
          time: 25.25
          numerator: 10
          denominator: 8
        }
        time_signatures {
          time: 27.75
          numerator: 11
          denominator: 8
        }
        time_signatures {
          time: 30.5
          numerator: 12
          denominator: 8
        }
        time_signatures {
          time: 33.5
          numerator: 2
          denominator: 2
        }
        time_signatures {
          time: 35.5
          numerator: 3
          denominator: 2
        }
        time_signatures {
          time: 38.5
          numerator: 4
          denominator: 2
        }
        time_signatures {
          time: 42.5
          numerator: 4
          denominator: 4
        }
        time_signatures {
          time: 44.5
          numerator: 2
          denominator: 2
        }
        key_signatures {
        }
        tempos {
          qpm: 120
        }
        notes {
          pitch: 72
          velocity: 64
          end_time: 0.5
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 0.5
          end_time: 1.5
          numerator: 1
          denominator: 2
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 1.5
          end_time: 3.0
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 3.0
          end_time: 5.0
          numerator: 1
          denominator: 1
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 5.0
          end_time: 6.5
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 6.5
          end_time: 7.5
          numerator: 1
          denominator: 2
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 7.5
          end_time: 9.0
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 9.0
          end_time: 10.5
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 10.5
          end_time: 12.0
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 12.0
          end_time: 13.5
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 13.5
          end_time: 14.0
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 14.0
          end_time: 14.25
          numerator: 1
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 14.25
          end_time: 14.75
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 14.75
          end_time: 15.5
          numerator: 3
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 15.5
          end_time: 16.0
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 16.0
          end_time: 16.5
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 16.5
          end_time: 17.0
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 17.0
          end_time: 17.5
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 17.5
          end_time: 17.75
          numerator: 1
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 17.75
          end_time: 18.5
          numerator: 3
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 18.5
          end_time: 19.25
          numerator: 3
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 19.25
          end_time: 20.0
          numerator: 3
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 20.0
          end_time: 20.5
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 20.5
          end_time: 21.0
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 21.0
          end_time: 21.75
          numerator: 3
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 21.75
          end_time: 22.5
          numerator: 3
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 22.5
          end_time: 23.0
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 23.0
          end_time: 24.5
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 24.5
          end_time: 25.25
          numerator: 3
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 25.25
          end_time: 26.75
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 26.75
          end_time: 27.25
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 27.25
          end_time: 27.75
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 27.75
          end_time: 29.25
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 29.25
          end_time: 30.0
          numerator: 3
          denominator: 8
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 30.0
          end_time: 30.5
          numerator: 1
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 30.5
          end_time: 32.0
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 32.0
          end_time: 33.5
          numerator: 3
          denominator: 4
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 33.5
          end_time: 34.5
          numerator: 1
          denominator: 2
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 34.5
          end_time: 35.5
          numerator: 1
          denominator: 2
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 35.5
          end_time: 36.5
          numerator: 1
          denominator: 2
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 36.5
          end_time: 37.5
          numerator: 1
          denominator: 2
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 37.5
          end_time: 38.5
          numerator: 1
          denominator: 2
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 38.5
          end_time: 40.5
          numerator: 1
          denominator: 1
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 40.5
          end_time: 42.5
          numerator: 1
          denominator: 1
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 42.5
          end_time: 44.5
          numerator: 1
          denominator: 1
          voice: 1
        }
        notes {
          pitch: 72
          velocity: 64
          start_time: 44.5
          end_time: 46.5
          numerator: 1
          denominator: 1
          voice: 1
        }
        total_time: 46.5
        part_infos {
          name: "Flute"
        }
        source_info {
          source_type: SCORE_BASED
          encoding_type: MUSIC_XML
          parser: MAGENTA_MUSIC_XML
        }
        """)
    self.assertProtoEquals(expected_ns, ns)

  def test_key_missing_fifths(self):
    xml = br"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
      <!DOCTYPE score-partwise PUBLIC
          "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
          "http://www.musicxml.org/dtds/partwise.dtd">
      <score-partwise version="3.0">
        <part-list>
          <score-part id="P1">
            <part-name/>
          </score-part>
        </part-list>
        <part id="P1">
          <measure number="1">
            <attributes>
              <divisions>2</divisions>
              <key>
                <!-- missing fifths element. -->
              </key>
              <time>
                <beats>4</beats>
                <beat-type>4</beat-type>
              </time>
            </attributes>
            <note>
              <pitch>
                <step>G</step>
                <octave>4</octave>
              </pitch>
              <duration>2</duration>
              <voice>1</voice>
              <type>quarter</type>
            </note>
          </measure>
        </part>
      </score-partwise>
    """
    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(xml)
      temp_file.flush()
      with self.assertRaises(musicxml_parser.KeyParseException):
        musicxml_parser.MusicXMLDocument(temp_file.name)

  def test_harmony_missing_degree(self):
    xml = br"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
      <!DOCTYPE score-partwise PUBLIC
          "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
          "http://www.musicxml.org/dtds/partwise.dtd">
      <score-partwise version="3.0">
        <part-list>
          <score-part id="P1">
            <part-name/>
          </score-part>
        </part-list>
        <part id="P1">
          <measure number="1">
            <attributes>
              <divisions>2</divisions>
              <time>
                <beats>4</beats>
                <beat-type>4</beat-type>
              </time>
            </attributes>
            <note>
              <pitch>
                <step>G</step>
                <octave>4</octave>
              </pitch>
              <duration>2</duration>
              <voice>1</voice>
              <type>quarter</type>
            </note>
            <harmony>
              <degree>
                <!-- missing degree-value text -->
                <degree-value></degree-value>
              </degree>
            </harmony>
          </measure>
        </part>
      </score-partwise>
    """
    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(xml)
      temp_file.flush()
      with self.assertRaises(musicxml_parser.ChordSymbolParseException):
        musicxml_parser.MusicXMLDocument(temp_file.name)

  def test_transposed_keysig(self):
    xml = br"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
      <!DOCTYPE score-partwise PUBLIC
          "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
          "http://www.musicxml.org/dtds/partwise.dtd">
      <score-partwise version="3.0">
        <part-list>
          <score-part id="P1">
            <part-name/>
          </score-part>
        </part-list>
        <part id="P1">
          <measure number="1">
          <attributes>
            <divisions>4</divisions>
            <key>
              <fifths>-3</fifths>
              <mode>major</mode>
            </key>
            <time>
              <beats>4</beats>
              <beat-type>4</beat-type>
            </time>
            <clef>
              <sign>G</sign>
              <line>2</line>
            </clef>
            <transpose>
              <diatonic>-5</diatonic>
              <chromatic>-9</chromatic>
            </transpose>
            </attributes>
            <note>
              <pitch>
                <step>G</step>
                <octave>4</octave>
              </pitch>
              <duration>2</duration>
              <voice>1</voice>
              <type>quarter</type>
            </note>
          </measure>
        </part>
      </score-partwise>
    """
    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(xml)
      temp_file.flush()
      musicxml_parser.MusicXMLDocument(temp_file.name)
      sequence = musicxml_reader.musicxml_file_to_sequence_proto(temp_file.name)
      self.assertEqual(1, len(sequence.key_signatures))
      self.assertEqual(music_pb2.NoteSequence.KeySignature.G_FLAT,
                       sequence.key_signatures[0].key)

  def test_beats_composite(self):
    xml = br"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
      <!DOCTYPE score-partwise PUBLIC
          "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
          "http://www.musicxml.org/dtds/partwise.dtd">
      <score-partwise version="3.0">
        <part-list>
          <score-part id="P1">
            <part-name/>
          </score-part>
        </part-list>
        <part id="P1">
          <measure number="1">
            <attributes>
              <divisions>2</divisions>
              <time>
                <beats>4+5</beats>
                <beat-type>4</beat-type>
              </time>
            </attributes>
            <note>
              <pitch>
                <step>G</step>
                <octave>4</octave>
              </pitch>
              <duration>2</duration>
              <voice>1</voice>
              <type>quarter</type>
            </note>
          </measure>
        </part>
      </score-partwise>
    """
    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(xml)
      temp_file.flush()
      with self.assertRaises(musicxml_parser.TimeSignatureParseException):
        musicxml_parser.MusicXMLDocument(temp_file.name)

  def test_invalid_note_type(self):
    xml = br"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
      <!DOCTYPE score-partwise PUBLIC
          "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
          "http://www.musicxml.org/dtds/partwise.dtd">
      <score-partwise version="3.0">
        <part-list>
          <score-part id="P1">
            <part-name/>
          </score-part>
        </part-list>
        <part id="P1">
          <measure number="1">
            <attributes>
              <divisions>2</divisions>
              <time>
                <beats>4</beats>
                <beat-type>4</beat-type>
              </time>
            </attributes>
            <note>
              <pitch>
                <step>G</step>
                <octave>4</octave>
              </pitch>
              <duration>2</duration>
              <voice>1</voice>
              <type>blarg</type>
            </note>
          </measure>
        </part>
      </score-partwise>
    """
    with tempfile.NamedTemporaryFile() as temp_file:
      temp_file.write(xml)
      temp_file.flush()
      with self.assertRaises(musicxml_parser.InvalidNoteDurationTypeException):
        musicxml_parser.MusicXMLDocument(temp_file.name)


if __name__ == '__main__':
  tf.test.main()
