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

from collections import defaultdict
import os.path

# internal imports

import tensorflow as tf
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.music.musicxml_parser import MusicXMLDocument
from magenta.music.musicxml_reader import musicxml_to_sequence_proto


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
  """

  def setUp(self):

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

  def checkmusicxmlandsequence(self, musicxml, sequence_proto):
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
    seq_instruments = defaultdict(lambda: defaultdict(list))
    for seq_note in sequence_proto.notes:
      seq_instruments[
          (seq_note.instrument, seq_note.program)]['notes'].append(seq_note)

    sorted_seq_instrument_keys = sorted(
        seq_instruments.keys(),
        key=lambda (instrument_id, program_id): (instrument_id, program_id))

    self.assertEqual(len(musicxml.parts), len(seq_instruments))
    for musicxml_part, seq_instrument_key in zip(
        musicxml.parts, sorted_seq_instrument_keys):

      seq_instrument_notes = seq_instruments[seq_instrument_key]['notes']
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

  def checkmusicxmltosequence(self, filename):
    """Test the translation from MusicXML to Sequence proto."""
    source_musicxml = MusicXMLDocument(filename)
    sequence_proto = musicxml_to_sequence_proto(source_musicxml)
    self.checkmusicxmlandsequence(source_musicxml, sequence_proto)

  def checkFMajorScale(self, filename):
    """Verify MusicXML scale file.

    Verify that it contains the correct pitches (sounding pitch) and durations.

    Args:
      filename: file to test.
    """

    # Expected QuantizedSequence
    # Sequence tuple = (midi_pitch, velocity, start_seconds, end_seconds)
    expected_quantized_sequence = sequences_lib.QuantizedSequence()
    expected_quantized_sequence.steps_per_quarter = self.steps_per_quarter
    expected_quantized_sequence.qpm = 120.0
    expected_quantized_sequence.time_signature = (
        sequences_lib.QuantizedSequence.TimeSignature(numerator=4,
                                                      denominator=4))
    testing_lib.add_quantized_track_to_sequence(
        expected_quantized_sequence, 0,
        [
            (65, 64, 0, 4), (67, 64, 4, 8), (69, 64, 8, 12),
            (70, 64, 12, 16), (72, 64, 16, 20), (74, 64, 20, 24),
            (76, 64, 24, 28), (77, 64, 28, 32)
        ]
    )

    # Convert MusicXML to QuantizedSequence
    source_musicxml = MusicXMLDocument(filename)
    sequence_proto = musicxml_to_sequence_proto(source_musicxml)
    quantized = sequences_lib.QuantizedSequence()
    quantized.from_note_sequence(sequence_proto, self.steps_per_quarter)

    # Check equality
    self.assertEqual(expected_quantized_sequence, quantized)

  def testsimplemusicxmltosequence(self):
    """Test the simple flute scale MusicXML file."""
    self.checkmusicxmltosequence(self.flute_scale_filename)
    self.checkFMajorScale(self.flute_scale_filename)

  def testcomplexmusicxmltosequence(self):
    """Test the complex band score MusicXML file."""
    self.checkmusicxmltosequence(self.band_score_filename)

  def testtransposedxmltosequence(self):
    """Test the translation from transposed MusicXML to Sequence proto.

    Compare a transposed MusicXML file (clarinet) to an identical untransposed
    sequence (flute).
    """
    untransposed_musicxml = MusicXMLDocument(self.flute_scale_filename)
    transposed_musicxml = MusicXMLDocument(self.clarinet_scale_filename)
    untransposed_proto = musicxml_to_sequence_proto(untransposed_musicxml)
    self.checkmusicxmlandsequence(transposed_musicxml, untransposed_proto)
    self.checkFMajorScale(self.clarinet_scale_filename)

  def testcompressedxmltosequence(self):
    """Test the translation from compressed MusicXML to Sequence proto.

    Compare a compressed MusicXML file to an identical uncompressed sequence.
    """
    uncompressed_musicxml = MusicXMLDocument(self.flute_scale_filename)
    compressed_musicxml = MusicXMLDocument(self.compressed_filename)
    uncompressed_proto = musicxml_to_sequence_proto(uncompressed_musicxml)
    self.checkmusicxmlandsequence(compressed_musicxml, uncompressed_proto)
    self.checkFMajorScale(self.flute_scale_filename)

  def testmultiplecompressedxmltosequence(self):
    """Test the translation from compressed MusicXML with multiple rootfiles.

    The example MXL file contains a MusicXML file of the Flute F Major scale,
    as well as the PNG rendering of the score contained within the single MXL
    file.
    """
    uncompressed_musicxml = MusicXMLDocument(self.flute_scale_filename)
    compressed_musicxml = MusicXMLDocument(
        self.multiple_rootfile_compressed_filename)
    uncompressed_proto = musicxml_to_sequence_proto(uncompressed_musicxml)
    self.checkmusicxmlandsequence(compressed_musicxml, uncompressed_proto)
    self.checkFMajorScale(self.flute_scale_filename)

  def testrhythmdurationsxmltosequence(self):
    """Test the rhythm durations MusicXML file."""
    self.checkmusicxmltosequence(self.rhythm_durations_filename)

if __name__ == '__main__':
  tf.test.main()
