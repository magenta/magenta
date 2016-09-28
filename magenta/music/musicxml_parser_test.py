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
import tempfile

# internal imports
from musicxml_parser import *
from musicxml_reader import *
import tensorflow as tf

# self.flute_scale_filename contains an F-major scale of 8 quarter notes each

# self.clarinet_scale_filename contains a F-major scale of 8 quarter notes
# each appearing as written pitch. This means the key is written as
# G-major but sounds as F-major. The MIDI pitch numbers must be transposed
# to be input into Magenta

# self.band_score_filename contains a number of instruments in written
# pitch. The score has two time signatures (6/8 and 2/4) and two sounding
# keys (Bb-major and Eb major). The file also contains chords and
# multiple voices (see Oboe part in measure 57), as well as dynamics,
# articulations, slurs, ties, hairpins, grace notes, tempo changes,
# and multiple barline types (double, repeat)

# self.compressed_filename contains the same content as
# self.flute_scale_filename, but compressed in MXL format


class MusicXMLParserTest(tf.test.TestCase):

  def setUp(self):
    self.flute_scale_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/flute_scale.xml')

    self.clarinet_scale_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/clarinet_scale.xml')

    self.band_score_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/el_capitan.xml')

    self.compressed_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/flute_scale.mxl')

  def CheckMusicXMLAndSequence(self, musicxml, sequence_proto):
    """Compares MusicXMLDocument object against a sequence proto.

    Args:
      musicxml: A MusicXMLDocument object.
      sequence_proto: A tensorflow.magenta.Sequence proto.
    """
    # Test time signature changes.
    self.assertEqual(len(musicxml.getTimeSignatures()),
                     len(sequence_proto.time_signatures))
    for musicxml_time, sequence_time in zip(musicxml.getTimeSignatures(),
                                        sequence_proto.time_signatures):
      self.assertEqual(musicxml_time.numerator, sequence_time.numerator)
      self.assertEqual(musicxml_time.denominator, sequence_time.denominator)
      self.assertAlmostEqual(musicxml_time.time_position, sequence_time.time)

    # Test key signature changes.
    self.assertEqual(len(musicxml.getKeySignatures()),
                     len(sequence_proto.key_signatures))
    for musicxml_key, sequence_key in zip(musicxml.getKeySignatures(),
                                      sequence_proto.key_signatures):

      if musicxml_key.mode == "major":
        mode = 0
      elif musicxml_key.mode == "minor":
        mode = 1

      # The Key enum in music.proto does NOT follow MIDI / MusicXML specs
      # Convert from MIDI / MusicXML key to music.proto key
      music_proto_keys = [11, 6, 1, 8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]
      key = music_proto_keys[musicxml_key.key + 7]
      self.assertEqual(key, sequence_key.key)
      self.assertEqual(mode, sequence_key.mode)
      self.assertAlmostEqual(musicxml_key.time_position, sequence_key.time)

    # Test tempos.
    musicxml_tempos = musicxml.getTempos()
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
        self.assertAlmostEqual(musicxml_note.time_position,
                               sequence_note.start_time)
        self.assertAlmostEqual(musicxml_note.time_position \
                               + musicxml_note.seconds,
                               sequence_note.end_time)

  def CheckMusicXMLToSequence(self, filename):
    """Test the translation from MusicXML to Sequence proto."""
    source_musicxml = MusicXMLDocument(filename)
    sequence_proto = musicxml_to_sequence_proto(source_musicxml)
    self.CheckMusicXMLAndSequence(source_musicxml, sequence_proto)

  def testSimpleMusicXMLToSequence(self):
    self.CheckMusicXMLToSequence(self.flute_scale_filename)

  def testComplexMusicXMLToSequence(self):
    self.CheckMusicXMLToSequence(self.band_score_filename)

  def testTransposedMusicXMLToSequence(self):
    """Test the translation from MusicXML to Sequence proto when the music
    is transposed. Compare a transpoed MusicXML file (clarinet) to an
    identical untransposed sequence (flute)
    """
    untransposed_musicxml = MusicXMLDocument(self.flute_scale_filename)
    transposed_musicxml = MusicXMLDocument(self.clarinet_scale_filename)
    untransposed_proto = musicxml_to_sequence_proto(untransposed_musicxml)
    self.CheckMusicXMLAndSequence(transposed_musicxml, untransposed_proto)

  def testCompressedMusicXMLToSequence(self):
    """Test the translation from MusicXML to Sequence proto when the music
    is compressed in MXL format. Compare a compressed MusicXML file to an
    identical uncompressed sequence
    """
    uncompressed_musicxml = MusicXMLDocument(self.flute_scale_filename)
    compressed_musicxml = MusicXMLDocument(self.compressed_filename)
    uncompressed_proto = musicxml_to_sequence_proto(uncompressed_musicxml)
    self.CheckMusicXMLAndSequence(compressed_musicxml, uncompressed_proto)

if __name__ == '__main__':
  tf.test.main()
