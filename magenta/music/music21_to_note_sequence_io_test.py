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
"""Tests for consistency between PrettyMusic21 and NoteSequence proto."""

import os

# internal imports
import music21
import tensorflow as tf

from magenta.music import pretty_music21
from magenta.music.music21_to_note_sequence_io import _MUSIC21_TO_NOTE_SEQUENCE_MODE
from magenta.music.music21_to_note_sequence_io import _PRETTY_MUSIC21_TO_NOTE_SEQUENCE_KEY_NAME
from magenta.music.music21_to_note_sequence_io import music21_to_sequence_proto
from magenta.music.music21_to_note_sequence_io import pretty_music21_to_sequence_proto
from magenta.protobuf import music_pb2


class Music21ScoretoNoteSequenceTest(tf.test.TestCase):

  def setUp(self):
    """Get the file path to the test MusicXML file."""
    fname = 'bach-one_phrase-4_voices.xml'
    self.source_fpath = os.path.join(tf.resource_loader.get_data_files_path(),
                                     'testdata', fname)

  def testMusic21ToSequenceFromMusicXML(self):
    """Test consistency between pretty_music21 and NoteSequence store of XML."""
    parser = music21.musicxml.xmlToM21.MusicXMLImporter()
    music21_score = parser.scoreFromFile(self.source_fpath)
    simple_score = pretty_music21.PrettyMusic21(
        music21_score, os.path.basename(self.source_fpath))
    sequence_proto = music21_to_sequence_proto(
        music21_score, os.path.basename(self.source_fpath))
    self.CompareNoteSequenceAndMusic21Score(sequence_proto, simple_score)

  def testPrettyMusic21ToSequenceFromMusicXML(self):
    """Test consistency between pretty_music21 and NoteSequence store of XML."""
    parser = music21.musicxml.xmlToM21.MusicXMLImporter()
    music21_score = parser.scoreFromFile(self.source_fpath)
    simple_score = pretty_music21.PrettyMusic21(
        music21_score, os.path.basename(self.source_fpath))
    sequence_proto = pretty_music21_to_sequence_proto(
        simple_score, os.path.basename(self.source_fpath))
    self.CompareNoteSequenceAndMusic21Score(sequence_proto, simple_score)

  def testPrettyMusic21ToSequenceFromMusicXMLWithSourceFnamePassedToFormer(
      self):
    """Test consistency between pretty_music21 and NoteSequence store of XML."""
    parser = music21.musicxml.xmlToM21.MusicXMLImporter()
    music21_score = parser.scoreFromFile(self.source_fpath)

    simple_score = pretty_music21.PrettyMusic21(
        music21_score, os.path.basename(self.source_fpath))

    sequence_proto = pretty_music21_to_sequence_proto(simple_score)
    self.assertEqual(sequence_proto.filename, simple_score.filename)

  def CompareNoteSequenceAndMusic21Score(self, sequence_proto, score):
    """Compares a NoteSequence proto to a PrettyMusic21 object.

    Args:
      sequence_proto: A tensorflow.magenta.Sequence proto.
      score: A pretty_music21.PrettyMusic21 object.
    """
    # Test score info.
    self.assertEqual(sequence_proto.source_info.parser,
                     music_pb2.NoteSequence.SourceInfo.MUSIC21)
    self.assertEqual(sequence_proto.filename, score.filename)

    # Test time signature changes.
    self.assertEqual(
        len(score.time_signature_changes), len(sequence_proto.time_signatures))
    for score_time, sequence_time in zip(score.time_signature_changes,
                                         sequence_proto.time_signatures):
      self.assertEqual(score_time.numerator, sequence_time.numerator)
      self.assertEqual(score_time.denominator, sequence_time.denominator)
      self.assertAlmostEqual(score_time.time, sequence_time.time)

    # Test key signature changes.
    self.assertEqual(
        len(score.key_signature_changes), len(sequence_proto.key_signatures))
    for score_key, sequence_key in zip(score.key_signature_changes,
                                       sequence_proto.key_signatures):
      key_pitch_idx = _PRETTY_MUSIC21_TO_NOTE_SEQUENCE_KEY_NAME.values().index(
          sequence_key.key)
      self.assertEqual(
          score_key.key.upper(),
          _PRETTY_MUSIC21_TO_NOTE_SEQUENCE_KEY_NAME.keys()[key_pitch_idx])
      key_mode_idx = _MUSIC21_TO_NOTE_SEQUENCE_MODE.values().index(
          sequence_key.mode)
      self.assertEqual(score_key.mode,
                       _MUSIC21_TO_NOTE_SEQUENCE_MODE.keys()[key_mode_idx])
      self.assertAlmostEqual(score_key.time, sequence_key.time)

    # Test tempos.
    self.assertEqual(len(score.tempo_changes), len(sequence_proto.tempos))
    for score_tempo, sequence_tempo in zip(score.tempo_changes,
                                           sequence_proto.tempos):
      self.assertAlmostEqual(score_tempo.qpm, sequence_tempo.qpm)
      self.assertAlmostEqual(score_tempo.time, sequence_tempo.time)

    # Test part info.
    self.assertEqual(len(score.part_infos), len(sequence_proto.part_infos))
    for score_part_infos, sequence_part_infos in zip(
        score.part_infos, sequence_proto.part_infos):
      self.assertEqual(score_part_infos.index, sequence_part_infos.part)
      self.assertEqual(score_part_infos.name, sequence_part_infos.name)

    # Test parts and notes.
    for score_note, sequence_note in zip(score.sorted_notes,
                                         sequence_proto.notes):
      self.assertAlmostEqual(score_note.pitch_midi, sequence_note.pitch)
      self.assertAlmostEqual(score_note.start_time, sequence_note.start_time)
      self.assertAlmostEqual(score_note.end_time, sequence_note.end_time)
      self.assertEqual(score_note.part_index, sequence_note.part)


if __name__ == '__main__':
  tf.test.main()
