"""Tests to converting MusicXML to NoteSequence proto."""

import os

# internal imports
import tensorflow as tf

from magenta.music.music_xml_io import music_xml_to_sequence_proto


class MusicXmlIOTest(tf.test.TestCase):

  def setUp(self):
    """Get the file path to the test MusicXML file."""
    fname = 'bach-one_phrase-4_voices.xml'
    self.music_xml_fpath = os.path.join(
        tf.resource_loader.get_data_files_path(), 'testdata', fname)

  def testMusicXMLToNoteSequence(self):
    """A few quick tests for converting MusicXML to NoteSequence proto."""
    sequence = music_xml_to_sequence_proto(self.music_xml_fpath)
    self.assertEqual(len(sequence.time_signatures), 2)
    self.assertEqual(len(sequence.key_signatures), 1)
    self.assertEqual(len(sequence.notes), 47)


if __name__ == '__main__':
  tf.test.main()
