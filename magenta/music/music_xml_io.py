"""Convert MusicXML files to NoteSequence protos, currently uses music21.

Typical usage example:
  note_sequence = music_xml_to_sequence_proto('/path/to/file/mysong.xml')
"""

import os

# internal imports
import music21

from magenta.music.music21_to_note_sequence_io import music21_to_sequence_proto


def music_xml_to_sequence_proto(musicxml_fpath):
  """Converts a MusicXML file into NoteSequence proto.

  Args:
    musicxml_fpath: A string of the absolute path, including filename, to the
        MusicXML file to be parsed and converted.

  Returns:
    A NoteSequence.
  """
  parser = music21.musicxml.xmlToM21.MusicXMLImporter()
  music21_score = parser.scoreFromFile(musicxml_fpath)
  sequence_proto = music21_to_sequence_proto(music21_score,
                                             os.path.basename(musicxml_fpath))
  return sequence_proto
