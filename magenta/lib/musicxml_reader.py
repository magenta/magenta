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
"""MusicXML import.

Input wrappers for converting MusicXML into tensorflow.magenta.NoteSequence.
"""

from collections import defaultdict
import sys
# pylint: disable=g-import-not-at-top
if sys.version_info.major <= 2:
  from cStringIO import StringIO
else:
  from io import StringIO


# internal imports
import tensorflow as tf
from musicxml_parser import *
from magenta.protobuf import music_pb2
# pylint: enable=g-import-not-at-top

class MusicXMLConversionError(Exception):
    pass

def musicxml_to_sequence_proto(musicxml_document):
  """Convert MusicXML file contents to a tensorflow.magenta.NoteSequence proto.

  Converts a MusicXML file encoded as a string into a
  tensorflow.magenta.NoteSequence proto.

  Args:
    musicxml_document: A parsed MusicXML file.
    This file has been parsed by class MusicXMLDocument

  Returns:
    A tensorflow.magenta.NoteSequence proto.

  Raises:
    MusicXMLConversionError: An error occurred when parsing the MusicXML file.
  """

  sequence = music_pb2.NoteSequence()

  # Populate header.
  sequence.ticks_per_quarter = musicxml_document.midi_resolution

  # Populate time signatures.
  musicxml_time_signatures = musicxml_document.getTimeSignatures()
  for musicxml_time_signature in musicxml_time_signatures:
    time_signature = sequence.time_signatures.add()
    time_signature.time = musicxml_time_signature.time_position
    time_signature.numerator = musicxml_time_signature.numerator
    time_signature.denominator = musicxml_time_signature.denominator

  # Populate key signatures.
  musicxml_key_signatures = musicxml_document.getKeySignatures()
  for musicxml_key in musicxml_key_signatures:
    key_signature = sequence.key_signatures.add()
    key_signature.time = musicxml_key.time_position
    # The Key enum in music.proto does NOT follow MIDI / MusicXML specs
    # Convert from MIDI / MusicXML key to music.proto key
    music_proto_keys = [11, 6, 1, 8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]
    key_signature.key = music_proto_keys[musicxml_key.key + 7]
    if musicxml_key.mode == "major":
      key_signature.mode = key_signature.MAJOR
    elif musicxml_key.mode == "minor":
      key_signature.mode = key_signature.MINOR

  # Populate tempo changes.
  musicxml_tempos = musicxml_document.getTempos()
  for musicxml_tempo in musicxml_tempos:
    tempo = sequence.tempos.add()
    tempo.time = musicxml_tempo.time_position
    tempo.qpm = musicxml_tempo.bpm

  # Populate notes from each MusicXML part across all voices
  # Unlike MIDI import, notes are not sorted
  sequence.total_time = musicxml_document.total_time
  for musicxml_part in musicxml_document.parts:
    for musicxml_measure in musicxml_part.measures:
      for musicxml_note in musicxml_measure.notes:
        if not musicxml_note.is_rest:
          note = sequence.notes.add()
          note.instrument = musicxml_note.midi_channel
          note.program = musicxml_note.midi_program
          note.start_time = musicxml_note.time_position
          note.end_time = musicxml_note.time_position + musicxml_note.seconds
          note.pitch = musicxml_note.pitch[1] # Index 1 = MIDI pitch number
          note.velocity = musicxml_note.velocity

  # TODO(@douglaseck): Estimate note type (e.g. quarter note) and populate
  # note.numerator and note.denominator.

  return sequence

def musicxml_file_to_sequence_proto(musicxml_file):
  """Converts a MusicXML file to a tensorflow.magenta.NoteSequence proto.

  Args:
    musicxml_file: A string path to a MusicXML file.

  Returns:
    A tensorflow.magenta.Sequence proto.

  Raises:
    MusicXMLConversionError: Invalid musicxml_file.
  """
  musicxml_document = MusicXMLDocument(musicxml_file)
  return musicxml_to_sequence_proto(musicxml_document)
