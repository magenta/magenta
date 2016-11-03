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
"""MusicXML import.

Input wrappers for converting MusicXML into tensorflow.magenta.NoteSequence.
"""

# internal imports

from magenta.music.musicxml_parser import MusicXMLDocument
from magenta.protobuf import music_pb2


class MusicXMLConversionError(Exception):
  """MusicXML conversion error handler."""
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
  musicxml_time_signatures = musicxml_document.get_time_signatures()
  for musicxml_time_signature in musicxml_time_signatures:
    time_signature = sequence.time_signatures.add()
    time_signature.time = musicxml_time_signature.time_position
    time_signature.numerator = musicxml_time_signature.numerator
    time_signature.denominator = musicxml_time_signature.denominator

  # Populate key signatures.
  musicxml_key_signatures = musicxml_document.get_key_signatures()
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
  musicxml_tempos = musicxml_document.get_tempos()
  for musicxml_tempo in musicxml_tempos:
    tempo = sequence.tempos.add()
    tempo.time = musicxml_tempo.time_position
    tempo.qpm = musicxml_tempo.qpm

  # Populate notes from each MusicXML part across all voices
  # Unlike MIDI import, notes are not sorted
  sequence.total_time = musicxml_document.total_time_secs
  for musicxml_part in musicxml_document.parts:
    for musicxml_measure in musicxml_part.measures:
      for musicxml_note in musicxml_measure.notes:
        if not musicxml_note.is_rest:
          note = sequence.notes.add()
          note.instrument = musicxml_note.midi_channel
          note.program = musicxml_note.midi_program
          note.start_time = musicxml_note.note_duration.time_position

          # Fix negative time errors from incorrect MusicXML
          if note.start_time < 0:
            note.start_time = 0

          note.end_time = note.start_time + musicxml_note.note_duration.seconds
          note.pitch = musicxml_note.pitch[1]  # Index 1 = MIDI pitch number
          note.velocity = musicxml_note.velocity

          durationratio = musicxml_note.note_duration.duration_ratio()
          note.numerator = durationratio.numerator
          note.denominator = durationratio.denominator

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
