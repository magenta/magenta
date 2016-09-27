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
"""Conversion between Pretty_Music21 score objects and NoteSequence proto."""

# TODO(annahuang): Time in score-based quarter notes, which does not take
# tempo markings into account.

from collections import OrderedDict
import os

# internal imports
from magenta.music import pretty_music21
from magenta.protobuf import music_pb2


_DEFAULT_VELOCITY = 120

_FILE_EXTENSION_TO_NOTE_SEQUENCE_ENCODING_TYPE = {
    '.xml': music_pb2.NoteSequence.SourceInfo.MUSIC_XML,
    '.mxl': music_pb2.NoteSequence.SourceInfo.MUSIC_XML,
    '.abc': music_pb2.NoteSequence.SourceInfo.ABC,
}

# Settings from MuseScore.
# Adopting MuseScore tick-length-values convention where quarter note equal 480.
# https://musescore.org/plugin-development/tick-length-values.
_TICKS_PER_QUARTER_NOTE = 480

_MUSIC21_TO_NOTE_SEQUENCE_MODE = {
    'major': music_pb2.NoteSequence.KeySignature.MAJOR,
    'minor': music_pb2.NoteSequence.KeySignature.MINOR,
    None: music_pb2.NoteSequence.KeySignature.NOT_SPECIFIED
}

_PRETTY_MUSIC21_TO_NOTE_SEQUENCE_PITCH_NAME = {
    'Fbb': music_pb2.NoteSequence.F_FLAT_FLAT,
    'Cbb': music_pb2.NoteSequence.C_FLAT_FLAT,
    'Gbb': music_pb2.NoteSequence.G_FLAT_FLAT,
    'Dbb': music_pb2.NoteSequence.D_FLAT_FLAT,
    'Abb': music_pb2.NoteSequence.A_FLAT_FLAT,
    'Ebb': music_pb2.NoteSequence.E_FLAT_FLAT,
    'Bbb': music_pb2.NoteSequence.B_FLAT_FLAT,
    'Fb': music_pb2.NoteSequence.F_FLAT,
    'Cb': music_pb2.NoteSequence.C_FLAT,
    'Gb': music_pb2.NoteSequence.G_FLAT,
    'Db': music_pb2.NoteSequence.D_FLAT,
    'Ab': music_pb2.NoteSequence.A_FLAT,
    'Eb': music_pb2.NoteSequence.E_FLAT,
    'Bb': music_pb2.NoteSequence.B_FLAT,
    'F': music_pb2.NoteSequence.F,
    'C': music_pb2.NoteSequence.C,
    'G': music_pb2.NoteSequence.G,
    'D': music_pb2.NoteSequence.D,
    'A': music_pb2.NoteSequence.A,
    'E': music_pb2.NoteSequence.E,
    'B': music_pb2.NoteSequence.B,
    'F#': music_pb2.NoteSequence.F_SHARP,
    'C#': music_pb2.NoteSequence.C_SHARP,
    'G#': music_pb2.NoteSequence.G_SHARP,
    'D#': music_pb2.NoteSequence.D_SHARP,
    'A#': music_pb2.NoteSequence.A_SHARP,
    'E#': music_pb2.NoteSequence.E_SHARP,
    'B#': music_pb2.NoteSequence.B_SHARP,
    'F##': music_pb2.NoteSequence.F_SHARP_SHARP,
    'C##': music_pb2.NoteSequence.C_SHARP_SHARP,
    'G##': music_pb2.NoteSequence.G_SHARP_SHARP,
    'D##': music_pb2.NoteSequence.D_SHARP_SHARP,
    'A##': music_pb2.NoteSequence.A_SHARP_SHARP,
    'E##': music_pb2.NoteSequence.E_SHARP_SHARP,
    'B##': music_pb2.NoteSequence.B_SHARP_SHARP
}

_PRETTY_MUSIC21_TO_NOTE_SEQUENCE_KEY_NAME = OrderedDict({
    'C': music_pb2.NoteSequence.KeySignature.C,
    'C#': music_pb2.NoteSequence.KeySignature.C_SHARP,
    'Db': music_pb2.NoteSequence.KeySignature.D_FLAT,
    'D': music_pb2.NoteSequence.KeySignature.D,
    'D#': music_pb2.NoteSequence.KeySignature.D_SHARP,
    'Eb': music_pb2.NoteSequence.KeySignature.E_FLAT,
    'E': music_pb2.NoteSequence.KeySignature.E,
    'F': music_pb2.NoteSequence.KeySignature.F,
    'F#': music_pb2.NoteSequence.KeySignature.F_SHARP,
    'Gb': music_pb2.NoteSequence.KeySignature.G_FLAT,
    'G': music_pb2.NoteSequence.KeySignature.G,
    'G#': music_pb2.NoteSequence.KeySignature.G_SHARP,
    'Ab': music_pb2.NoteSequence.KeySignature.A_FLAT,
    'A': music_pb2.NoteSequence.KeySignature.A,
    'A#': music_pb2.NoteSequence.KeySignature.A_SHARP,
    'Bb': music_pb2.NoteSequence.KeySignature.B_FLAT,
    'B': music_pb2.NoteSequence.KeySignature.B
})


class Music21ConversionError(Exception):
  """Exception for unknown musical objects."""
  pass


def music21_to_sequence_proto(music21_score, filename, collection_name=None):
  """Converts a music21 score object to note sequence proto.

  Args:
    music21_score: A music21 score.
    filename: A string for the source filename from which the score was
        extracted.
    collection_name: An optional string for the collection that the source
        score belonged to.

  Returns:
    A NoteSequence.

  Raises:
    TypeError: If music21_score is already a PrettyMusic21 object.
  """
  if isinstance(music21_score, pretty_music21.PrettyMusic21):
    raise TypeError(
        'Takes a music21.stream.Score object, not a PrettyMusic21 object.')
  score = pretty_music21.PrettyMusic21(music21_score)
  return pretty_music21_to_sequence_proto(
      score, filename, collection_name=collection_name)


def pretty_music21_to_sequence_proto(score, filename=None,
                                     collection_name=None):
  """Converts a pretty_music21 score object to note sequence proto.

  Args:
    score: A PrettyMusic21 object.
    filename: A optional string for the source filename from which the score was
        extracted.
    collection_name: An optional string for the collection that the source
        score belonged to.

  Returns:
    A NoteSequence.

  Raises:
    TypeError: When score is not a PrettyMusic21 object.
    Music21ConversionError: When encountering an unknown mode.
  """
  if not isinstance(score, pretty_music21.PrettyMusic21):
    raise TypeError('Not a PrettyMusic21 object.')
  sequence = music_pb2.NoteSequence()

  # Populate meta data.
  sequence.id = str(score.id)
  sequence.filename = str(filename)
  if score.filename is not None:
    sequence.filename = score.filename
  sequence.collection_name = str(collection_name)
  if score.composer is not None:
    sequence.collection_name = score.composer

  # Populate score source and parser info.
  if filename is None:
    filename = score.filename
  if filename is not None:
    _, file_extension = os.path.splitext(filename)
    if file_extension in _FILE_EXTENSION_TO_NOTE_SEQUENCE_ENCODING_TYPE:
      sequence.source_info.source_type = (
          _FILE_EXTENSION_TO_NOTE_SEQUENCE_ENCODING_TYPE[file_extension])
    else:
      sequence.source_info.source_type = (
          music_pb2.NoteSequence.SourceInfo.UNKNOWN_ENCODING_TYPE)
    sequence.source_info.source_type = (
        music_pb2.NoteSequence.SourceInfo.SCORE_BASED)
  else:
    sequence.source_info.source_type = (
        music_pb2.NoteSequence.SourceInfo.UNKNOWN_SOURCE_TYPE)

  sequence.source_info.parser = music_pb2.NoteSequence.SourceInfo.MUSIC21

  # Set ticks per quarter as ticks per quarter note.
  sequence.ticks_per_quarter = _TICKS_PER_QUARTER_NOTE

  # TODO(annahuang): All time in quarter note length, include performance time.
  sequence.total_time = score.total_time

  # Populate time signatures.
  for score_time in score.time_signature_changes:
    time_signature = sequence.time_signatures.add()
    time_signature.time = score_time.time
    time_signature.numerator = score_time.numerator
    time_signature.denominator = score_time.denominator

  # Populate key signatures.
  for score_key in score.key_signature_changes:
    key_signature = sequence.key_signatures.add()
    key_signature.time = score_key.time
    key_signature.key = score_key.tonic_pitchclass

    # Set key mode.
    mode = score_key.mode
    if mode in _MUSIC21_TO_NOTE_SEQUENCE_MODE:
      key_signature.mode = _MUSIC21_TO_NOTE_SEQUENCE_MODE[mode]
    elif mode is None:
      key_signature.mode = music_pb2.NoteSequence.KeySignature.NOT_SPECIFIED
    else:
      raise Music21ConversionError('Unknown key mode %d' % mode)

  # Populate tempo changes.
  for tempo_change in score.tempo_changes:
    tempo = sequence.tempos.add()
    tempo.time = tempo_change.time
    tempo.qpm = tempo_change.qpm

  # Populate part information.
  part_indices = []
  for info in score.part_infos:
    parts_info = sequence.part_infos.add()
    parts_info.part = info.index
    parts_info.name = info.name
    part_indices.append(info.index)

  # Populate notes.
  for score_note in score.sorted_notes:
    note = sequence.notes.add()
    note.part = score_note.part_index
    note.velocity = _DEFAULT_VELOCITY
    note.start_time = score_note.start_time
    note.end_time = score_note.end_time
    note.pitch = score_note.pitch_midi
    note.pitch_name = _PRETTY_MUSIC21_TO_NOTE_SEQUENCE_PITCH_NAME[
        score_note.pitch_name]

  return sequence
