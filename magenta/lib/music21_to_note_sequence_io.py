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

""" Music 21 ops.

Input and output wrappers for converting between Music21 score objects and NoteSequence proto.
"""

import sys

import pretty_music21
import tensorflow as tf

from magenta.protobuf import music_pb2

# Settings from MuseScore.
# Using the MuseScore tick-length-values convention where quarter note equals 480.
# https://musescore.org/plugin-development/tick-length-values.
_TICKS_PER_QUARTER_NOTE = 480

# https://musescore.org/en/plugin-development/tonal-pitch-class-enum
_MUSESCORE_PITCH_CLASS_ENUM = {'Fbb': -1, 'Cbb': 0, 'Gbb': 1, 'Dbb': 2, 'Abb': 3, 'Ebb': 4, 'Bbb': 5,
                               'Fb': 6, 'Cb': 7, 'Gb': 8, 'Db': 9, 'Ab': 10, 'Eb': 11, 'Bb': 12,
                               'F': 13, 'C': 14, 'G': 15, 'D': 16, 'A': 17, 'E': 18, 'B': 19,
                               'F#': 20, 'C#': 21, 'G#': 22, 'D#': 23, 'A#': 24, 'E#': 25, 'B#': 26,
                               'F##': 27, 'C##': 28, 'G##': 29, 'D##': 30, 'A##': 31, 'E##': 32, 'B##': 33}
_MUSIC21_TO_NOTE_SEQUENCE_KEY = {-6: 6, -5: 1, -4: 8, -3: 3, -2: 10, -1: 5,
                                 0: 0, 1: 7, 2: 2, 3: 9, 4: 4, 5: 11, 6: 6}
_MUSIC21_TO_NOTE_SEQUENCE_MODE = {'major': 0, 'minor': 1}


class Music21ConversionError(Exception):
  pass


def music21_to_sequence_proto(score_data, default_collection_name='N/A',
                              default_filename='N/A', continue_on_exception=False, verbose=False):
  # TODO: Time in score-based quarter notes, which does not take tempo markings into account.
  if isinstance(score_data, pretty_music21.PrettyMusic21):
    score = score_data
  else:
    try:
      score = pretty_music21.PrettyMusic21(score_data)
    except:
      if continue_on_exception:
        tf.logging.error('Music21 score decoding error %s: %s', sys.exc_info()[0],
                         sys.exc_info()[1])
        return None
      else:
        raise Music21ConversionError('Music21 score decoding error %s: %s',
                                     sys.exc_info()[0], sys.exc_info()[1])

  sequence = music_pb2.NoteSequence()
  # Populate header.
  # TODO: ID not necessarily unique if put into other database
  sequence.id = score.id

  sequence.filename = default_filename
  if score.title is not None:
    sequence.filename = score.title

  sequence.collection_name = default_collection_name
  if score.composer is not None:
    sequence.collection_name = score.composer

  sequence.ticks_per_beat = _TICKS_PER_QUARTER_NOTE

  # TODO: total_time in quarter note length
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
    key_signature.key = score_key.key_number
    score_mode = score_key.mode
    if score_mode == 0:
      key_signature.mode = key_signature.MAJOR
    elif score_mode == 1:
      key_signature.mode = key_signature.MINOR
    else:
      raise Music21ConversionError('Invalid key mode %d' % score_mode)

  # Populate tempo changes.
  for tempo_change in score.tempo_changes:
    tempo = sequence.tempos.add()
    tempo.time = tempo_change.time
    tempo.bpm = tempo_change.bpm

  # Populate part information.
  for info in score.part_infos:
    part_info = sequence.part_info.add()
    part_info.part = info.index
    part_info.name = info.name

  # Populate notes.
  for score_note in score.sorted_notes:
      note = sequence.notes.add()
      note.part = score_note.part
      note.start_time = score_note.start
      note.end_time = score_note.end
      note.pitch = score_note.pitch
      # TODO: pitch_class not in proto yet, could create a pull request
      # if score_note.pitch_class is not None:
      #    note.pitch_class = score_note.pitch_class

  return sequence
