# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Parser for ABC files.

http://abcnotation.com/wiki/abc:standard:v2.1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fractions import Fraction
import re

# internal imports

import six
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

from magenta.music import constants
from magenta.protobuf import music_pb2


def parse_tunebook_file(filename):
  """Parse an ABC Tunebook file."""
  # 'r' mode will decode the file as utf-8 in py3.
  return parse_tunebook(tf.gfile.Open(filename, 'r').read())


def parse_tunebook(tunebook):
  """Parse an ABC Tunebook string."""
  # Split tunebook into sections based on empty lines.
  sections = []
  current_lines = []
  for line in tunebook.splitlines():
    line = line.strip()
    if not line:
      if current_lines:
        sections.append(current_lines)
        current_lines = []
    else:
      current_lines.append(line)
  if current_lines:
    sections.append(current_lines)

  # If there are multiple sections, the first one may be a header.
  # The first section is a header if it does not contain an X information field.
  header = []
  if len(sections) > 1 and not any(
      [line.startswith('X:') for line in sections[0]]):
    header = sections.pop(0)

  # The header sets default values for each tune, so prepend it to every
  # tune that is being parsed.
  return [ABCTune(header + tune).note_sequence for tune in sections]


class ABCTune(object):
  """Class for parsing an individual ABC tune."""

  # http://abcnotation.com/wiki/abc:standard:v2.1#decorations
  DECORATION_TO_VELOCITY = {
      '!pppp!': 30,
      '!ppp!': 30,
      '!pp!': 45,
      '!p!': 60,
      '!mp!': 75,
      '!mf!': 90,
      '!f!': 105,
      '!ff!': 120,
      '!fff!': 127,
      '!ffff!': 127,
  }

  # http://abcnotation.com/wiki/abc:standard:v2.1#pitch
  ABC_NOTE_TO_MIDI = {
      'C': 60,
      'D': 62,
      'E': 64,
      'F': 65,
      'G': 67,
      'A': 69,
      'B': 71,
      'c': 72,
      'd': 74,
      'e': 76,
      'f': 77,
      'g': 79,
      'a': 81,
      'b': 83,
  }

  # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
  SIG_TO_KEYS = {
      7: ['C#', 'A#m', 'G#Mix', 'D#Dor', 'E#Phr', 'F#Lyd', 'B#Loc'],
      6: ['F#', 'D#m', 'C#Mix', 'G#Dor', 'A#Phr', 'BLyd', 'E#Loc'],
      5: ['B', 'G#m', 'F#Mix', 'C#Dor', 'D#Phr', 'ELyd', 'A#Loc'],
      4: ['E', 'C#m', 'BMix', 'F#Dor', 'G#Phr', 'ALyd', 'D#Loc'],
      3: ['A', 'F#m', 'EMix', 'BDor', 'C#Phr', 'DLyd', 'G#Loc'],
      2: ['D', 'Bm', 'AMix', 'EDor', 'F#Phr', 'GLyd', 'C#Loc'],
      1: ['G', 'Em', 'DMix', 'ADor', 'BPhr', 'CLyd', 'F#Loc'],
      0: ['C', 'Am', 'GMix', 'DDor', 'EPhr', 'FLyd', 'BLoc'],
      -1: ['F', 'Dm', 'CMix', 'GDor', 'APhr', 'BbLyd', 'ELoc'],
      -2: ['Bb', 'Gm', 'FMix', 'CDor', 'DPhr', 'EbLyd', 'ALoc'],
      -3: ['Eb', 'Cm', 'BbMix', 'FDor', 'GPhr', 'AbLyd', 'DLoc'],
      -4: ['Ab', 'Fm', 'EbMix', 'BbDor', 'CPhr', 'DbLyd', 'GLoc'],
      -5: ['Db', 'Bbm', 'AbMix', 'EbDor', 'FPhr', 'GbLyd', 'CLoc'],
      -6: ['Gb', 'Ebm', 'DbMix', 'AbDor', 'BbPhr', 'CbLyd', 'FLoc'],
      -7: ['Cb', 'Abm', 'GbMix', 'DbDor', 'EbPhr', 'FbLyd', 'BbLoc'],
  }

  KEY_TO_SIG = {}
  for sig, keys in six.iteritems(SIG_TO_KEYS):
    for key in keys:
      KEY_TO_SIG[key.lower()] = sig

  KEY_TO_PROTO_KEY = {
      'c': music_pb2.NoteSequence.KeySignature.C,
      'c#': music_pb2.NoteSequence.KeySignature.C_SHARP,
      'db': music_pb2.NoteSequence.KeySignature.D_FLAT,
      'd': music_pb2.NoteSequence.KeySignature.D,
      'd#': music_pb2.NoteSequence.KeySignature.D_SHARP,
      'eb': music_pb2.NoteSequence.KeySignature.E_FLAT,
      'e': music_pb2.NoteSequence.KeySignature.E,
      'f': music_pb2.NoteSequence.KeySignature.F,
      'f#': music_pb2.NoteSequence.KeySignature.F_SHARP,
      'gb': music_pb2.NoteSequence.KeySignature.G_FLAT,
      'g': music_pb2.NoteSequence.KeySignature.G,
      'g#': music_pb2.NoteSequence.KeySignature.G_SHARP,
      'ab': music_pb2.NoteSequence.KeySignature.A_FLAT,
      'a': music_pb2.NoteSequence.KeySignature.A,
      'a#': music_pb2.NoteSequence.KeySignature.A_SHARP,
      'bb': music_pb2.NoteSequence.KeySignature.B_FLAT,
      'b': music_pb2.NoteSequence.KeySignature.B,
  }

  SHARPS_ORDER = 'FCGDAEB'
  FLATS_ORDER = 'BEADGCF'

  INFORMATION_FIELD_PATTERN = re.compile(r'([A-Za-z]):\s*(.*)')

  def __init__(self, tune_lines):
    self._ns = music_pb2.NoteSequence()
    # Standard ABC fields.
    self._ns.source_info.source_type = (
        music_pb2.NoteSequence.SourceInfo.SCORE_BASED)
    self._ns.source_info.encoding_type = (
        music_pb2.NoteSequence.SourceInfo.ABC)
    self._ns.source_info.parser = (
        music_pb2.NoteSequence.SourceInfo.MAGENTA_ABC)
    self._ns.ticks_per_quarter = constants.STANDARD_PPQ

    self._current_time = 0
    self._accidentals = ABCTune._sig_to_accidentals(0)
    self._current_unit_note_length = None

    # Default dynamic should be !mf! as per:
    # http://abcnotation.com/wiki/abc:standard:v2.1#decorations
    self._current_velocity = ABCTune.DECORATION_TO_VELOCITY['!mf!']

    self._in_header = True
    self._header_tempo_unit = None
    self._header_tempo_rate = None
    for line in tune_lines:
      line = re.sub('%.*$', '', line)  # Strip comments.
      line = line.strip()  # Strip whitespace.
      if not line:
        continue

      # If the lines begins with a letter and a colon, it's an information
      # field. Extract it.
      info_field_match = ABCTune.INFORMATION_FIELD_PATTERN.match(line)
      if info_field_match:
        self._parse_information_field(
            info_field_match.group(1), info_field_match.group(2))
      else:
        if self._in_header:
          self._set_values_from_header()
          self._in_header = False
        self._parse_music_code(line)
    if self._in_header:
      self._set_values_from_header()

  @property
  def note_sequence(self):
    return self._ns

  @staticmethod
  def _sig_to_accidentals(sig):
    accidentals = {pitch: 0 for pitch in 'ABCDEFG'}
    if sig > 0:
      for i in range(sig):
        accidentals[ABCTune.SHARPS_ORDER[i]] = 1
    elif sig < 0:
      for i in range(abs(sig)):
        accidentals[ABCTune.FLATS_ORDER[i]] = -1
    return accidentals

  @property
  def _qpm(self):
    """Returns the current QPM."""
    if self._ns.tempos:
      return self._ns.tempos[-1].qpm
    else:
      # No QPM has been specified, so will use the default one.
      return constants.DEFAULT_QUARTERS_PER_MINUTE

  def _set_values_from_header(self):
    # Set unit note length. May depend on the current meter, so this has to be
    # calculated at the end of the header.
    self._set_unit_note_length_from_header()

    # Set the tempo if it was specified in the header. May depend on current
    # unit note length, so has to be calculated after that is set.
    # _header_tempo_unit may be legitimately None, so check _header_tempo_rate.
    if self._header_tempo_rate:
      self._add_tempo(self._header_tempo_unit, self._header_tempo_rate)

  def _set_unit_note_length_from_header(self):
    """Sets the current unit note length.

    Should be called immediately after parsing the header.

    Raises:
      ValueError: If multiple time signatures were set in the header.
    """
    # http://abcnotation.com/wiki/abc:standard:v2.1#lunit_note_length

    if self._current_unit_note_length:
      # If it has been set explicitly, leave it as is.
      pass
    elif not self._ns.time_signatures:
      # For free meter, the default unit note length is 1/8.
      self._current_unit_note_length = Fraction(1, 8)
    else:
      # Otherwise, base it on the current meter.
      if len(self._ns.time_signatures) != 1:
        raise ValueError('Multiple time signatures set in header.')
      current_ts = self._ns.time_signatures[0]
      ratio = current_ts.numerator / current_ts.denominator
      if ratio < 0.75:
        self._current_unit_note_length = Fraction(1, 16)
      else:
        self._current_unit_note_length = Fraction(1, 8)

  def _add_tempo(self, tempo_unit, tempo_rate):
    if tempo_unit is None:
      tempo_unit = self._current_unit_note_length

    tempo = self._ns.tempos.add()
    tempo.time = self._current_time
    tempo.qpm = float((tempo_unit / Fraction(1, 4)) * tempo_rate)

  def _apply_broken_rhythm(self, broken_rhythm):
    """Applies a broken rhythm symbol to the two most recently added notes."""
    # http://abcnotation.com/wiki/abc:standard:v2.1#broken_rhythm

    if len(self._ns.notes) < 2:
      raise ValueError('Cannot apply a broken rhythm with fewer than 2 notes')

    note1 = self._ns.notes[-2]
    note2 = self._ns.notes[-1]
    note1_len = note1.end_time - note1.start_time
    note2_len = note2.end_time - note2.start_time
    if note1_len != note2_len:
      raise ValueError(
          'Cannot apply broken rhythm to two notes of different lengths')

    time_adj = note1_len / (2 ** len(broken_rhythm))
    if broken_rhythm[0] == '<':
      note1.end_time -= time_adj
      note2.start_time -= time_adj
    elif broken_rhythm[0] == '>':
      note1.end_time += time_adj
      note2.start_time += time_adj
    else:
      raise ValueError('Could not parse broken rhythm token: {}'.format(
          broken_rhythm))

  # http://abcnotation.com/wiki/abc:standard:v2.1#pitch
  NOTE_PATTERN = re.compile(
      r'(__|_|=|\^|\^\^)?([A-Ga-g])([\',]*)(\d*/*\d*)')

  # http://abcnotation.com/wiki/abc:standard:v2.1#broken_rhythm
  BROKEN_RHYTHM_PATTERN = re.compile(r'(<+|>+)')

  # http://abcnotation.com/wiki/abc:standard:v2.1#use_of_fields_within_the_tune_body
  INLINE_INFORMATION_FIELD_PATTERN = re.compile(r'\[([A-Za-z]):\s*([^\]]+)\]')

  def _parse_music_code(self, line):
    """Parse the music code within an ABC file."""

    # http://abcnotation.com/wiki/abc:standard:v2.1#the_tune_body
    pos = 0
    broken_rhythm = None
    while pos < len(line):
      char = line[pos]

      note_match = ABCTune.NOTE_PATTERN.match(line, pos)
      broken_rhythm_match = ABCTune.BROKEN_RHYTHM_PATTERN.match(line, pos)
      inline_information_field_match = (
          ABCTune.INLINE_INFORMATION_FIELD_PATTERN.match(line, pos))
      if note_match:
        pos = note_match.end()

        note = self._ns.notes.add()
        note.velocity = self._current_velocity
        note.start_time = self._current_time

        note.pitch = ABCTune.ABC_NOTE_TO_MIDI[note_match.group(2)]

        # Accidentals
        if note_match.group(1):
          for accidental in note_match.group(1).split():
            if accidental == '^':
              note.pitch += 1
            elif accidental == '_':
              note.pitch -= 1
            elif accidental == '=':
              pass
            else:
              raise ValueError('Invalid accidental: {}'.format(accidental))
        else:
          # No accidentals, so modify according to current key.
          note.pitch += self._accidentals[note_match.group(2).upper()]

        # Octaves
        if note_match.group(3):
          for octave in note_match.group(3):
            if octave == '\'':
              note.pitch += 12
            elif octave == ',':
              note.pitch -= 12
            else:
              raise ValueError('Invalid octave: {}'.format(octave))

        if (note.pitch < constants.MIN_MIDI_PITCH or
            note.pitch > constants.MAX_MIDI_PITCH):
          raise ValueError('pitch {} is invalid'.format(note.pitch))

        # Note length
        length = self._current_unit_note_length
        # http://abcnotation.com/wiki/abc:standard:v2.1#note_lengths
        if note_match.group(4):
          slash_count = note_match.group(4).count('/')
          if slash_count == len(note_match.group(4)):
            # Handle A// shorthand case.
            length /= 2 ** slash_count
          elif note_match.group(4).startswith('/'):
            length /= int(note_match.group(4)[1:])
          else:
            length *= int(note_match.group(4))

        # Advance clock based on note length.
        self._current_time += (1 / (self._qpm / 60)) * (length / Fraction(1, 4))

        note.end_time = self._current_time

        if broken_rhythm:
          self._apply_broken_rhythm(broken_rhythm)
          broken_rhythm = None
      elif broken_rhythm_match:
        pos = broken_rhythm_match.end()
        if broken_rhythm:
          raise ValueError('Cannot specify a broken rhythm twice in a row.')
        broken_rhythm = broken_rhythm_match.group(1)
      elif inline_information_field_match:
        pos = inline_information_field_match.end()
        self._parse_information_field(
            inline_information_field_match.group(1),
            inline_information_field_match.group(2))
      elif char == '"':
        # Text annotation
        # http://abcnotation.com/wiki/abc:standard:v2.1#chord_symbols
        # http://abcnotation.com/wiki/abc:standard:v2.1#annotations
        endpos = line.find('"', pos + 1)
        if endpos == -1:
          raise ValueError('Could not find end of text annotation')
        annotation = line[pos + 1:endpos]
        pos = endpos + 1

        ta = self._ns.text_annotations.add()
        ta.time = self._current_time
        ta.text = annotation
        if annotation[0] in ABCTune.ABC_NOTE_TO_MIDI:
          # http://abcnotation.com/wiki/abc:standard:v2.1#chord_symbols
          ta.annotation_type = (
              music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL)
        else:
          ta.annotation_type = (
              music_pb2.NoteSequence.TextAnnotation.UNKNOWN)
      else:
        pos += 1

  # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
  KEY_PATTERN = re.compile(
      r'([A-G])\s*([#b]?)\s*'
      r'((?:(?:maj|ion|min|aeo|mix|dor|phr|lyd|loc|m)[^ ]*)?)',
      re.IGNORECASE)

  # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
  KEY_ACCIDENTALS_PATTERN = re.compile(r'(__|_|=|\^|\^\^)?([A-Ga-g])')

  @staticmethod
  def parse_key(key):
    """Parse an ABC key string."""

    # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
    key_match = ABCTune.KEY_PATTERN.match(key)
    if not key_match:
      raise ValueError('Could not parse key: {}'.format(key))

    key_components = list(key_match.groups())

    # Shorten the mode to be at most 3 letters long.
    mode = key_components[2][:3].lower()

    # "Minor" and "Aeolian" are special cases that are abbreviated to 'm'.
    # "Major" and "Ionian" are special cases that are abbreviated to ''.
    if mode == 'min' or mode == 'aeo':
      mode = 'm'
    elif mode == 'maj' or mode == 'ion':
      mode = ''

    sig = ABCTune.KEY_TO_SIG[''.join(key_components[0:2] + [mode]).lower()]

    proto_key = ABCTune.KEY_TO_PROTO_KEY[''.join(key_components[0:2]).lower()]

    if mode == '':  # pylint: disable=g-explicit-bool-comparison
      proto_mode = music_pb2.NoteSequence.KeySignature.MAJOR
    elif mode == 'm':
      proto_mode = music_pb2.NoteSequence.KeySignature.MINOR
    elif mode == 'mix':
      proto_mode = music_pb2.NoteSequence.KeySignature.MIXOLYDIAN
    elif mode == 'dor':
      proto_mode = music_pb2.NoteSequence.KeySignature.DORIAN
    elif mode == 'phr':
      proto_mode = music_pb2.NoteSequence.KeySignature.PHRYGIAN
    elif mode == 'lyd':
      proto_mode = music_pb2.NoteSequence.KeySignature.LYDIAN
    elif mode == 'loc':
      proto_mode = music_pb2.NoteSequence.KeySignature.LOCRIAN
    else:
      raise ValueError('Unknown mode: {}'.format(mode))

    # Match the rest of the string for possible modifications.
    pos = key_match.end()
    exppos = key[pos:].find('exp')
    if exppos != -1:
      # Only explicit accidentals will be used.
      accidentals = ABCTune._sig_to_accidentals(0)
      pos += exppos + 3
    else:
      accidentals = ABCTune._sig_to_accidentals(sig)

    while pos < len(key):
      note_match = ABCTune.KEY_ACCIDENTALS_PATTERN.match(key, pos)
      if note_match:
        pos += len(note_match.group(0))

        note = note_match.group(2).upper()
        if note_match.group(1):
          if note_match.group(1) == '^':
            accidentals[note] = 1
          elif note_match.group(1) == '_':
            accidentals[note] = -1
          elif note_match.group(1) == '=':
            accidentals[note] = 0
          else:
            raise ValueError(
                'Invalid accidental: {}'.format(note_match.group(1)))
      else:
        pos += 1

    return accidentals, proto_key, proto_mode

  # http://abcnotation.com/wiki/abc:standard:v2.1#outdated_information_field_syntax
  # This syntax is deprecated but must still be supported.
  TEMPO_DEPRECATED_PATTERN = re.compile(r'C?\s*=?\s*(\d+)$')

  # http://abcnotation.com/wiki/abc:standard:v2.1#qtempo
  TEMPO_PATTERN = re.compile(r'(?:"[^"]*")?\s*((?:\d+/\d+\s*)+)\s*=\s*(\d+)')
  TEMPO_PATTERN_STRING_ONLY = re.compile(r'"([^"]*)"$')

  def _parse_information_field(self, field_name, field_content):
    # http://abcnotation.com/wiki/abc:standard:v2.1#information_fields
    if field_name == 'A':
      pass
    elif field_name == 'B':
      pass
    elif field_name == 'C':
      # Composer
      # http://abcnotation.com/wiki/abc:standard:v2.1#ccomposer
      self._ns.sequence_metadata.composers.append(field_content)

      # The first composer will be set as the primary artist.
      if not self._ns.sequence_metadata.artist:
        self._ns.sequence_metadata.artist = field_content
    elif field_name == 'D':
      pass
    elif field_name == 'F':
      pass
    elif field_name == 'G':
      pass
    elif field_name == 'H':
      pass
    elif field_name == 'I':
      pass
    elif field_name == 'K':
      # Key
      # http://abcnotation.com/wiki/abc:standard:v2.1#kkey
      accidentals, proto_key, proto_mode = ABCTune.parse_key(field_content)
      self._accidentals = accidentals
      ks = self._ns.key_signatures.add()
      ks.key = proto_key
      ks.mode = proto_mode
      ks.time = self._current_time
    elif field_name == 'L':
      # Unit note length
      # http://abcnotation.com/wiki/abc:standard:v2.1#lunit_note_length
      length = field_content.split('/', 1)

      # Handle the case of L:1 being equivalent to L:1/1
      if len(length) < 2:
        length.append('1')

      try:
        numerator = int(length[0])
        denominator = int(length[1])
      except ValueError as e:
        raise ValueError(e, 'Could not parse unit note length: {}'.format(
            field_content))

      self._current_unit_note_length = Fraction(numerator, denominator)
    elif field_name == 'M':
      # Meter
      # http://abcnotation.com/wiki/abc:standard:v2.1#mmeter
      if field_content.upper() == 'C':
        ts = self._ns.time_signatures.add()
        ts.numerator = 4
        ts.denominator = 4
        ts.time = self._current_time
      elif field_content.upper() == 'C|':
        ts = self._ns.time_signatures.add()
        ts.numerator = 2
        ts.denominator = 2
        ts.time = self._current_time
      elif field_content.lower() == 'none':
        pass
      else:
        timesig = field_content.split('/', 1)
        if len(timesig) != 2:
          raise ValueError('Could not parse meter: {}'.format(field_content))

        ts = self._ns.time_signatures.add()
        ts.time = self._current_time
        try:
          ts.numerator = int(timesig[0])
          ts.denominator = int(timesig[1])
        except ValueError as e:
          raise ValueError(e, 'Could not parse meter: {}'.format(field_content))
    elif field_name == 'm':
      pass
    elif field_name == 'N':
      pass
    elif field_name == 'O':
      pass
    elif field_name == 'P':
      # TODO(fjord): implement part parsing.
      pass
    elif field_name == 'Q':
      # Tempo
      # http://abcnotation.com/wiki/abc:standard:v2.1#qtempo

      tempo_match = ABCTune.TEMPO_PATTERN.match(field_content)
      deprecated_tempo_match = ABCTune.TEMPO_DEPRECATED_PATTERN.match(
          field_content)
      tempo_string_only_match = ABCTune.TEMPO_PATTERN_STRING_ONLY.match(
          field_content)
      if tempo_match:
        tempo_rate = int(tempo_match.group(2))
        tempo_unit = Fraction(0)
        for beat in tempo_match.group(1).split():
          tempo_unit += Fraction(beat)
      elif deprecated_tempo_match:
        # http://abcnotation.com/wiki/abc:standard:v2.1#outdated_information_field_syntax
        # In the deprecated syntax, the tempo is interpreted based on the unit
        # note length, which is potentially dependent on the current meter.
        # Set tempo_unit to None for now, and the current unit note length will
        # be filled in later.
        tempo_unit = None
        tempo_rate = int(deprecated_tempo_match.group(1))
      elif tempo_string_only_match:
        tf.logging.warning(
            'Ignoring string-only tempo marking: {}'.format(field_content))
        return
      else:
        raise ValueError('Could not parse tempo: {}'.format(field_content))

      if self._in_header:
        # If we're in the header, save these until we've finished parsing the
        # header. The deprecated syntax relies on the unit note length and
        # meter, which may not be set yet. At the end of the header, we'll fill
        # in the necessary information and add these.
        self._header_tempo_unit = tempo_unit
        self._header_tempo_rate = tempo_rate
      else:
        self._add_tempo(tempo_unit, tempo_rate)
    elif field_name == 'R':
      pass
    elif field_name == 'r':
      pass
    elif field_name == 'S':
      pass
    elif field_name == 's':
      pass
    elif field_name == 'T':
      # Title
      # http://abcnotation.com/wiki/abc:standard:v2.1#ttune_title

      if not self._in_header:
        # TODO(fjord): Non-header titles are used to name parts of tunes, but
        # NoteSequence doesn't currently have any place to put that information.
        tf.logging.warning(
            'Ignoring non-header title: {}'.format(field_content))
        return

      # If there are multiple titles, separate them with semicolons.
      if self._ns.sequence_metadata.title:
        self._ns.sequence_metadata.title += '; ' + field_content
      else:
        self._ns.sequence_metadata.title = field_content
    elif field_name == 'U':
      pass
    elif field_name == 'V':
      pass
    elif field_name == 'W':
      pass
    elif field_name == 'w':
      pass
    elif field_name == 'X':
      # Reference number
      # http://abcnotation.com/wiki/abc:standard:v2.1#xreference_number
      self._ns.reference_number = int(field_content)
    elif field_name == 'Z':
      pass
    else:
      tf.logging.warning(
          'Unknown field name {} with content {}'.format(
              field_name, field_content))
