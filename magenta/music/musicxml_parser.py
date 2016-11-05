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
"""MusicXML parser.

Simple MusicXML parser used to convert MusicXML
into tensorflow.magenta.NoteSequence.
"""

# Imports
# Python 2 uses integer division for integers. Using this gives the Python 3
# behavior of producing a float when dividing integers
from __future__ import division

from fractions import Fraction
import xml.etree.ElementTree as ET
from zipfile import ZipFile

# internal imports

from magenta.music import constants

DEFAULT_MIDI_PROGRAM = 0    # Default MIDI Program (0 = grand piano)
DEFAULT_MIDI_CHANNEL = 0    # Default MIDI Channel (0 = first channel)
MUSICXML_MIME_TYPE = 'application/vnd.recordare.musicxml+xml'


class MusicXMLParseException(Exception):
  """Exception thrown when the MusicXML contents cannot be parsed."""
  pass


class PitchStepParseException(MusicXMLParseException):
  """Exception thrown when a pitch step cannot be parsed.

  Will happen if pitch step is not one of A, B, C, D, E, F, or G
  """
  pass


class MusicXMLParserState(object):
  """Maintains internal state of the MusicXML parser."""

  def __init__(self):
    # Default to one division per measure
    # From the MusicXML documentation: "The divisions element indicates
    # how many divisions per quarter note are used to indicate a note's
    # duration. For example, if duration = 1 and divisions = 2,
    # this is an eighth note duration."
    self.divisions = 1

    # Default to a tempo of 120 quarter notes per minute
    # MusicXML calls this tempo, but Magenta calls this qpm
    # Therefore, the variable is called qpm, but reads the
    # MusicXML tempo attribute
    # (120 qpm is the default tempo according to the
    # Standard MIDI Files 1.0 Specification)
    self.qpm = 120

    # Duration of a single quarter note in seconds
    self.seconds_per_quarter = 0.5

    # Running total of time for the current event in seconds.
    # Resets to 0 on every part. Affected by <forward> and <backup> elements
    self.time_position = 0

    # Default to a MIDI velocity of 64 (mf)
    self.velocity = 64

    # Default MIDI program (0 = grand piano)
    self.midi_program = DEFAULT_MIDI_PROGRAM

    # Current MIDI channel (usually equal to the part number)
    self.midi_channel = DEFAULT_MIDI_CHANNEL

    # Keep track of previous note to get chord timing correct
    # This variable stores an instance of the Note class (defined below)
    self.previous_note = None

    # Keep track of current transposition level in +/- semitones.
    self.transpose = 0


class MusicXMLDocument(object):
  """Internal representation of a MusicXML Document.

  Represents the top level object which holds the MusicXML document
  Responsible for loading the .xml or .mxl file using the _get_score method
  If the file is .mxl, this class uncompresses it

  After the file is loaded, this class then parses the document into memory
  using the parse method.
  """

  def __init__(self, filename):
    self._score = self._get_score(filename)
    self.parts = []
    # ScoreParts indexed by id.
    self._score_parts = {}
    self.midi_resolution = constants.STANDARD_PPQ
    self._state = MusicXMLParserState()
    # Total time in seconds
    self.total_time_secs = 0
    self._parse()

  @staticmethod
  def _get_score(filename):
    """Given a MusicXML file, return the score as an xml.etree.ElementTree.

    Given a MusicXML file, return the score as an xml.etree.ElementTree
    If the file is compress (ends in .mxl), uncompress it first

    Args:
        filename: The path of a MusicXML file

    Returns:
      The score as an xml.etree.ElementTree.
    """
    score = None
    if filename.endswith('.mxl'):
      # Compressed MXL file. Uncompress in memory.
      filename = ZipFile(filename)

      # A compressed MXL file may contain multiple files, but only one
      # MusicXML file. Read the META-INF/container.xml file inside of the
      # MXL file to locate the MusicXML file within the MXL file
      # http://www.musicxml.com/tutorial/compressed-mxl-files/zip-archive-structure/

      # Raise a MusicXMLParseException if multiple MusicXML files found
      namelist = filename.namelist()
      container_file = [x for x in namelist if x == 'META-INF/container.xml']
      compressed_file_name = ''

      try:
        container = ET.fromstring(filename.read(container_file[0]))
        for rootfile_tag in container.findall('./rootfiles/rootfile'):
          if 'media-type' in rootfile_tag.attrib:
            if rootfile_tag.attrib['media-type'] == MUSICXML_MIME_TYPE:
              if not compressed_file_name:
                compressed_file_name = rootfile_tag.attrib['full-path']
              else:
                raise MusicXMLParseException(
                    'Multiple MusicXML files found in compressed archive')
          else:
            # No media-type attribute, so assume this is the MusicXML file
            if not compressed_file_name:
              compressed_file_name = rootfile_tag.attrib['full-path']
            else:
              raise MusicXMLParseException(
                  'Multiple MusicXML files found in compressed archive')
      except ET.ParseError as e:
        raise MusicXMLParseException(e)

      try:
        score = ET.fromstring(filename.read(compressed_file_name))
      except ET.ParseError as e:
        raise MusicXMLParseException(e)
    else:
      # Uncompressed XML file.
      try:
        tree = ET.parse(filename)
        score = tree.getroot()
      except ET.ParseError as e:
        raise MusicXMLParseException(e)

    return score

  def _parse(self):
    """Parse the uncompressed MusicXML document."""
    # Parse part-list
    xml_part_list = self._score.find('part-list')
    if xml_part_list is not None:
      for element in xml_part_list:
        if element.tag == 'score-part':
          score_part = ScorePart(element)
          self._score_parts[score_part.id] = score_part

    # Parse parts
    for score_part_index, child in enumerate(self._score.findall('part')):
      part = Part(child, self._score_parts, self._state)
      self.parts.append(part)
      score_part_index += 1
      if self._state.time_position > self.total_time_secs:
        self.total_time_secs = self._state.time_position

  def get_time_signatures(self):
    """Return a list of all the time signatures used in this score.

    Does not support polymeter (i.e. assumes all parts have the same
    time signature, such as Part 1 having a time signature of 6/8
    while Part 2 has a simultaneous time signature of 2/4).

    Ignores duplicate time signatures to prevent Magenta duplicate
    time signature error. This happens when multiple parts have the
    same time signature is used in multiple parts at the same time.

    Example: If Part 1 has a time siganture of 4/4 and Part 2 also
    has a time signature of 4/4, then only instance of 4/4 is sent
    to Magenta.

    Returns:
      A list of all TimeSignature objects used in this score.
    """
    time_signatures = []
    for part in self.parts:
      for measure in part.measures:
        if measure.time_signature is not None:
          if measure.time_signature not in time_signatures:
            # Prevent duplicate time signatures
            time_signatures.append(measure.time_signature)

    return time_signatures

  def get_key_signatures(self):
    """Return a list of all the key signatures used in this score.

    Support different key signatures in different parts (score in
    written pitch).

    Ignores duplicate key signatures to prevent Magenta duplicate key
    signature error. This happens when multiple parts have the same
    key signature at the same time.

    Example: If the score is in written pitch and the
    flute is written in the key of Bb major, the trombone will also be
    written in the key of Bb major. However, the clarinet and trumpet
    will be written in the key of C major because they are Bb transposing
    instruments.

    If no key signatures are found, create a default key signature of
    C major.

    Returns:
      A list of all KeySignature objects used in this score.
    """
    key_signatures = []
    for part in self.parts:
      for measure in part.measures:
        if measure.key_signature is not None:
          if measure.key_signature not in key_signatures:
            # Prevent duplicate key signatures
            key_signatures.append(measure.key_signature)

    if not key_signatures:
      # If there are no key signatures, add C major at the beginning
      key_signature = KeySignature(self._state)
      key_signature.time_position = 0
      key_signatures.append(key_signature)

    return key_signatures

  def get_tempos(self):
    """Return a list of all tempos in this score.

    If no tempos are found, create a default tempo of 120 qpm.

    Returns:
      A list of all Tempo objects used in this score.
    """
    tempos = []

    if self.parts:
      part = self.parts[0]  # Use only first part
      for measure in part.measures:
        for tempo in measure.tempos:
          tempos.append(tempo)

    # If no tempos, add a default of 120 at beginning
    if not tempos:
      tempo = Tempo(self._state)
      tempo.qpm = self._state.qpm
      tempo.time_position = 0
      tempos.append(tempo)

    return tempos


class ScorePart(object):
  """"Internal representation of a MusicXML <score-part>.

  A <score-part> element contains MIDI program and channel info
  for the <part> elements in the MusicXML document.

  If no MIDI info is found for the part, use the default MIDI channel (0)
  and default to the Grand Piano program (MIDI Program #1).
  """

  def __init__(self, xml_score_part=None):
    self.id = ''
    self.part_name = ''
    self.midi_channel = DEFAULT_MIDI_CHANNEL
    self.midi_program = DEFAULT_MIDI_PROGRAM
    if xml_score_part is not None:
      self._parse(xml_score_part)

  def _parse(self, xml_score_part):
    """Parse the <score-part> element to an in-memory representation."""
    self.id = xml_score_part.attrib['id']

    if xml_score_part.find('part-name') is not None:
      self.part_name = xml_score_part.find('part-name').text or ''

    xml_midi_instrument = xml_score_part.find('midi-instrument')
    if (xml_midi_instrument is not None and
        xml_midi_instrument.find('midi-channel') is not None and
        xml_midi_instrument.find('midi-program') is not None):
      self.midi_channel = int(xml_midi_instrument.find('midi-channel').text)
      self.midi_program = int(xml_midi_instrument.find('midi-program').text)
    else:
      # If no MIDI info, use the default MIDI channel.
      self.midi_channel = DEFAULT_MIDI_CHANNEL
      # Use the default MIDI program
      self.midi_program = DEFAULT_MIDI_PROGRAM

  def __str__(self):
    score_str = 'ScorePart: ' + self.part_name
    score_str += ', Channel: ' + str(self.midi_channel)
    score_str += ', Program: ' + str(self.midi_program)
    return score_str


class Part(object):
  """Internal represention of a MusicXML <part> element."""

  def __init__(self, xml_part, score_parts, state):
    self.id = ''
    self.score_part = None
    self.measures = []
    self._state = state
    self._parse(xml_part, score_parts)

  def _parse(self, xml_part, score_parts):
    """Parse the <part> element."""
    self.id = xml_part.attrib['id']
    if self.id in score_parts:
      self.score_part = score_parts[self.id]
    else:
      # If this part references a score-part id that was not found in the file,
      # construct a default score-part.
      self.score_part = ScorePart()

    # Reset the time position when parsing each part
    self._state.time_position = 0
    self._state.midi_channel = self.score_part.midi_channel
    self._state.midi_program = self.score_part.midi_program
    self._state.transpose = 0

    xml_measures = xml_part.findall('measure')
    for child in xml_measures:
      measure = Measure(child, self._state)
      self.measures.append(measure)

  def __str__(self):
    part_str = 'Part: ' + self.score_part.part_name
    return part_str


class Measure(object):
  """Internal represention of the MusicXML <measure> element."""

  def __init__(self, xml_measure, state):
    self.xml_measure = xml_measure
    self.notes = []
    self.tempos = []
    self.time_signature = None
    self.key_signature = None
    self.current_ticks = 0  # Cumulative tick counter for this measure
    self.state = state
    self._parse()

  def _parse(self):
    """Parse the <measure> element."""

    for child in self.xml_measure:
      if child.tag == 'attributes':
        self._parse_attributes(child)
      elif child.tag == 'backup':
        self._parse_backup(child)
      elif child.tag == 'direction':
        self._parse_direction(child)
      elif child.tag == 'forward':
        self._parse_forward(child)
      elif child.tag == 'note':
        note = Note(child, self.state)
        self.notes.append(note)
        # Keep track of current note as previous note for chord timings
        self.state.previous_note = note
      else:
        # Ignore other tag types because they are not relevant to Magenta.
        pass

  def _parse_attributes(self, xml_attributes):
    """Parse the MusicXML <attributes> element."""

    for child in xml_attributes:
      if child.tag == 'divisions':
        self.state.divisions = int(child.text)
      elif child.tag == 'key':
        self.key_signature = KeySignature(self.state, child)
      elif child.tag == 'time':
        self.time_signature = TimeSignature(self.state, child)
      elif child.tag == 'transpose':
        transpose = int(child.find('chromatic').text)
        self.state.transpose = transpose
        if self.key_signature is not None:
          self.key_signature.key += transpose
      else:
        # Ignore other tag types because they are not relevant to Magenta.
        pass

  def _parse_backup(self, xml_backup):
    """Parse the MusicXML <backup> element.

    This moves the global time position backwards.

    Args:
      xml_backup: XML element with tag type 'backup'.
    """

    xml_duration = xml_backup.find('duration')
    backup_duration = int(xml_duration.text)
    midi_ticks = backup_duration * (constants.STANDARD_PPQ
                                    / self.state.divisions)
    seconds = ((midi_ticks / constants.STANDARD_PPQ)
               * self.state.seconds_per_quarter)
    self.state.time_position -= seconds

  def _parse_direction(self, xml_direction):
    """Parse the MusicXML <direction> element."""

    for child in xml_direction:
      if child.tag == 'sound':
        if child.get('tempo') is not None:
          tempo = Tempo(self.state, child)
          self.tempos.append(tempo)
          self.state.qpm = tempo.qpm
          self.state.seconds_per_quarter = 60 / self.state.qpm
          if child.get('dynamics') is not None:
            self.state.velocity = int(child.get('dynamics'))

  def _parse_forward(self, xml_forward):
    """Parse the MusicXML <forward> element.

    This moves the global time position forward.

    Args:
      xml_forward: XML element with tag type 'forward'.
    """

    xml_duration = xml_forward.find('duration')
    forward_duration = int(xml_duration.text)
    midi_ticks = forward_duration * (constants.STANDARD_PPQ
                                     / self.state.divisions)
    seconds = ((midi_ticks / constants.STANDARD_PPQ)
               * self.state.seconds_per_quarter)
    self.state.time_position += seconds


class Note(object):
  """Internal representation of a MusicXML <note> element."""

  def __init__(self, xml_note, state):
    self.xml_note = xml_note
    self.voice = 1
    self.is_rest = False
    self.is_in_chord = False
    self.is_grace_note = False
    self.pitch = None               # Tuple (Pitch Name, MIDI number)
    self.note_duration = NoteDuration(state)
    self.state = state
    self._parse()

  def _parse(self):
    """Parse the MusicXML <note> element."""

    self.midi_channel = self.state.midi_channel
    self.midi_program = self.state.midi_program
    self.velocity = self.state.velocity

    for child in self.xml_note:
      if child.tag == 'chord':
        self.is_in_chord = True
      elif child.tag == 'duration':
        self.note_duration.parse_duration(self.is_in_chord, self.is_grace_note,
                                          child.text)
      elif child.tag == 'pitch':
        self._parse_pitch(child)
      elif child.tag == 'rest':
        self.is_rest = True
      elif child.tag == 'voice':
        self.voice = int(child.text)
      elif child.tag == 'dot':
        self.note_duration.dots += 1
      elif child.tag == 'type':
        self.note_duration.type = child.text
      elif child.tag == 'time-modification':
        # A time-modification element represents a tuplet_ratio
        self._parse_tuplet(child)
      else:
        # Ignore other tag types because they are not relevant to Magenta.
        pass

  def _parse_pitch(self, xml_pitch):
    """Parse the MusicXML <pitch> element."""
    step = xml_pitch.find('step').text
    alter_text = ''
    alter = 0.0
    if xml_pitch.find('alter') is not None:
      alter_text = xml_pitch.find('alter').text
    octave = xml_pitch.find('octave').text

    # Parse alter string to a float (floats represent microtonal alterations)
    if alter_text:
      alter = float(alter_text)

    # Check if this is a semitone alter (i.e. an integer) or microtonal (float)
    alter_semitones = int(alter)  # Number of semitones
    is_microtonal_alter = (alter != alter_semitones)

    # Visual pitch representation
    alter_string = ''
    if alter_semitones == -2:
      alter_string = 'bb'
    elif alter_semitones == -1:
      alter_string = 'b'
    elif alter_semitones == 1:
      alter_string = '#'
    elif alter_semitones == 2:
      alter_string = 'x'

    if is_microtonal_alter:
      alter_string += ' (+microtones) '

    # N.B. - pitch_string does not account for transposition
    pitch_string = step + alter_string + octave

    # Compute MIDI pitch number (C4 = 60, C1 = 24, C0 = 12)
    midi_pitch = self.pitch_to_midi_pitch(step, alter, octave)
    # Transpose MIDI pitch
    midi_pitch += self.state.transpose
    self.pitch = (pitch_string, midi_pitch)

  def _parse_tuplet(self, xml_time_modification):
    """Parses a tuplet ratio.

    Represented in MusicXML by the <time-modification> element.

    Args:
      xml_time_modification: An xml time-modification element.
    """
    numerator = int(xml_time_modification.find('actual-notes').text)
    denominator = int(xml_time_modification.find('normal-notes').text)
    self.note_duration.tuplet_ratio = Fraction(numerator, denominator)

  @staticmethod
  def pitch_to_midi_pitch(step, alter, octave):
    """Convert MusicXML pitch representation to MIDI pitch number."""
    pitch_class = 0
    if step == 'C':
      pitch_class = 0
    elif step == 'D':
      pitch_class = 2
    elif step == 'E':
      pitch_class = 4
    elif step == 'F':
      pitch_class = 5
    elif step == 'G':
      pitch_class = 7
    elif step == 'A':
      pitch_class = 9
    elif step == 'B':
      pitch_class = 11
    else:
      # Raise exception for unknown step (ex: 'Q')
      raise PitchStepParseException('Unable to parse pitch step ' + step)

    pitch_class = (pitch_class + int(alter)) % 12
    midi_pitch = (12 + pitch_class) + (int(octave) * 12)
    return midi_pitch

  def __str__(self):
    note_string = '{duration: ' + str(self.note_duration.duration)
    note_string += ', midi_ticks: ' + str(self.note_duration.midi_ticks)
    note_string += ', seconds: ' + str(self.note_duration.seconds)
    if self.is_rest:
      note_string += ', rest: ' + str(self.is_rest)
    else:
      note_string += ', pitch: ' + self.pitch[0]
      note_string += ', MIDI pitch: ' + str(self.pitch[1])

    note_string += ', voice: ' + str(self.voice)
    note_string += ', velocity: ' + str(self.velocity) + '} '
    note_string += '(@time: ' + str(self.note_duration.time_position) + ')'
    return note_string


class NoteDuration(object):
  """Internal representation of a MusicXML note's duration properties."""

  TYPE_RATIO_MAP = {'maxima': Fraction(8, 1), 'long': Fraction(4, 1),
                    'breve': Fraction(2, 1), 'whole': Fraction(1, 1),
                    'half': Fraction(1, 2), 'quarter': Fraction(1, 4),
                    'eighth': Fraction(1, 8), '16th': Fraction(1, 16),
                    '32nd': Fraction(1, 32), '64th': Fraction(1, 64),
                    '128th': Fraction(1, 128), '256th': Fraction(1, 256),
                    '512th': Fraction(1, 512), '1024th': Fraction(1, 1024)}

  def __init__(self, state):
    self.duration = 0                   # MusicXML duration
    self.midi_ticks = 0                 # Duration in MIDI ticks
    self.seconds = 0                    # Duration in seconds
    self.time_position = 0              # Onset time in seconds
    self.dots = 0                       # Number of augmentation dots
    self.type = 'quarter'               # MusicXML duration type
    self.tuplet_ratio = Fraction(1, 1)  # Ratio for tuplets (default to 1)
    self.is_grace_note = True           # Assume true until not found
    self.state = state

  def parse_duration(self, is_in_chord, is_grace_note, duration):
    """Parse the duration of a note and compute timings."""
    self.duration = int(duration)

    # Due to an error in Sibelius' export, force this note to have the
    # duration of the previous note if it is in a chord
    if is_in_chord:
      self.duration = self.state.previous_note.note_duration.duration

    self.midi_ticks = self.duration
    self.midi_ticks *= (constants.STANDARD_PPQ / self.state.divisions)

    self.seconds = (self.midi_ticks / constants.STANDARD_PPQ)
    self.seconds *= self.state.seconds_per_quarter

    self.time_position = self.state.time_position

    # Not sure how to handle durations of grace notes yet as they
    # steal time from subsequent notes and they do not have a
    # <duration> tag in the MusicXML
    self.is_grace_note = is_grace_note

    if is_in_chord:
      # If this is a chord, set the time position to the time position
      # of the previous note (i.e. all the notes in the chord will have
      # the same time position)
      self.time_position = self.state.previous_note.note_duration.time_position
    else:
      # Only increment time positions once in chord
      self.state.time_position += self.seconds

  def _convert_type_to_ratio(self):
    """Convert the MusicXML note-type-value to a Python Fraction.

    Examples:
    - whole = 1/1
    - half = 1/2
    - quarter = 1/4
    - 32nd = 1/32

    Returns:
      A Fraction object representing the note type.
    """
    return self.TYPE_RATIO_MAP[self.type]

  def duration_ratio(self):
    """Compute the duration ratio of the note as a Python Fraction.

    Examples:
    - Whole Note = 1
    - Quarter Note = 1/4
    - Dotted Quarter Note = 3/8
    - Triplet eighth note = 1/12

    Returns:
      The duration ratio as a Python Fraction.
    """
    # Get ratio from MusicXML note type
    duration_ratio = Fraction(1, 1)
    type_ratio = self._convert_type_to_ratio()

    # Compute tuplet ratio
    duration_ratio /= self.tuplet_ratio
    type_ratio /= self.tuplet_ratio

    # Add augmentation dots
    one_half = Fraction(1, 2)
    dot_sum = Fraction(0, 1)
    for dot in range(self.dots):
      dot_sum += (one_half ** (dot + 1)) * type_ratio

    duration_ratio = type_ratio + dot_sum

    # If the note is a grace note, force its ratio to be 0
    # because it does not have a <duration> tag
    if self.is_grace_note:
      duration_ratio = Fraction(0, 1)

    return duration_ratio

  def duration_float(self):
    """Return the duration ratio as a float."""
    ratio = self.duration_ratio()
    return ratio.numerator / ratio.denominator


class TimeSignature(object):
  """Internal representation of a MusicXML time signature.

  Does not support:
  - Composite time signatures: 3+2/8
  - Alternating time signatures 2/4 + 3/8
  - Senza misura
  """

  def __init__(self, state, xml_time):
    self.xml_time = xml_time
    self.numerator = -1
    self.denominator = -1
    self.time_position = -1
    self.state = state
    self._parse()

  def _parse(self):
    """Parse the MusicXML <time> element."""
    self.numerator = int(self.xml_time.find('beats').text)
    self.denominator = int(self.xml_time.find('beat-type').text)
    self.time_position = self.state.time_position

  def __str__(self):
    time_sig_str = str(self.numerator) + '/' + str(self.denominator)
    time_sig_str += ' (@time: ' + str(self.time_position) + ')'
    return time_sig_str

  def __eq__(self, other):
    isequal = self.numerator == other.numerator
    isequal = isequal and (self.denominator == other.denominator)
    isequal = isequal and (self.time_position == other.time_position)
    return isequal


class KeySignature(object):
  """Internal representation of a MusicXML key signature."""

  def __init__(self, state, xml_key=None):
    self.xml_key = xml_key
    # MIDI and MusicXML identify key by using "fifths":
    # -1 = F, 0 = C, 1 = G etc.
    self.key = 0
    # mode is "major" or "minor" only: MIDI only supports major and minor
    self.mode = 'major'
    self.time_position = -1
    self.state = state
    if xml_key is not None:
      self._parse()

  def _parse(self):
    """Parse the MusicXML <key> element into a MIDI compatible key.

    If the mode is not minor (e.g. dorian), default to "major"
    because MIDI only supports major and minor modes.
    """
    self.key = int(self.xml_key.find('fifths').text)
    mode = self.xml_key.find('mode')
    # Anything not minor will be interpreted as major
    if mode != 'minor':
      mode = 'major'
    self.mode = mode
    self.time_position = self.state.time_position

  def __str__(self):
    keys = (['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D',
             'A', 'E', 'B', 'F#', 'C#'])
    key_string = keys[self.key + 7] + ' ' + self.mode
    key_string += ' (@time: ' + str(self.time_position) + ')'
    return key_string

  def __eq__(self, other):
    isequal = self.key == other.key
    isequal = isequal and (self.mode == other.mode)
    isequal = isequal and (self.time_position == other.time_position)
    return isequal


class Tempo(object):
  """Internal representation of a MusicXML tempo."""

  def __init__(self, state, xml_sound=None):
    self.xml_sound = xml_sound
    self.qpm = -1
    self.time_position = -1
    self.state = state
    if xml_sound is not None:
      self._parse()

  def _parse(self):
    """Parse the MusicXML <sound> element and retrieve the tempo.

    If no tempo is specified, default to DEFAULT_QUARTERS_PER_MINUTE
    """
    self.qpm = float(self.xml_sound.get('tempo'))
    if self.qpm == 0:
      # If tempo is 0, set it to default
      self.qpm = constants.DEFAULT_QUARTERS_PER_MINUTE
    self.time_position = self.state.time_position

  def __str__(self):
    tempo_str = 'Tempo: ' + str(self.qpm)
    tempo_str += ' (@time: ' + str(self.time_position) + ')'
    return tempo_str
