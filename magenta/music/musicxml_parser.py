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

import xml.etree.ElementTree as ET
from zipfile import ZipFile
from magenta.music import constants


# Constants

# Global variables
# Default to one division per measure
#CURRENT_DIVISIONS = 1

# Default to a tempo of 120 beats per minute
# (This is the default tempo according to the
# Standard MIDI Files 1.0 Specification)
CURRENT_TEMPO = 120

# Duration of a single beat in seconds
CURRENT_SECONDS_PER_BEAT = 0.5

# Running total of time for the current event.
# Resets to 0 on every part. Affected by <forward> and <backup> elements
#CURRENT_TIME_POSITION = 0

# Default to a MIDI velocity of 64 (mf)
CURRENT_VELOCITY = 64

# Default MIDI program (1 = grand piano)
CURRENT_MIDI_PROGRAM = 1

# Current MIDI channel (usually equal to the part number)
CURRENT_MIDI_CHANNEL = 1

# Keep track of previous note to get chord timing correct
PREVIOUS_NOTE = None

# Keep track of current transposition level
CURRENT_TRANSPOSE = 0


class MusicXMLParserState(object):
  """Maintains internal state of the MusicXML parser"""
  def __init__(self):
    # Default to one division per measure
    self.divisions = 1

    # Default to a tempo of 120 beats per minute
    # (This is the default tempo according to the
    # Standard MIDI Files 1.0 Specification)
    self.tempo = 120

    self.seconds_per_beat = 0.5
    self.time_position = 0
    self.velocity = 64
    self.midi_program = 1
    self.midi_channel = 1
    self.previous_note = None
    self.transpose = 0


class MusicXMLDocument(object):
  """Internal representation of a MusicXML Document

  Represents the top level object which holds the MusicXML document
  Responsible for loading the .xml or .mxl file using the getscore method
  If the file is .mxl, this class uncompresses it

  After the file is loaded, this class then parses the document into memory
  using the parse method.
  """
  def __init__(self, filename):
    self.filename = filename
    self.score = self.getscore(filename)
    self.parts = []
    self.score_parts = []
    self.midi_resolution = constants.STANDARD_PPQ
    self.state = MusicXMLParserState()
    # Total time in seconds
    self.total_time = 0
    self.parse()

  @staticmethod
  def getscore(filename):
    """Given a MusicXML file, return the score as an xml.etree.ElementTree

    Given a MusicXML file, return the score as an xml.etree.ElementTree
    If the file is compress (ends in .mxl), uncompress it first

    Args:
        filename: The path of a MusicXML file
    """
    if filename.endswith('.mxl'):
      # Compressed MXL file. Uncompress in memory.
      filename = ZipFile(filename)

      # Ignore files in the META-INF directory.
      # Only retrieve the first file
      namelist = filename.namelist()
      files = [name for name in namelist if not name.startswith('META-INF/')]
      compressed_file_name = files[0]
      score = ET.fromstring(filename.read(compressed_file_name))
    else:
      # Uncompressed XML file.
      tree = ET.parse(filename)
      score = tree.getroot()

    return score

  def parse(self):
    """Parse the uncompressed MusicXML document"""
    # Parse part-list
    xml_part_list = self.score.find("part-list")
    for element in xml_part_list:
      if element.tag == "score-part":
        score_part = ScorePart(element)
        self.score_parts.append(score_part)

    # Parse parts
    xml_parts = self.score.findall("part")
    score_part_index = 0
    for child in xml_parts:
      # If a score part is missing, add a default score part
      if score_part_index >= len(self.score_parts):
        score_part = ScorePart()
        self.score_parts.append(score_part)

      part = Part(child, self.score_parts[score_part_index], self.state)
      self.parts.append(part)
      score_part_index = score_part_index + 1
      if CURRENT_TIME_POSITION > self.total_time:
        self.total_time = CURRENT_TIME_POSITION

  def gettimesignatures(self):
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
    """
    time_signatures = []
    for part in self.parts:
      for measure in part.measures:
        if measure.time_signature is not None:
          if not measure.time_signature in time_signatures:
            # Prevent duplicate time signatures
            time_signatures.append(measure.time_signature)

    return time_signatures

  def getkeysignatures(self):
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
    """
    key_signatures = []
    for part in self.parts:
      for measure in part.measures:
        if measure.key_signature is not None:
          if not measure.key_signature in key_signatures:
            # Prevent duplicate key signatures
            key_signatures.append(measure.key_signature)

    if len(key_signatures) == 0:
      # If there are no key signatures, add C major at the beginning
      key_signature = KeySignature()
      key_signature.time_position = 0
      key_signatures.append(key_signature)

    return key_signatures

  def gettempos(self):
    """Return a list of all tempos in this score.
    If no tempos are found, create a default tempo of 120 qpm.
    """
    tempos = []
    #for part in self.parts:
    part = self.parts[0] # Use only first part
    for measure in part.measures:
      for tempo in measure.tempos:
        tempos.append(tempo)

    # If no tempos, add a default of 120 at beginning
    if len(tempos) == 0:
      tempo = Tempo()
      tempo.qpm = CURRENT_TEMPO
      tempo.time_position = 0
      tempos.append(tempo)

    return tempos


class ScorePart(object):
  """"Internal representation of a MusicXML <score-part>.
  A <score-part> element contains MIDI program and channel info
  for the <part> elements in the MusicXML document.

  If no MIDI info is found for the part, use the next available
  MIDI channel and default to the Grand Piano program (MIDI Program #1)
  """
  def __init__(self, xml_score_part=None):
    self.xml_score_part = xml_score_part
    self.part_name = ""
    self.midi_channel = 1
    self.midi_program = 1
    if xml_score_part != None:
      self.parse()

  def parse(self):
    """Parse the <score-part> element to an in-memory representation"""
    global CURRENT_MIDI_CHANNEL

    if self.xml_score_part.find("part-name") != None:
      self.part_name = self.xml_score_part.find("part-name").text

    xml_midi_instrument = self.xml_score_part.find("midi-instrument")
    if xml_midi_instrument != None and \
      xml_midi_instrument.find("midi-channel") != None and \
      xml_midi_instrument.find("midi-program") != None:
      self.midi_channel = int(xml_midi_instrument.find("midi-channel").text)
      self.midi_program = int(xml_midi_instrument.find("midi-program").text)
    else:
      # If no MIDI info, increment MIDI channel and use default program
      self.midi_channel = CURRENT_MIDI_CHANNEL + 1
      CURRENT_MIDI_CHANNEL = self.midi_channel
      self.midi_program = 1

  def __str__(self):
    score_str = "ScorePart: " + self.part_name
    score_str += ", Channel: " + str(self.midi_channel)
    score_str += ", Program: " + str(self.midi_program)
    return score_str


class Part(object):
  """Internal represention of a MusicXML <part> element"""
  def __init__(self, xml_part, score_part, state):
    self.xml_part = xml_part
    self.score_part = score_part
    self.measures = []
    self.transposes = False
    self.state = state
    self.parse()

  def parse(self):
    """Parse the <part> element"""
    #global CURRENT_TIME_POSITION
    global CURRENT_MIDI_PROGRAM
    global CURRENT_MIDI_CHANNEL
    global CURRENT_TRANSPOSE

    # Reset the time position when parsing each part
    #CURRENT_TIME_POSITION = 0
    self.state.time_position = 0
    CURRENT_MIDI_CHANNEL = self.score_part.midi_channel
    CURRENT_MIDI_PROGRAM = self.score_part.midi_program
    CURRENT_TRANSPOSE = 0

    xml_measures = self.xml_part.findall("measure")
    for child in xml_measures:
      measure = Measure(child, self, self.state)
      self.measures.append(measure)

  def getnotesinvoice(self, voice):
    """Get all the notes in this part for the specified voice number"""
    notes = []
    for measure in self.measures:
      for note in measure.notes:
        if note.voice == voice:
          notes.append(note)

    return notes

  def __str__(self):
    part_str = "Part: " + self.score_part.part_name
    return part_str


class Measure(object):
  """Internal represention of the MusicXML <measure> element"""
  def __init__(self, xml_measure, part, state):
    self.xml_measure = xml_measure
    self.notes = []
    self.tempos = []
    self.time_signature = None
    self.key_signature = KeySignature() # Default to C major
    self.current_ticks = 0      # Cumulative tick counter for this measure
    self.transpose = 0          # Number of semitones to transpose notes
    self.part = part
    self.state = state
    self.parse()

  def parse(self):
    """Parse the <measure> element"""
    global PREVIOUS_NOTE

    for child in self.xml_measure:
      if child.tag == "attributes":
        self.parseattributes(child)
      elif child.tag == "backup":
        self.parsebackup(child)
      elif child.tag == "direction":
        self.parsedirection(child)
      elif child.tag == "forward":
        self.parseforward(child)
      elif child.tag == "note":
        note = Note(child, self.state)
        self.notes.append(note)
        # Keep track of current note as previous note for chord timings
        PREVIOUS_NOTE = note

  def parseattributes(self, xml_attributes):
    """Parse the MusicXML <attributes> element"""
    #global CURRENT_DIVISIONS
    global CURRENT_TRANSPOSE

    for child in xml_attributes:
      if child.tag == "divisions":
        #CURRENT_DIVISIONS = int(child.text)
        self.state.divisions = int(child.text)
      elif child.tag == "key":
        self.key_signature = KeySignature(child)
      elif child.tag == "time":
        self.time_signature = TimeSignature(child)
      elif child.tag == "transpose":
        self.transpose = int(child.find("chromatic").text)
        CURRENT_TRANSPOSE = self.transpose
        self.key_signature.key += self.transpose
        self.part.transposes = True

  def parsebackup(self, xml_backup):
    """Parse the MusicXML <backup> element.
    This moves the global time position backwards.
    """
    global CURRENT_TIME_POSITION

    xml_duration = xml_backup.find("duration")
    backup_duration = int(xml_duration.text)
    midi_ticks = backup_duration * (constants.STANDARD_PPQ
                                    / self.state.divisions)
    seconds = (midi_ticks / constants.STANDARD_PPQ) * CURRENT_SECONDS_PER_BEAT
    CURRENT_TIME_POSITION -= seconds

  def parsedirection(self, xml_direction):
    """Parse the MusicXML <direction> element."""
    global CURRENT_TEMPO
    global CURRENT_SECONDS_PER_BEAT
    global CURRENT_VELOCITY

    for child in xml_direction:
      if child.tag == "sound":
        if child.get("tempo") != None:
          tempo = Tempo(child)
          self.tempos.append(tempo)
          CURRENT_TEMPO = tempo.qpm
          CURRENT_SECONDS_PER_BEAT = 60 / CURRENT_TEMPO
          if child.get("dynamics") != None:
            CURRENT_VELOCITY = int(child.get("dynamics"))

  def parseforward(self, xml_forward):
    """Parse the MusicXML <backup> element.
    This moves the global time position forward.
    """
    global CURRENT_TIME_POSITION

    xml_duration = xml_forward.find('duration')
    forward_duration = int(xml_duration.text)
    midi_ticks = forward_duration * (constants.STANDARD_PPQ
                                     / self.state.divisions)
    seconds = (midi_ticks / constants.STANDARD_PPQ) * CURRENT_SECONDS_PER_BEAT
    CURRENT_TIME_POSITION += seconds


class Note(object):
  """Internal representation of a MusicXML <note> element"""
  def __init__(self, xml_note, state):
    self.xml_note = xml_note
    self.duration = 0               # MusicXML duration
    self.midi_ticks = 0             # Duration in MIDI ticks
    self.seconds = 0                # Duration in seconds
    self.time_position = 0          # Onset time in seconds
    self.voice = 1
    self.is_rest = False
    self.is_in_chord = False
    self.is_grace_note = False
    self.pitch = None               # Tuple (Pitch Name, MIDI number)
    self.state = state
    self.parse()

  def parse(self):
    """Parse the MusicXML <note> element"""
    global CURRENT_TIME_POSITION

    self.midi_channel = CURRENT_MIDI_CHANNEL
    self.midi_program = CURRENT_MIDI_PROGRAM
    self.velocity = CURRENT_VELOCITY

    for child in self.xml_note:
      if child.tag == "chord":
        self.is_in_chord = True
      elif child.tag == "duration":
        self.duration = int(child.text)

        self.midi_ticks = self.duration
        self.midi_ticks *= (constants.STANDARD_PPQ / self.state.divisions)

        self.seconds = (self.midi_ticks / constants.STANDARD_PPQ)
        self.seconds *= CURRENT_SECONDS_PER_BEAT

        self.time_position = CURRENT_TIME_POSITION

        if self.is_in_chord:
          # If this is a chord, subtract the duration from the
          # previous note and make this note occur at the same time
          PREVIOUS_NOTE.time_position -= self.seconds
          self.time_position -= self.seconds
        else:
          # Only increment time positions once in chord
          CURRENT_TIME_POSITION += self.seconds

      elif child.tag == "pitch":
        self.parsepitch(child)
      elif child.tag == "rest":
        self.is_rest = True
      elif child.tag == "voice":
        self.voice = int(child.text)

  def parsepitch(self, xml_pitch):
    """Parse the MusicXML <pitch> element"""
    step = xml_pitch.find("step").text
    alter = 0
    if xml_pitch.find("alter") != None:
      alter = xml_pitch.find("alter").text
    octave = xml_pitch.find("octave").text

    # Visual pitch representation
    alter_string = ""
    if alter == "-2":
      alter_string = "bb"
    elif alter == "-1":
      alter_string = "b"
    elif alter == "1":
      alter_string = "#"
    elif alter == "2":
      alter_string = "x"

    # N.B. - pitch_string does not account for transposition
    pitch_string = step + alter_string + octave

    # Compute MIDI pitch number (C4 = 60, C1 = 24, C0 = 12)
    midi_pitch = self.pitchtomidipitch(step, alter, octave)
    # Transpose MIDI pitch
    midi_pitch = midi_pitch + CURRENT_TRANSPOSE
    self.pitch = (pitch_string, midi_pitch)

  @staticmethod
  def pitchtomidipitch(step, alter, octave):
    """Convert MusicXML pitch representation to MIDI pitch number"""
    pitch_class = 0
    if step == "C":
      pitch_class = 0
    elif step == "D":
      pitch_class = 2
    elif step == "E":
      pitch_class = 4
    elif step == "F":
      pitch_class = 5
    elif step == "G":
      pitch_class = 7
    elif step == "A":
      pitch_class = 9
    elif step == "B":
      pitch_class = 11

    pitch_class = (pitch_class + int(alter)) % 12
    midi_pitch = (12 + pitch_class) + (int(octave) * 12)
    return midi_pitch

  def __str__(self):
    note_string = "{duration: " + str(self.duration)
    note_string += ", midi_ticks: " + str(self.midi_ticks)
    note_string += ", seconds: " + str(self.seconds)
    if self.is_rest:
      note_string += ", rest: " + str(self.is_rest)
    else:
      note_string += ", pitch: " + self.pitch[0]
      note_string += ", MIDI pitch: " + str(self.pitch[1])

    note_string += ", voice: " + str(self.voice)
    note_string += ", velocity: " + str(self.velocity) + "} "
    note_string += "(@time: " + str(self.time_position) + ")"
    return note_string


class TimeSignature(object):
  """Internal representation of a MusicXML time signature
  Does not support:
  - Composite time signatures: 3+2/8
  - Alternating time signatures 2/4 + 3/8
  - Senza misura
  """
  def __init__(self, xml_time):
    self.xml_time = xml_time
    self.numerator = -1
    self.denominator = -1
    self.time_position = -1
    self.parse()

  def parse(self):
    """Parse the MusicXML <time> element"""
    self.numerator = int(self.xml_time.find("beats").text)
    self.denominator = int(self.xml_time.find("beat-type").text)
    self.time_position = CURRENT_TIME_POSITION

  def __str__(self):
    time_sig_str = str(self.numerator) + "/" + str(self.denominator)
    time_sig_str += " (@time: " + str(self.time_position) + ")"
    return time_sig_str

  def __eq__(self, other):
    isequal = self.numerator == other.numerator
    isequal = isequal and (self.denominator == other.denominator)
    isequal = isequal and (self.time_position == other.time_position)
    return isequal


class KeySignature(object):
  """Internal representation of a MusicXML key signature"""
  def __init__(self, xml_key=None):
    self.xml_key = xml_key
    # MIDI and MusicXML identify key by using "fifths":
    # -1 = F, 0 = C, 1 = G etc.
    self.key = 0
    # mode is "major" or "minor" only: MIDI only supports major and minor
    self.mode = "major"
    self.time_position = -1
    if xml_key != None:
      self.parse()

  def parse(self):
    """Parse the MusicXML <key> element into a MIDI compatible key
    If the mode is not minor (e.g. dorian), default to "major"
    because MIDI only supports major and minor modes.
    """
    self.key = int(self.xml_key.find("fifths").text)
    mode = self.xml_key.find("mode")
    # Anything not minor will be interpreted as major
    if mode != "minor":
      mode = "major"
    self.mode = mode
    self.time_position = CURRENT_TIME_POSITION

  def __str__(self):
    keys = (["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D",
             "A", "E", "B", "F#", "C#"])
    key_string = keys[self.key + 7] + " " + self.mode
    key_string += " (@time: " + str(self.time_position) + ")"
    return key_string

  def __eq__(self, other):
    isequal = self.key == other.key
    isequal = isequal and (self.mode == other.mode)
    isequal = isequal and (self.time_position == other.time_position)
    return isequal


class Tempo(object):
  """Internal representation of a MusicXML tempo"""
  def __init__(self, xml_sound=None):
    self.xml_sound = xml_sound
    self.qpm = -1
    self.time_position = -1
    if xml_sound != None:
      self.parse()

  def parse(self):
    """Parse the MusicXML <sound> element and retrieve the tempo.
    If no tempo is specified, default to DEFAULT_QUARTERS_PER_MINUTE
    """
    self.qpm = float(self.xml_sound.get("tempo"))
    if self.qpm == 0:
      # If tempo is 0, set it to default
      self.qpm = constants.DEFAULT_QUARTERS_PER_MINUTE
    self.time_position = CURRENT_TIME_POSITION

  def __str__(self):
    tempo_str = "Tempo: " + str(self.qpm)
    tempo_str += " (@time: " + str(self.time_position) + ")"
    return tempo_str
