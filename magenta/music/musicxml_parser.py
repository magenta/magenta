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

from zipfile import ZipFile
import xml.etree.ElementTree as ET

# Constants

# Number of MIDI ticks per beat. Required by NoteSequence structure
TICKS_PER_BEAT = 960

# Global variables
# Default to one division per measure
current_divisions = 1

# Default to a tempo of 120 beats per minute
# (This is the default tempo according to the
# Standard MIDI Files 1.0 Specification)
current_tempo = 120

# Duration of a single beat in seconds
current_seconds_per_beat = 0.5

# Running total of time for the current event.
# Resets to 0 on every part. Affected by <forward> and <backup> elements
current_time_position = 0

# Default to a MIDI velocity of 64 (mf)
current_velocity = 64

# Default MIDI program (1 = grand piano)
current_midi_program = 1

# Current MIDI channel (usually equal to the part number)
current_midi_channel = 1

# Keep track of previous note to get chord timing correct
previous_note = None

# Keep track of current transposition level
current_transpose = 0


class MusicXMLDocument:
  def __init__(self, filename):
    self.filename = filename
    self.score = self.getScore(filename)
    self.parts = []
    self.score_parts = []
    self.midi_resolution = TICKS_PER_BEAT
    # Total time in seconds
    self.total_time = 0
    self.parse()

  def getScore(self, filename):
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
    # Parse part list
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

      part = Part(child, self.score_parts[score_part_index])
      self.parts.append(part)
      score_part_index = score_part_index + 1
      if current_time_position > self.total_time:
        self.total_time = current_time_position

  def getTimeSignatures(self):
    time_signatures = []
    for part in self.parts:
      for measure in part.measures:
        if measure.time_signature is not None:
          if not measure.time_signature in time_signatures:
            # Prevent duplicate time signatures
            time_signatures.append(measure.time_signature)

    return time_signatures

  def getKeySignatures(self):
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

  def getTempos(self):
    tempos = []
    for part in self.parts:
      for measure in part.measures:
        for tempo in measure.tempos:
          tempos.append(tempo)

    # If no tempos, add a default of 120 at beginning
    if len(tempos) == 0:
        tempo = Tempo()
        tempo.qpm = current_tempo
        tempo.time_position = 0
        tempos.append(tempo)

    return tempos


class ScorePart:
  def __init__(self, xml_score_part = None):
    self.xml_score_part = xml_score_part
    self.part_name = ""
    self.midi_channel = 1
    self.midi_program = 1
    if xml_score_part != None:
      self.parse()

  def parse(self):
    global current_midi_channel

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
      self.midi_channel = current_midi_channel + 1
      current_midi_channel = self.midi_channel
      self.midi_program = 1

  def __str__(self):
    score_str = "ScorePart: " + self.part_name
    score_str += ", Channel: " + str(self.midi_channel)
    score_str += ", Program: " + str(self.midi_program)
    return score_str


class Part:
  def __init__(self, xml_part, score_part):
    self.xml_part = xml_part
    self.score_part = score_part
    self.measures = []
    self.transposes = False
    self.parse()

  def parse(self):
    global current_time_position
    global current_midi_program
    global current_midi_channel
    global current_transpose

    # Reset the time position when parsing each part
    current_time_position = 0
    current_midi_channel = self.score_part.midi_channel
    current_midi_program = self.score_part.midi_program
    current_transpose = 0

    xml_measures = self.xml_part.findall("measure")
    for child in xml_measures:
      measure = Measure(child, self)
      self.measures.append(measure)

  def getNotesInVoice(self, voice):
    notes = []
    for measure in self.measures:
      for note in measure.notes:
        if note.voice == voice:
          notes.append(note)

    return notes

  def __str__(self):
    part_str = "Part: " + self.score_part.part_name
    return part_str


class Measure:
  def __init__(self, xml_measure, part):
    self.xml_measure = xml_measure
    self.notes = []
    self.tempos = []
    self.time_signature = None
    self.key_signature = KeySignature() # Default to C major
    self.current_ticks = 0      # Cumulative tick counter for this measure
    self.transpose = 0          # Number of semitones to transpose notes
    self.part = part
    self.parse()

  def parse(self):
    global previous_note

    measure_number = self.xml_measure.attrib['number']
    for child in self.xml_measure:
      if child.tag == "attributes":
        self.parseAttributes(child)
      elif child.tag == "backup":
        self.parseBackup(child)
      elif child.tag == "direction":
        self.parseDirection(child)
      elif child.tag == "forward":
        self.parseForward(child)
      elif child.tag == "note":
        note = Note(child)
        self.notes.append(note)
        # Keep track of current note as previous note for chord timings
        previous_note = note

  def parseAttributes(self, xml_attributes):
    global current_divisions
    global current_transpose

    for child in xml_attributes:
      if child.tag == "divisions":
        current_divisions = int(child.text)
      elif child.tag == "key":
        self.key_signature = KeySignature(child)
      elif child.tag == "time":
        self.time_signature = TimeSignature(child)
      elif child.tag == "transpose":
        self.transpose = int(child.find("chromatic").text)
        current_transpose = self.transpose
        self.key_signature.key += self.transpose
        self.part.transposes = True

  def parseBackup(self, xml_backup):
    global current_time_position

    xml_duration = xml_backup.find("duration")
    backup_duration = int(xml_duration.text)
    midi_ticks = backup_duration * (TICKS_PER_BEAT / current_divisions)
    seconds = (midi_ticks / TICKS_PER_BEAT) * current_seconds_per_beat
    current_time_position -= seconds

  def parseDirection(self, xml_direction):
    global current_tempo
    global current_seconds_per_beat
    global current_velocity

    for child in xml_direction:
      if child.tag == "sound":
        if child.get("tempo") != None:
          tempo = Tempo(child)
          self.tempos.append(tempo)
          current_tempo = tempo.qpm
          current_seconds_per_beat = 60 / current_tempo
          if child.get("dynamics") != None:
            current_velocity = int(child.get("dynamics"))

  def parseForward(self, xml_forward):
    global current_time_position

    xml_duration = xml_forward.find('duration')
    forward_duration = int(xml_duration.text)
    midi_ticks = forward_duration * (TICKS_PER_BEAT / current_divisions)
    seconds = (midi_ticks / TICKS_PER_BEAT) * current_seconds_per_beat
    current_time_position += seconds


class Note:
  def __init__(self, xml_note):
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
    self.parse()

  def parse(self):
    global current_time_position
    global previous_note

    self.midi_channel = current_midi_channel
    self.midi_program = current_midi_program
    self.velocity = current_velocity

    for child in self.xml_note:
      if child.tag == "chord":
        self.is_in_chord = True
      elif child.tag == "duration":
        self.duration = int(child.text)

        self.midi_ticks = self.duration
        self.midi_ticks *= (TICKS_PER_BEAT / current_divisions)

        self.seconds = (self.midi_ticks / TICKS_PER_BEAT)
        self.seconds *= current_seconds_per_beat

        self.time_position = current_time_position

        if self.is_in_chord:
          # If this is a chord, subtract the duration from the
          # previous note and make this note occur at the same time
          previous_note.time_position -= self.seconds
          self.time_position -= self.seconds
        else:
            # Only increment time positions once in chord
            current_time_position += self.seconds

      elif child.tag == "pitch":
        self.parsePitch(child)
      elif child.tag == "rest":
        self.is_rest = True
      elif child.tag == "voice":
        self.voice = int(child.text)

  def parsePitch(self, xml_pitch):
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
    midi_pitch = self.pitchToMIDIPitch(step, alter, octave)
    # Transpose MIDI pitch
    midi_pitch = midi_pitch + current_transpose
    self.pitch = (pitch_string, midi_pitch)

  def pitchToMIDIPitch(self, step, alter, octave):
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


class TimeSignature:
  def __init__(self, xml_time):
    self.xml_time = xml_time
    self.numerator = -1
    self.denominator = -1
    self.time_position = -1
    self.parse()

  def parse(self):
    self.numerator = int(self.xml_time.find("beats").text)
    self.denominator = int(self.xml_time.find("beat-type").text)
    self.time_position = current_time_position

  def __str__(self):
    time_sig_str = str(self.numerator) + "/" + str(self.denominator)
    time_sig_str += " (@time: " + str(self.time_position) + ")"
    return time_sig_str

  def __eq__(self, other):
    isEqual = self.numerator == other.numerator
    isEqual = isEqual and (self.denominator == other.denominator)
    isEqual = isEqual and (self.time_position == other.time_position)
    return isEqual


class KeySignature:
  def __init__(self, xml_key = None):
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
    self.key = int(self.xml_key.find("fifths").text)
    mode = self.xml_key.find("mode")
    # Anything not minor will be interpreted as major
    if mode != "minor":
      mode = "major"
    self.mode = mode
    self.time_position = current_time_position

  def __str__(self):
    keys = (["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D",
            "A", "E", "B", "F#", "C#"])
    key_string = keys[self.key + 7] + " " + self.mode
    key_string += " (@time: " + str(self.time_position) + ")"
    return key_string

  def __eq__(self, other):
    isEqual = self.key == other.key
    isEqual = isEqual and (self.mode == other.mode)
    isEqual = isEqual and (self.time_position == other.time_position)
    return isEqual


class Tempo:
  def __init__(self, xml_sound = None):
    self.xml_sound = xml_sound
    self.qpm = -1
    self.time_position = -1
    if xml_sound != None:
      self.parse()

  def parse(self):
    self.qpm = float(self.xml_sound.get("tempo"))
    if self.qpm == 0:
        self.qpm = 120   # If tempo is 0, set it to default
    self.time_position = current_time_position

  def __str__(self):
    tempo_str = "Tempo: " + str(self.qpm)
    tempo_str += " (@time: " + str(self.time_position) + ")"
    return tempo_str
