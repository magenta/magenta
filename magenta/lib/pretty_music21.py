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

"""A read_only pretty_midi like wrapper for Music21 score objects for ease of conversion to NoteSequence proto."""

from collections import namedtuple

import music21


# Default bpm if tempo mark not available.  Chosen to so that quarterLength equals 1s.
_DEFAULT_BPM = 60

# Music21 to NoteSequence proto conversions on key and mode
_MAJOR_KEY_MUSIC21_TO_NOTE_SEQUENCE = {-6: 6, -5: 1, -4: 8, -3: 3, -2: 10, -1: 5,
                                       0: 0, 1: 7, 2: 2, 3: 9, 4: 4, 5: 11, 6: 6}

_MINOR_KEY_MUSIC21_TO_NOTE_SEQUENCE = {-6: 3, -5: -2, -4: 5, -3: 0, -2: 7, -1: 2,
                                           0: -3, 1: 4, 2: -1, 3: 6, 4: 1, 5: 8, 6: 3}

_MUSIC21_TO_NOTE_SEQUENCE_MODE = {'major': 0, 'minor': 1}

# To account for Musescore settings, and borrowing from Musescore standards
_VOICES_PER_STAFF = 4
_MUSESCORE_PITCH_CLASS_ENUM = {'Fbb': -1, 'Cbb': 0, 'Gbb': 1, 'Dbb': 2, 'Abb': 3, 'Ebb': 4, 'Bbb': 5,
                               'Fb': 6, 'Cb': 7, 'Gb': 8, 'Db': 9, 'Ab': 10, 'Eb': 11, 'Bb': 12,
                               'F': 13, 'C': 14, 'G': 15, 'D': 16, 'A': 17, 'E': 18, 'B': 19,
                               'F#': 20, 'C#': 21, 'G#': 22, 'D#': 23, 'A#': 24, 'E#': 25, 'B#': 26,
                               'F##': 27, 'C##': 28, 'G##': 29, 'D##': 30, 'A##': 31, 'E##': 32, 'B##': 33}

TimeSignature = namedtuple("TimeSignature", ['time', 'numerator', 'denominator'])
Tempo = namedtuple("Tempo", ["time", "bpm"])
KeySignature = namedtuple("KeySignature", ["time", "key_number", "mode"])
PartInfo = namedtuple("PartInfo", ["index", "name"])
Note = namedtuple("Note", ["pitch", "pitch_class", "start", "end", "part"])


def convert_time(time):
  """In preparation for when we want to transform time to be in another unit"""
  # time: quarter note length
  return time


class PrettyMusic21(object):
  """A read_only pretty_midi like wrapper for music21 _score objects"""

  # TODO: Should assert when parts do not sure the same markings (time sig, key sig, tempo, etc),
  #       as note sequence proto assumes these are the same for all parts.
  #       To do this, we can loop through the top voice, and get the measures for where the a marking changes
  #       and then check to make sure all the other voices have the same change.
  # TODO: Time in score-based quarter notes, which does not take tempo markings into account.
  # Look at: http://web.mit.edu/music21/doc/moduleReference/moduleBase.html#music21.base.Music21Object.seconds
  # Search Stream.metronomeMarkBoundaries(srcObj=None)
  #       in http://web.mit.edu/music21/doc/moduleReference/moduleStream.html

  def __init__(self, score):
    self._score = score
    self._parts = [part.semiFlat for part in score.parts]

  @property
  def id(self):
    return str(self._score.id)

  @property
  def title(self):
    if self._score.metadata is not None:
      return self._score.metadata.title
    return None

  @property
  def composer(self):
    if self._score.metadata is not None:
      return self._score.metadata.composer
    return None

  @property
  def total_time(self):
    return convert_time(self._score.duration.quarterLength)

  @property
  def time_signature_changes(self):
    """Collect unique time signature changes in score, and add pick-up time signatures as necessary."""
    # TODO: to some extent assumes all voices have the same time signature
    # TODO: assumes that when there is a time signature present, it is embedded in a measure
    time_sig_changes = []
    for part_num, part in enumerate(self._parts):
      for time_sig in part.getElementsByClass('TimeSignature'):
        measure = time_sig.getContextByClass("Measure")
        # TODO: raise exception?
        assert measure is not None, 'Time sig needs to be in a measure'
        global_time = convert_time(part.elementOffset(time_sig))
        if measure.duration.quarterLength < time_sig.barDuration.quarterLength:
          # First, add the pick-up time sig
          pickup_time_sig = music21.meter.bestTimeSignature(measure)
          pickup_time_sig_change = TimeSignature(global_time, pickup_time_sig.numerator,
                                                 pickup_time_sig.denominator)
          if pickup_time_sig_change not in time_sig_changes:
            time_sig_changes.append(pickup_time_sig_change)

          # Advance global tick to the beginning of next measure, to prepare for adding full time sig
          # Can not use part for retrieving measure offset because flatted parts do not contain measures
          global_time = convert_time(global_time + measure.duration.quarterLength)

          # This line didn't work b/c measure would have an id that is not recongized in original parts
          # global_time = convert_time(part.elementOffset(measure.next("Measure")))

          # TODO: check that the next time signature change is not at this exact time and different,
          #     in practice during reconstruction, the later time signature would take effect anyway

        # Add the full time sig
        full_time_sig_change = TimeSignature(global_time, time_sig.numerator, time_sig.denominator)
        if full_time_sig_change not in time_sig_changes:
          time_sig_changes.append(full_time_sig_change)
    return time_sig_changes

  @property
  def tempo_changes(self):
    """Collect unique tempo changes.  It no tempo indication, defaults to DEFAULT_BPM"""
    # TODO: Some scores do not have tempo markings.  Give a default tempo marking.
    tempo_changes = []
    for part_num, part in enumerate(self._parts):
      for metronome_mark in part.getElementsByClass('MetronomeMark'):
        global_time = convert_time(part.elementOffset(metronome_mark))
        tempo_change = Tempo(global_time, metronome_mark.number)
        if tempo_change not in tempo_changes:
          tempo_changes.append(tempo_change)
    if not len(tempo_changes):
      tempo_changes.append(Tempo(0, _DEFAULT_BPM))
    return tempo_changes

  @property
  def key_signature_changes(self):
    """Collect unique key signature changes."""
    key_sig_changes = []
    for part_num, part in enumerate(self._parts):
      for ks in part.getElementsByClass('KeySignature'):
        global_time = convert_time(part.elementOffset(ks))

        mode = None
        if ks.mode in _MUSIC21_TO_NOTE_SEQUENCE_MODE:
          mode = _MUSIC21_TO_NOTE_SEQUENCE_MODE[ks.mode]

        key_number = None
        # major mode
        if mode == 0 and ks.sharps in _MAJOR_KEY_MUSIC21_TO_NOTE_SEQUENCE:
          key_number = _MAJOR_KEY_MUSIC21_TO_NOTE_SEQUENCE[ks.sharps]
        # minor mode
        elif mode == 1 and ks.sharps in _MINOR_KEY_MUSIC21_TO_NOTE_SEQUENCE:
          key_number = _MINOR_KEY_MUSIC21_TO_NOTE_SEQUENCE[ks.sharps]

        key_sig_change = KeySignature(global_time, key_number, mode)
        if key_sig_change not in key_sig_changes:
          key_sig_changes.append(key_sig_change)
    return key_sig_changes

  @property
  def part_infos(self):
    """Collect part information as global index in score and part name."""
    part_infos = []
    for part_num, part in enumerate(self._parts):
      index = part_num * _VOICES_PER_STAFF
      part_info = PartInfo(index, part.id)
      part_infos.append(part_info)
    return part_infos

  @property
  def parts(self):
    """Collect all the notes, each part in a list."""
    simple_parts = []
    for part_num, part in enumerate(self._parts):
      simple_part = []
      for music21_note in part.getElementsByClass('Note'):
        pitch_midi = music21_note.pitch.midi
        pitch_class = music21_to_musescore_pitch_class(music21_note)
        # TODO: distinguish between symbolic time and performance time
        start = convert_time(part.elementOffset(music21_note))
        end = start + convert_time(music21_note.duration.quarterLength)
        part_index = part_num * _VOICES_PER_STAFF
        note = Note(pitch_midi, pitch_class, start, end, part_index)
        simple_part.append(note)
        # TODO: put in note.numerator and note.denominator
      simple_parts.append(simple_part)
    return simple_parts

  @property
  def sorted_notes(self):
    flatted_notes = []
    for part in self.parts:
      flatted_notes.extend(part)
    return sorted(flatted_notes, key=lambda x: x.start)


def music21_to_musescore_pitch_class(music21_note):
  """Convert music21 pitch names to MuseScore like pitch class enumeration."""
  # music21 represents flat(s) as '-' and '--'
  music21_note_pitch = music21_note.pitch.name.replace('-', 'b')
  if music21_note_pitch in _MUSESCORE_PITCH_CLASS_ENUM:
    return _MUSESCORE_PITCH_CLASS_ENUM[music21_note_pitch]
  return None
