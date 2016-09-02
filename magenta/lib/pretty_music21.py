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
"""A read_only pretty_midi-like wrapper for music21 score objects.

It exposes a small set of score attrbutes from music21, mostly chosen to
support magenta.protobuf.NoteSequence.
"""

from collections import namedtuple
import hashlib

# internal imports
import music21

# Default qpm if tempo mark not available.
_DEFAULT_QPM = 120

TimeSignature = namedtuple('TimeSignature',
                           ['time', 'numerator', 'denominator'])
Tempo = namedtuple('Tempo', ['time', 'qpm'])
KeySignature = namedtuple('KeySignature', ['time', 'key', 'num_sharps', 'mode',
                                           'tonic_pitchclass'])
PartInfo = namedtuple('PartInfo', ['index', 'name'])

# TODO(annahuang): Add octave.
Note = namedtuple('Note', ['pitch_midi', 'pitch_name', 'start_time', 'end_time',
                           'part_index'])


class PrettyMusic21Error(Exception):
  """Exception for music attributes violating what PrettyMusic21 supports."""
  pass


def _extract_key_signature_attributes(key_signature):
  """Extracts attributes of key signatures from music21.key.KeySignature.

  Args:
    key_signature: A music21.key.KeySignature object.

  Returns:
    tonic_pitch_name: A string that specifies the pitch name of the tonic in the
        key extracted from key_signature.
    key_signature.sharps: An integer specifying the nubmer of sharps in this
        key signature.
    mode: A string giving the mode, usually 'major' or 'minor'

  Raises:
    TypeError: When key_signature is not of the expected type
        music21.key.KeySignature.
    PrettyMusic21Error: When the number of sharps in key_signature is out of
        the expected range.
    AttributeError: When key_signature does not have the mode attribute.
  """
  if not isinstance(key_signature, music21.key.KeySignature):
    raise TypeError(
        'Type Music21.key.KeySignature expected but %s type received.' %
        (type(key_signature)))
  if key_signature.sharps < -7 or key_signature.sharps > 7:
    raise PrettyMusic21Error(
        'Key Signatures with more than 7 sharps are flats are not supported.')
  try:
    mode = key_signature.mode
  except AttributeError:
    mode = None
  if mode is not None and mode.lower() == 'minor':
    # To convert a major key to its relative minor, plus 3 to the number of
    # sharps and get the corresponding major key pitch name.
    # For example, 3 flats (-3) is Eb major, or c minor. While -3 + 3 is C.
    tonic_pitch = music21.key.sharpsToPitch(key_signature.sharps + 3)
  else:
    # Assume to be major.
    tonic_pitch = music21.key.sharpsToPitch(key_signature.sharps)
  tonic_pitch_name = tonic_pitch.name.replace('-', 'b')
  if mode is not None and mode.lower() == 'minor':
    tonic_pitch_name = tonic_pitch_name.lower()
  return tonic_pitch_name, key_signature.sharps, mode, tonic_pitch.pitchClass


class PrettyMusic21(object):
  """A read_only pretty_midi-like wrapper for music21 score objects.

  Args:
    score: A PrettyMusic21 object.
    filename: A string for the source filename from which the score was
        extracted.
  """
  # TODO(annahuang): Should raise an error when parts don't share same markings.
  # (time sig, key sig, tempo, etc),
  # as note sequence proto assumes these are the same for all parts.
  # To do this, we can loop through the top voice, and get the measures
  # for where the a marking changes
  # and then check to make sure all the other voices have the same change.

  # TODO(annahuang): Add performance time.
  # Time currently is symbolic, and does not take tempo markings into account.
  # Look at: http://web.mit.edu/music21/doc/moduleReference/moduleBase.html
  #   #music21.base.Music21Object.seconds
  # Search Stream.metronomeMarkBoundaries(srcObj=None)
  #   in http://web.mit.edu/music21/doc/moduleReference/moduleStream.html

  def __init__(self, score, filename=None):
    self._score = score
    self._parts = [part.semiFlat for part in score.parts]
    self._filename = filename

  @property
  def id(self):
    """Uses a hash of the score string as id."""
    # TODO(annahuang): Check why the hash is not the same for scores created
    # from the same file.
    converter = music21.converter.subConverters.ConverterText()
    return hashlib.sha1(str(converter.write(self._score, None)))

  @property
  def title(self):
    """Returns the title of the score if available."""
    if self._score.metadata is not None:
      return self._score.metadata.title
    return self._filename

  @property
  def composer(self):
    """Returns the composer of the score if available."""
    if self._score.metadata is not None:
      return self._score.metadata.composer
    return None

  @property
  def filename(self):
    return self._filename

  @property
  def total_time(self):
    """Returns the total quarterLength duration of the score."""
    return self._convert_time(self._score.duration.quarterLength)

  @property
  def time_signature_changes(self):
    """Collects unique time signature changes, and add pick-up when necessary.

    Returns:
      A list of unique TimeSignature namedtuples, sorted by the time attribute.

    Raises:
      PrettyMusic21Error: When a time signature is not within a measure
        container.
    """
    # TODO(annahuang): Don't assume all voices have the same time signature.
    # Assumes time signatures are always embedded in a measure.
    time_sig_changes = set()
    for part in self._parts:
      for time_sig in part.getElementsByClass('TimeSignature'):
        measure = time_sig.getContextByClass('Measure')
        if measure is None:
          raise PrettyMusic21Error('Time signatures need to be in a measure.')
        global_time = self._convert_time(part.elementOffset(time_sig))
        if measure.duration.quarterLength < time_sig.barDuration.quarterLength:
          # First, add the pick-up time signature.
          pickup_time_sig = music21.meter.bestTimeSignature(measure)
          pickup_time_sig_change = TimeSignature(global_time,
                                                 pickup_time_sig.numerator,
                                                 pickup_time_sig.denominator)
          time_sig_changes.add(pickup_time_sig_change)

          global_time = self._convert_time(global_time +
                                           measure.duration.quarterLength)

        # Add the full time signature.
        full_time_sig_change = TimeSignature(global_time, time_sig.numerator,
                                             time_sig.denominator)
        time_sig_changes.add(full_time_sig_change)

    return sorted(time_sig_changes, key=lambda x: x.time)

  @property
  def tempo_changes(self):
    """Collects unique tempo changes. If no tempo, defaults to _DEFAULT_QPM.

    Returns:
      A list of unique Tempo namedtuples, sorted by the time attribute.
    """
    tempo_changes = set()
    for part in self._parts:
      for metronome_mark in part.getElementsByClass('MetronomeMark'):
        global_time = self._convert_time(part.elementOffset(metronome_mark))
        tempo_change = Tempo(global_time, metronome_mark.number)
        tempo_changes.add(tempo_change)
    if not tempo_changes:
      tempo_changes.add(Tempo(0, _DEFAULT_QPM))
    return sorted(tempo_changes, key=lambda x: x.time)

  @property
  def key_signature_changes(self):
    """Collects unique key signature changes.

    Returns:
      A list of unique KeySignature namedtuples, sorted by the time attribute.
    """
    key_sig_changes = set()
    for part in self._parts:
      for ks in part.getElementsByClass('KeySignature'):
        global_time = self._convert_time(part.elementOffset(ks))
        key_sig_change = KeySignature(global_time,
                                      *(_extract_key_signature_attributes(ks)))
        key_sig_changes.add(key_sig_change)
    return sorted(key_sig_changes, key=lambda x: x.time)

  @property
  def part_infos(self):
    """Collects part information as global index in score and part name.

    Returns:
      A list of unique PartInfo namedtuples.
    """
    parts_infos = []
    for part_num, part in enumerate(self._parts):
      index = part_num
      parts_info = PartInfo(index, part.id)
      parts_infos.append(parts_info)
    return parts_infos

  @property
  def parts(self):
    """Collects all the notes, each part in a list.

    Returns:
      A list of lists of Note namedtuples.
    """
    simple_parts = []
    for part_num, part in enumerate(self._parts):
      simple_part = []
      for music21_note in part.getElementsByClass('Note'):
        pitch_midi = music21_note.pitch.midi
        # TODO(annahuang): Add octave.
        pitch_name = music21_note.pitch.name.replace('-', 'b')
        # TODO(annahuang): Distinguish between symbolic and performance time.
        start_time = self._convert_time(part.elementOffset(music21_note))
        end_time = start_time + self._convert_time(
            music21_note.duration.quarterLength)
        part_index = part_num
        note = Note(pitch_midi, pitch_name, start_time, end_time, part_index)
        simple_part.append(note)
        # TODO(annahuang): Add note.numerator and note.denominator.
      simple_parts.append(simple_part)
    return simple_parts

  @property
  def sorted_notes(self):
    """Sorts all the notes according to start_time time.

    Returns:
      A list of all Note namedtuples, sorted by start time.
    """
    flatted_notes = []
    for part in self.parts:
      flatted_notes.extend(part)
    return sorted(flatted_notes, key=lambda x: x.start_time)

  def _convert_time(self, quarter_length):
    """Transforms quarter-note counts into seconds according to _DEFAULT_QPM.

    Args:
      quarter_length: A float that specifies duration in quarter-note units.

    Returns:
      A float in seconds.
    """
    # TODO(annahuang): Take tempo change into account.
    # Time is in quarter-note counts from the beginning of the score.
    return quarter_length * 60.0 / _DEFAULT_QPM
