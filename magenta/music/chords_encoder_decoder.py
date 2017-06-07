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
"""Classes for converting between chord progressions and models inputs/outputs.

MajorMinorChordOneHotEncoding is an encoding.OneHotEncoding that specifies a
one-hot encoding for ChordProgression events, i.e. chord symbol strings. This
encoding has 25 classes, all 12 major and minor triads plus "no chord".

TriadChordOneHotEncoding is another encoding.OneHotEncoding that specifies a
one-hot encoding for ChordProgression events, i.e. chord symbol strings. This
encoding has 49 classes, all 12 major/minor/augmented/diminished triads plus
"no chord".
"""

# internal imports
from magenta.music import chord_symbols_lib
from magenta.music import constants
from magenta.music import encoder_decoder

NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
NO_CHORD = constants.NO_CHORD

# Mapping from pitch class index to name.
_PITCH_CLASS_MAPPING = ['C', 'C#', 'D', 'Eb', 'E', 'F',
                        'F#', 'G', 'Ab', 'A', 'Bb', 'B']


class ChordEncodingException(Exception):
  pass


class MajorMinorChordOneHotEncoding(encoder_decoder.OneHotEncoding):
  """Encodes chords as root + major/minor, with zero index for "no chord".

  Encodes chords as follows:
    0:     "no chord"
    1-12:  chords with a major triad, where 1 is C major, 2 is C# major, etc.
    13-24: chords with a minor triad, where 13 is C minor, 14 is C# minor, etc.
  """

  @property
  def num_classes(self):
    return 2 * NOTES_PER_OCTAVE + 1

  @property
  def default_event(self):
    return NO_CHORD

  def encode_event(self, event):
    if event == NO_CHORD:
      return 0

    root = chord_symbols_lib.chord_symbol_root(event)
    quality = chord_symbols_lib.chord_symbol_quality(event)

    if quality == chord_symbols_lib.CHORD_QUALITY_MAJOR:
      return root + 1
    elif quality == chord_symbols_lib.CHORD_QUALITY_MINOR:
      return root + NOTES_PER_OCTAVE + 1
    else:
      raise ChordEncodingException('chord is neither major nor minor: %s'
                                   % event)

  def decode_event(self, index):
    if index == 0:
      return NO_CHORD
    elif index - 1 < 12:
      # major
      return _PITCH_CLASS_MAPPING[index - 1]
    else:
      # minor
      return _PITCH_CLASS_MAPPING[index - NOTES_PER_OCTAVE - 1] + 'm'


class TriadChordOneHotEncoding(encoder_decoder.OneHotEncoding):
  """Encodes chords as root + triad type, with zero index for "no chord".

  Encodes chords as follows:
    0:     "no chord"
    1-12:  chords with a major triad, where 1 is C major, 2 is C# major, etc.
    13-24: chords with a minor triad, where 13 is C minor, 14 is C# minor, etc.
    25-36: chords with an augmented triad, where 25 is C augmented, etc.
    37-48: chords with a diminished triad, where 37 is C diminished, etc.
  """

  @property
  def num_classes(self):
    return 4 * NOTES_PER_OCTAVE + 1

  @property
  def default_event(self):
    return NO_CHORD

  def encode_event(self, event):
    if event == NO_CHORD:
      return 0

    root = chord_symbols_lib.chord_symbol_root(event)
    quality = chord_symbols_lib.chord_symbol_quality(event)

    if quality == chord_symbols_lib.CHORD_QUALITY_MAJOR:
      return root + 1
    elif quality == chord_symbols_lib.CHORD_QUALITY_MINOR:
      return root + NOTES_PER_OCTAVE + 1
    elif quality == chord_symbols_lib.CHORD_QUALITY_AUGMENTED:
      return root + 2 * NOTES_PER_OCTAVE + 1
    elif quality == chord_symbols_lib.CHORD_QUALITY_DIMINISHED:
      return root + 3 * NOTES_PER_OCTAVE + 1
    else:
      raise ChordEncodingException('chord is not a standard triad: %s' % event)

  def decode_event(self, index):
    if index == 0:
      return NO_CHORD
    elif index - 1 < 12:
      # major
      return _PITCH_CLASS_MAPPING[index - 1]
    elif index - NOTES_PER_OCTAVE - 1 < 12:
      # minor
      return _PITCH_CLASS_MAPPING[index - NOTES_PER_OCTAVE - 1] + 'm'
    elif index - 2 * NOTES_PER_OCTAVE - 1 < 12:
      # augmented
      return _PITCH_CLASS_MAPPING[index - 2 * NOTES_PER_OCTAVE - 1] + 'aug'
    else:
      # diminished
      return _PITCH_CLASS_MAPPING[index - 3 * NOTES_PER_OCTAVE - 1] + 'dim'


class PitchChordsEncoderDecoder(encoder_decoder.EventSequenceEncoderDecoder):
  """An encoder/decoder for chords that encodes chord root, pitches, and bass.

  This class has no label encoding and can only be used to encode chords as
  model input vectors. It can be used to help generate another type of event
  sequence (e.g. melody) conditioned on chords.
  """

  @property
  def input_size(self):
    return 3 * NOTES_PER_OCTAVE + 1

  @property
  def num_classes(self):
    raise NotImplementedError

  @property
  def default_event_label(self):
    raise NotImplementedError

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the chord progression.

    Indices [0, 36]:
    [0]: Whether or not this chord is "no chord".
    [1, 12]: A one-hot encoding of the chord root pitch class.
    [13, 24]: Whether or not each pitch class is present in the chord.
    [25, 36]: A one-hot encoding of the chord bass pitch class.

    Args:
      events: A magenta.music.ChordProgression object.
      position: An integer event position in the chord progression.

    Returns:
      An input vector, an self.input_size length list of floats.
    """
    chord = events[position]
    input_ = [0.0] * self.input_size

    if chord == NO_CHORD:
      input_[0] = 1.0
      return input_

    root = chord_symbols_lib.chord_symbol_root(chord)
    input_[1 + root] = 1.0

    pitches = chord_symbols_lib.chord_symbol_pitches(chord)
    for pitch in pitches:
      input_[1 + NOTES_PER_OCTAVE + pitch] = 1.0

    bass = chord_symbols_lib.chord_symbol_bass(chord)
    input_[1 + 2 * NOTES_PER_OCTAVE + bass] = 1.0

    return input_

  def events_to_label(self, events, position):
    raise NotImplementedError

  def class_index_to_event(self, class_index, events):
    raise NotImplementedError
