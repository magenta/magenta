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
"""

# internal imports
from magenta.music import chord_symbols_lib
from magenta.music import constants
from magenta.music import encoder_decoder

NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
NO_CHORD = constants.NO_CHORD


class ChordEncodingException(Exception):
  pass


class MajorMinorChordOneHotEncoding(encoder_decoder.OneHotEncoding):
  """Encodes chords as root + major/minor, with zero index for "no chord".

  Encodes chords as follows:
    0:     "no chord"
    1-12:  chords with a major triad, where 1 is C major, 2 is C# major, etc.
    13-24: chords with a minor triad, where 13 is C minor, 14 is C# minor, etc.
  """

  # Mapping from pitch class index to name.  Eventually this should be defined
  # more globally, but right now only `decode_chord` needs it.
  _PITCH_CLASS_MAPPING = ['C', 'C#', 'D', 'E-', 'E', 'F',
                          'F#', 'G', 'A-', 'A', 'B-', 'B']

  def __init__(self, chord_symbol_functions=
               chord_symbols_lib.ChordSymbolFunctions.get()):
    """Initialize the MajorMinorChordOneHotEncoding object.

    Args:
      chord_symbol_functions: ChordSymbolFunctions object with which to perform
          the actual transposition of chord symbol strings.
    """
    self._chord_symbol_functions = chord_symbol_functions

  @property
  def num_classes(self):
    return 2 * NOTES_PER_OCTAVE + 1

  @property
  def default_event(self):
    return NO_CHORD

  def encode_event(self, event):
    if event == NO_CHORD:
      return 0

    root = self._chord_symbol_functions.chord_symbol_root(event)
    quality = self._chord_symbol_functions.chord_symbol_quality(event)

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
      return self._PITCH_CLASS_MAPPING[index - 1]
    else:
      # minor
      return self._PITCH_CLASS_MAPPING[index - NOTES_PER_OCTAVE - 1] + 'm'
