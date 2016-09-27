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
"""Utility functions for working with chord symbols."""

import abc
import music21
import tensorflow as tf

# chord quality enum
CHORD_QUALITY_MAJOR = 0
CHORD_QUALITY_MINOR = 1
CHORD_QUALITY_AUGMENTED = 2
CHORD_QUALITY_DIMINISHED = 3
CHORD_QUALITY_OTHER = 4


class ChordSymbolException(Exception):
  pass


class ChordSymbolFunctions(object):
  """An abstract class for interpreting chord symbol strings.

  This abstract class is an interface specifying several functions for the
  interpretation of chord symbol strings:

  `transpose_chord_symbol` transposes a chord symbol a given number of steps.
  `chord_symbol_midi_pitches` returns a list of MIDI pitches in a chord.
  `chord_symbol_root` returns the root pitch class of a chord.
  `chord_symbol_quality` returns the "quality" of a chord.
  """
  __metaclass__ = abc.ABCMeta

  @staticmethod
  def get():
    """Returns the default implementation of ChordSymbolFunctions.

    Currently the default (and only) implementation of ChordSymbolFunctions is
    Music21ChordSymbolFunctions.

    Returns:
      A ChordSymbolFunctions object.
    """
    return Music21ChordSymbolFunctions()

  @abc.abstractmethod
  def transpose_chord_symbol(self, figure, transpose_amount):
    """Transposes a chord symbol figure string by the given amount.

    Args:
      figure: The chord symbol figure string to transpose.
      transpose_amount: The integer number of half steps to transpose.

    Returns:
      The transposed chord symbol figure string.

    Raises:
      ChordSymbolException: If the given chord symbol cannot be interpreted.
    """
    pass

  @abc.abstractmethod
  def chord_symbol_midi_pitches(self, figure):
    """Return the pitches of a chord as MIDI note values.

    Args:
      figure: The chord symbol figure string for which pitches are computed.

    Returns:
      A python list of pitches as integer MIDI note values.

    Raises:
      ChordSymbolException: If the given chord symbol cannot be interpreted.
    """
    pass

  @abc.abstractmethod
  def chord_symbol_root(self, figure):
    """Return the root pitch class of a chord.

    Args:
      figure: The chord symbol figure string for which pitches are computed.

    Returns:
      The pitch class of the chord root, an integer between 0 and 11 inclusive.

    Raises:
      ChordSymbolException: If the given chord symbol cannot be interpreted.
    """
    pass

  @abc.abstractmethod
  def chord_symbol_quality(self, figure):
    """Return the quality (major, minor, dimished, augmented) of a chord.

    Args:
      figure: The chord symbol figure string for which quality is computed.

    Returns:
      One of CHORD_QUALITY_MAJOR, CHORD_QUALITY_MINOR, CHORD_QUALITY_AUGMENTED,
      CHORD_QUALITY_DIMINISHED, or CHORD_QUALITY_UNKNOWN.

    Raises:
      ChordSymbolException: If the given chord symbol cannot be interpreted.
    """
    pass


class Music21ChordSymbolFunctions(ChordSymbolFunctions):
  """A class that uses music21 to interpret chord symbol strings."""

  # music21 returns this ugly string when it can't parse a chord.
  _MUSIC21_UNIDENTIFIED_CHORD = 'Chord Symbol Cannot Be Identified'

  # Mapping from strings returned by music21 to chord quality enum values.
  _music21_chord_quality_mapping = {
      'major': CHORD_QUALITY_MAJOR,
      'minor': CHORD_QUALITY_MINOR,
      'augmented': CHORD_QUALITY_AUGMENTED,
      'diminished': CHORD_QUALITY_DIMINISHED
  }

  def __init__(self):
    """Construct a Music21ChordSymbolFunctions object."""
    self._music21_chord_symbol_dict = {}

  def _to_music21_chord_symbol(self, figure):
    """Return a music21.harmony.ChordSymbol object instantiated with `figure`.

    Since this operation can be slow and chord symbols are often repeated many
    times in a single score, we also memoize the mapping.

    Args:
      figure: The chord symbol figure string.

    Returns:
      A music21.harmony.ChordSymbol object.

    Raises:
      ChordSymbolException: If the chord fails to be parsed by music21.
    """

    if figure in self._music21_chord_symbol_dict:
      return self._music21_chord_symbol_dict[figure]

    try:
      cs = music21.harmony.ChordSymbol(figure)
      self._music21_chord_symbol_dict[figure] = cs
      return cs
    except:  # pylint: disable=bare-except
      pass

    try:
      # music21 seems to have a hard time parsing some chord symbol figure
      # strings it itself produces! It sometimes produces strings like
      # "C7 add b9" or "G7 alter #5", and then can't re-parse them. In these
      # cases, let's try again with just the basic chord.
      cs = music21.harmony.ChordSymbol(figure.split()[0])
      self._music21_chord_symbol_dict[figure] = cs
      tf.logging.warn('Failed to parse chord symbol %s, '
                      'interpreting as %s', figure, figure.split()[0])
      return cs
    except:
      raise ChordSymbolException('unable to parse chord symbol: %s'
                                 % figure)

  def transpose_chord_symbol(self, figure, transpose_amount):
    chord_symbol = self._to_music21_chord_symbol(figure)
    transposed_figure = chord_symbol.transpose(transpose_amount).findFigure()
    if transposed_figure == self._MUSIC21_UNIDENTIFIED_CHORD:
      # music21 just returns error text instead of throwing.
      raise ChordSymbolException('unable to parse chord symbol: %s'
                                 % figure)
    else:
      return transposed_figure

  def chord_symbol_midi_pitches(self, figure):
    chord_symbol = self._to_music21_chord_symbol(figure)
    return [pitch.midi for pitch in chord_symbol.pitches]

  def chord_symbol_root(self, figure):
    chord_symbol = self._to_music21_chord_symbol(figure)
    return chord_symbol.root().pitchClass

  def chord_symbol_quality(self, figure):
    chord_symbol = self._to_music21_chord_symbol(figure)
    quality_string = chord_symbol.quality
    if quality_string not in self._music21_chord_quality_mapping:
      return CHORD_QUALITY_OTHER
    else:
      return self._music21_chord_quality_mapping[quality_string]
