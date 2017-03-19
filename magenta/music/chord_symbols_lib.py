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
import re

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
  `chord_symbol_pitches` returns a list of pitch classes in a chord.
  `chord_symbol_root` returns the root pitch class of a chord.
  `chord_symbol_bass` returns the bass pitch class of a chord.
  `chord_symbol_quality` returns the "quality" of a chord.
  """
  __metaclass__ = abc.ABCMeta

  @staticmethod
  def get():
    """Returns the default implementation of ChordSymbolFunctions.

    Currently the default (and only) implementation of ChordSymbolFunctions is
    BasicChordSymbolFunctions.

    Returns:
      A ChordSymbolFunctions object.
    """
    return BasicChordSymbolFunctions()

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
  def chord_symbol_pitches(self, figure):
    """Return the pitch classes contained in a chord.

    This will generally include the root pitch class, but not the bass if it is
    not otherwise one of the pitches in the chord.

    Args:
      figure: The chord symbol figure string for which pitches are computed.

    Returns:
      A python list of integer pitch class values.

    Raises:
      ChordSymbolException: If the given chord symbol cannot be interpreted.
    """
    pass

  @abc.abstractmethod
  def chord_symbol_root(self, figure):
    """Return the root pitch class of a chord.

    Args:
      figure: The chord symbol figure string for which the root is computed.

    Returns:
      The pitch class of the chord root, an integer between 0 and 11 inclusive.

    Raises:
      ChordSymbolException: If the given chord symbol cannot be interpreted.
    """
    pass

  @abc.abstractmethod
  def chord_symbol_bass(self, figure):
    """Return the bass pitch class of a chord.

    Args:
      figure: The chord symbol figure string for which the bass is computed.

    Returns:
      The pitch class of the chord bass, an integer between 0 and 11 inclusive.

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
      CHORD_QUALITY_DIMINISHED, or CHORD_QUALITY_OTHER.

    Raises:
      ChordSymbolException: If the given chord symbol cannot be interpreted.
    """
    pass


class BasicChordSymbolFunctions(ChordSymbolFunctions):
  """Functions for parsing and interpreting chord symbol figure strings.

  The functions in this class treat a chord symbol string as having four
  components:

    Root: The root pitch class of the chord, e.g. 'C#'.
    Kind: The type of chord, e.g. major, half-diminished, 13th. In the chord
        symbol figure string the chord kind is abbreviated, e.g. 'm' or '-'
        means minor, 'dim' or 'o' means diminished, '7' means dominant 7th.
    Scale degree modifications: Zero or more modifications to the scale degrees
        present in the chord. There are three different modification types:
        addition (add a new scale degree), subtraction (remove a scale degree),
        and alteration (modify a scale degree). For example, '#9' means to add
        a raised 9th, or to alter the 9th to be raised if a 9th was already
        present in the chord. Other possible modification strings include 'no3'
        (remove the 3rd scale degree), 'add2' (add the 2nd scale degree), 'b5'
        (flatten the 5th scale degree), etc.
    Bass: The bass pitch class of the chord. If missing, the bass pitch class is
        assumed to be the same as the root pitch class.

  Before doing any other operations, the functions in this class attempt to
  split the chord symbol figure string into these four components; if that
  attempt fails a ChordSymbolException is raised.

  After that, some operations leave some of the components unexamined, e.g.
  transposition only modifies the root and bass, leaving the chord kind and
  scale degree modifications unchanged.
  """

  # Intervals between scale steps.
  _STEPS_ABOVE = {'A': 2, 'B': 1, 'C': 2, 'D': 2, 'E': 1, 'F': 2, 'G': 2}

  # Scale steps to MIDI mapping.
  _STEPS_MIDI = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

  # Mapping from scale degree to offset in half steps.
  _DEGREE_OFFSETS = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

  # List of chord kinds with abbreviations and scale degrees. Scale degrees are
  # represented as strings here a) for human readability, and b) because the
  # number of semitones is insufficient when the chords have scale degree
  # modifications.
  _CHORD_KINDS = [
      # major triad
      (['', 'maj', 'M'],
       ['1', '3', '5']),

      # minor triad
      (['m', 'min', '-'],
       ['1', 'b3', '5']),

      # augmented triad
      (['+', 'aug'],
       ['1', '3', '#5']),

      # diminished triad
      (['o', 'dim'],
       ['1', 'b3', 'b5']),

      # dominant 7th
      (['7'],
       ['1', '3', '5', 'b7']),

      # major 7th
      (['maj7', 'M7'],
       ['1', '3', '5', '7']),

      # minor 7th
      (['m7', 'min7', '-7'],
       ['1', 'b3', '5', 'b7']),

      # diminished 7th
      (['o7', 'dim7'],
       ['1', 'b3', 'b5', 'bb7']),

      # augmented 7th
      (['+7', 'aug7'],
       ['1', '3', '#5', 'b7']),

      # half-diminished
      (['m7b5', '-7b5', '/o', '/o7'],
       ['1', 'b3', 'b5', 'b7']),

      # minor triad with major 7th
      (['mmaj7', 'mM7', 'minmaj7', 'minM7', '-maj7', '-M7',
        'm(maj7)', 'm(M7)', 'min(maj7)', 'min(M7)', '-(maj7)', '-(M7)'],
       ['1', 'b3', '5', '7']),

      # major 6th
      (['6'],
       ['1', '3', '5', '6']),

      # minor 6th
      (['m6', 'min6', '-6'],
       ['1', 'b3', '5', '6']),

      # dominant 9th
      (['9'],
       ['1', '3', '5', 'b7', '9']),

      # major 9th
      (['maj9', 'M9'],
       ['1', '3', '5', '7', '9']),

      # minor 9th
      (['m9', 'min9', '-9'],
       ['1', 'b3', '5', 'b7', '9']),

      # augmented 9th
      (['+9', 'aug9'],
       ['1', '3', '#5', 'b7', '9']),

      # 6/9 chord
      (['6/9'],
       ['1', '3', '5', '6', '9']),

      # dominant 11th
      (['11'],
       ['1', '3', '5', 'b7', '9', '11']),

      # major 11th
      (['maj11', 'M11'],
       ['1', '3', '5', '7', '9', '11']),

      # minor 11th
      (['m11', 'min11', '-11'],
       ['1', 'b3', '5', 'b7', '9', '11']),

      # dominant 13th
      (['13'],
       ['1', '3', '5', 'b7', '9', '11', '13']),

      # major 13th
      (['maj13', 'M13'],
       ['1', '3', '5', '7', '9', '11', '13']),

      # minor 13th
      (['m13', 'min13', '-13'],
       ['1', 'b3', '5', 'b7', '9', '11', '13']),

      # suspended 2nd
      (['sus2'],
       ['1', '2', '5']),

      # suspended 4th
      (['sus', 'sus4'],
       ['1', '4', '5']),

      # suspended 4th with dominant 7th
      (['sus7', '7sus'],
       ['1', '4', '5', 'b7']),

      # pedal point
      (['ped'],
       ['1']),

      # power chord
      (['5'],
       ['1', '5'])
  ]

  # Dictionary mapping chord kind abbreviations to names and scale degrees.
  _CHORD_KINDS_BY_ABBREV = dict((abbrev, degrees)
                                for abbrevs, degrees in _CHORD_KINDS
                                for abbrev in abbrevs)

  # Function to add a scale degree.
  def _add_scale_degree(self, degrees, degree, alter):
    if degree in degrees:
      raise ChordSymbolException('Scale degree already in chord: %d' % degree)
    if degree == 7:
      alter -= 1
    degrees[degree] = alter

  # Function to remove a scale degree.
  def _subtract_scale_degree(self, degrees, degree, unused_alter):
    if degree not in degrees:
      raise ChordSymbolException('Scale degree not in chord: %d' % degree)
    del degrees[degree]

  # Function to alter (or add) a scale degree.
  def _alter_scale_degree(self, degrees, degree, alter):
    if degree in degrees:
      degrees[degree] += alter
    else:
      degrees[degree] = alter

  # Scale degree modifications. There are three basic types of modifications:
  # addition, subtraction, and alteration. These have been expanded into six
  # types to aid in parsing, as each of the three basic operations has its own
  # requirements on the scale degree operand:
  #
  #  - Addition can accept altered and unaltered scale degrees.
  #  - Subtraction can only accept unaltered scale degrees.
  #  - Alteration can only accept altered scale degrees.

  _DEGREE_MODIFICATIONS = {
      'add': (_add_scale_degree, 0),
      'add#': (_add_scale_degree, 1),
      'addb': (_add_scale_degree, -1),
      'no': (_subtract_scale_degree, 0),
      '#': (_alter_scale_degree, 1),
      'b': (_alter_scale_degree, -1)
  }

  # Regular expression for chord root.
  # Examples: 'C', 'G#', 'Ab', 'D######'
  _ROOT_PATTERN = r'[A-G](?:#*|b*)(?![#b])'

  # Regular expression for chord kind (abbreviated).
  # Examples: '', 'm7b5', 'min', '-13', '+', 'm(M7)', 'dim', '/o7', 'sus2'
  _CHORD_KIND_PATTERN = '|'.join(re.escape(abbrev)
                                 for abbrev in _CHORD_KINDS_BY_ABBREV)

  # Regular expression for scale degree modifications. (To keep the regex
  # simpler, parentheses are not required to match here, e.g. '(#9', 'add2)',
  # '(b5(#9)', and 'no5)(b9' will all match.)
  # Examples: '#9', 'add6add9', 'no5(b9)', '(add2b5no3)', '(no5)(b9)'
  _MODIFICATIONS_PATTERN = r'(?:\(?(?:%s)[0-9]+\)?)*' % '|'.join(
      re.escape(mod) for mod in _DEGREE_MODIFICATIONS)

  # Regular expression for chord bass.
  # Examples: '', '/C', '/Bb', '/F##', '/Dbbbb'
  _BASS_PATTERN = '|/%s' % _ROOT_PATTERN

  # Regular expression for full chord symbol.
  _CHORD_SYMBOL_PATTERN = ''.join('(%s)' % pattern for pattern in [
      _ROOT_PATTERN,            # root pitch class
      _CHORD_KIND_PATTERN,      # chord kind
      _MODIFICATIONS_PATTERN,   # scale degree modifications
      _BASS_PATTERN]) + '$'     # bass pitch class
  _CHORD_SYMBOL_REGEX = re.compile(_CHORD_SYMBOL_PATTERN)

  # Regular expression for a single pitch class.
  # Examples: 'C', 'G#', 'Ab', 'D######'
  _PITCH_CLASS_PATTERN = r'([A-G])(#*|b*)$'
  _PITCH_CLASS_REGEX = re.compile(_PITCH_CLASS_PATTERN)

  # Regular expression for a single scale degree.
  # Examples: '1', '7', 'b3', '#5', 'bb7', '13'
  _SCALE_DEGREE_PATTERN = r'(#*|b*)([0-9]+)$'
  _SCALE_DEGREE_REGEX = re.compile(_SCALE_DEGREE_PATTERN)

  # Regular expression for a single scale degree modification. (To keep the
  # regex simpler, parentheses are not required to match here, so open or
  # closing paren could be missing, e.g. '(#9' and 'add2)' will both match.)
  # Examples: '#9', 'add6', 'no5', '(b5)', '(add9)'
  _MODIFICATION_PATTERN = r'\(?(%s)([0-9]+)\)?' % '|'.join(
      re.escape(mod) for mod in _DEGREE_MODIFICATIONS)
  _MODIFICATION_REGEX = re.compile(_MODIFICATION_PATTERN)

  def _parse_pitch_class(self, pitch_class_str):
    """Parse pitch class from string, returning scale step and alteration."""
    match = re.match(self._PITCH_CLASS_REGEX, pitch_class_str)
    step, alter = match.groups()
    return step, len(alter) * (1 if '#' in alter else -1)

  def _parse_root(self, root_str):
    """Parse chord root from string."""
    return self._parse_pitch_class(root_str)

  def _parse_degree(self, degree_str):
    """Parse scale degree from string (from internal kind representation)."""
    match = self._SCALE_DEGREE_REGEX.match(degree_str)
    alter, degree = match.groups()
    return int(degree), len(alter) * (1 if '#' in alter else -1)

  def _parse_kind(self, kind_str):
    """Parse chord kind from string, returning a scale degree dictionary."""
    degrees = self._CHORD_KINDS_BY_ABBREV[kind_str]
    # Here we make the assumption that each scale degree can be present in a
    # chord at most once. This is not generally true, as e.g. a chord could
    # contain both b9 and #9.
    return dict(self._parse_degree(degree_str) for degree_str in degrees)

  def _parse_modifications(self, modifications_str):
    """Parse scale degree modifications from string.

    This returns a list of function-degree-alteration triples. The function,
    when applied to the list of scale degrees, the degree to modify, and the
    alteration, performs the modification.

    Args:
      modifications_str: A string containing the scale degree modifications to
          apply to a chord, in standard chord symbol format.

    Returns:
      A Python list of scale degree modification tuples, each of which contains
      a) a function that applies the modification, b) the integer scale degree
      to which to apply the modifications, and c) the number of semitones in the
      modification.
    """
    modifications = []
    while modifications_str:
      match = self._MODIFICATION_REGEX.match(modifications_str)
      type_str, degree_str = match.groups()
      mod_fn, alter = self._DEGREE_MODIFICATIONS[type_str]
      modifications.append((mod_fn, int(degree_str), alter))
      modifications_str = modifications_str[match.end():]
      assert match.end() > 0
    return modifications

  def _parse_bass(self, bass_str):
    """Parse bass, returning scale step and alteration or None if no bass."""
    if bass_str:
      return self._parse_pitch_class(bass_str[1:])
    else:
      return None

  def _apply_modifications(self, degrees, modifications):
    """Apply scale degree modifications to a scale degree dictionary."""
    for mod_fn, degree, alter in modifications:
      mod_fn(self, degrees, degree, alter)

  def _split_chord_symbol(self, figure):
    """Split a chord symbol into root, kind, degree modifications, and bass."""
    match = self._CHORD_SYMBOL_REGEX.match(figure)
    if not match:
      raise ChordSymbolException('Unable to parse chord symbol: %s' % figure)
    root_str, kind_str, modifications_str, bass_str = match.groups()
    return root_str, kind_str, modifications_str, bass_str

  def _parse_chord_symbol(self, figure):
    """Parse a chord symbol string.

    This converts the chord symbol string to a tuple representation with the
    following components:

      Root: A tuple containing scale step and alteration.
      Degrees: A dictionary where the keys are integer scale degrees, and values
          are integer alterations. For example, if 9 -> -1 is in the dictionary,
          the chord contains a b9.
      Bass: A tuple containins scale step and alteration. If bass is
          unspecified, the chord root is used.

    Args:
      figure: A chord symbol figure string.

    Returns:
      A tuple containing the chord root pitch class, scale degrees, and
      bass pitch class.
    """
    root_str, kind_str, modifications_str, bass_str = self._split_chord_symbol(
        figure)

    root = self._parse_root(root_str)
    degrees = self._parse_kind(kind_str)
    modifications = self._parse_modifications(modifications_str)
    bass = self._parse_bass(bass_str)

    # Apply scale degree modifications.
    self._apply_modifications(degrees, modifications)

    return root, degrees, bass or root

  def _transpose_pitch_class(self, step, alter, transpose_amount):
    """Transposes a chord symbol figure string by the given amount."""
    transpose_amount %= 12

    # Transpose up as many steps as we can.
    while transpose_amount >= self._STEPS_ABOVE[step]:
      transpose_amount -= self._STEPS_ABOVE[step]
      step = chr(ord('A') + (ord(step) - ord('A') + 1) % 7)

    if transpose_amount > 0:
      if alter >= 0:
        # Transpose up one more step and remove sharps (or add flats).
        alter -= self._STEPS_ABOVE[step] - transpose_amount
        step = chr(ord('A') + (ord(step) - ord('A') + 1) % 7)
      else:
        # Remove flats.
        alter += transpose_amount

    return step, alter

  def _pitch_class_to_string(self, step, alter):
    """Convert a pitch class scale step and alteration to string."""
    return step + abs(alter) * ('#' if alter >= 0 else 'b')

  def _pitch_class_to_midi(self, step, alter):
    """Convert a pitch class scale step and alteration to MIDI note."""
    return (self._STEPS_MIDI[step] + alter) % 12

  def transpose_chord_symbol(self, figure, transpose_amount):
    """Transposes a chord symbol figure string by the given amount."""
    # Split chord symbol into root, kind, modifications, and bass.
    root_str, kind_str, modifications_str, bass_str = self._split_chord_symbol(
        figure)

    # Parse and transpose the root.
    root_step, root_alter = self._parse_root(root_str)
    transposed_root_step, transposed_root_alter = self._transpose_pitch_class(
        root_step, root_alter, transpose_amount)
    transposed_root_str = self._pitch_class_to_string(
        transposed_root_step, transposed_root_alter)

    # Parse bass.
    bass = self._parse_bass(bass_str)

    if bass:
      # Bass exists, transpose it.
      bass_step, bass_alter = bass  # pylint: disable=unpacking-non-sequence
      transposed_bass_step, transposed_bass_alter = self._transpose_pitch_class(
          bass_step, bass_alter, transpose_amount)
      transposed_bass_str = '/' + self._pitch_class_to_string(
          transposed_bass_step, transposed_bass_alter)
    else:
      # No bass.
      transposed_bass_str = bass_str

    return '%s%s%s%s' % (transposed_root_str, kind_str, modifications_str,
                         transposed_bass_str)

  def chord_symbol_pitches(self, figure):
    """Return the pitch classes contained in a chord."""
    root, degrees, _ = self._parse_chord_symbol(figure)
    root_step, root_alter = root
    root_pitch = self._pitch_class_to_midi(root_step, root_alter)
    normalized_degrees = [((degree - 1) % 7 + 1, alter)
                          for degree, alter in degrees.items()]
    return [(root_pitch + self._DEGREE_OFFSETS[degree] + alter) % 12
            for degree, alter in normalized_degrees]

  def chord_symbol_root(self, figure):
    """Return the root pitch class of a chord."""
    root_str, _, _, _ = self._split_chord_symbol(figure)
    root_step, root_alter = self._parse_root(root_str)
    return self._pitch_class_to_midi(root_step, root_alter)

  def chord_symbol_bass(self, figure):
    """Return the bass pitch class of a chord."""
    root_str, _, _, bass_str = self._split_chord_symbol(figure)
    bass = self._parse_bass(bass_str)
    if bass:
      bass_step, bass_alter = bass  # pylint: disable=unpacking-non-sequence
    else:
      # Bass is the same as root.
      bass_step, bass_alter = self._parse_root(root_str)
    return self._pitch_class_to_midi(bass_step, bass_alter)

  def chord_symbol_quality(self, figure):
    """Return the quality (major, minor, diminished, augmented) of a chord."""
    _, degrees, _ = self._parse_chord_symbol(figure)
    if 1 not in degrees or 3 not in degrees or 5 not in degrees:
      return CHORD_QUALITY_OTHER
    triad = degrees[1], degrees[3], degrees[5]
    if triad == (0, 0, 0):
      return CHORD_QUALITY_MAJOR
    elif triad == (0, -1, 0):
      return CHORD_QUALITY_MINOR
    elif triad == (0, 0, 1):
      return CHORD_QUALITY_AUGMENTED
    elif triad == (0, -1, -1):
      return CHORD_QUALITY_DIMINISHED
    else:
      return CHORD_QUALITY_OTHER
