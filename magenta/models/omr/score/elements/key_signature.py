"""Music key signature inference.

The accidentals classes are Accidentals, which are reset for each new measure,
and KeySignature, which is persisted for a staff at a time (because it is
expected to be repeated on each new staff). The key signature must follow the
expected pattern, or subsequent accidentals will fail to be added to it, and
should be added to the Accidentals instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# internal imports
import librosa
import six
from six import moves

from magenta.models.omr.protobuf import musicscore_pb2
from magenta.music import constants

Glyph = musicscore_pb2.Glyph  # pylint: disable=invalid-name


class _BaseAccidentals(object):
  """Holds accidentals which are not part of the key signature."""

  def __init__(self, clef, accidentals=None):
    self.clef = clef
    self._accidentals = dict(accidentals or {})

  def _normalize_position(self, position):
    """No octave normalization.

    Accidentals only apply to the current octave, whereas `KeySignature`
    overrides this to make its accidentals octave-invariant.

    Args:
      position: The vertical staff y position.

    Returns:
      The normalized position.
    """
    return position

  def get_accidental_for_position(self, position):
    return self._accidentals.get(self._normalize_position(position), Glyph.NONE)


class Accidentals(_BaseAccidentals):
  """Simple map of staff y position to accidental value."""

  def __init__(self, clef):
    super(Accidentals, self).__init__(clef)

  def put(self, position, accidental):
    self._accidentals[position] = accidental


class KeySignature(_BaseAccidentals):
  """Music key signature.

  Tracks the expected order of accidentals in a key signature. If we detect that
  an accidental does not match the next expected accidental, it will be treated
  as a normal accidental and not part of the key signature.
  """

  def _normalize_position(self, position):
    """Normalize base notes by octave for the key signature.

    The key signature contains one accidental for one base notes, which applies
    to the same pitch class in all octaves.

    Args:
      position: The staff y position of the glyph.

    Returns:
      The base note normalized by octave. This causes an accidental in the key
          signature to apply to the same note in all octaves.
    """
    return position % len(constants.MAJOR_SCALE)

  def try_put(self, position, accidental):
    """Adds an accidental to the key signature if applicable.

    Args:
      position: The accidental glyph y position.
      accidental: The accidental glyph type.

    Returns:
      True if the accidental was successfully added to the key signature. False
        if the key signature would be invalid when adding the new accidental.
    """
    can_put = self._can_put(position, accidental)
    if can_put:
      self._accidentals[self._normalize_position(position)] = accidental
    return can_put

  def _can_put(self, position, accidental):
    if not self._accidentals:
      pitch_class = self.clef.y_position_to_midi(position) % 12
      return (accidental in _KEY_SIGNATURE_PITCH_CLASS_LIST and
              pitch_class == _KEY_SIGNATURE_PITCH_CLASS_LIST[accidental][0])
    return (position, accidental) == self.get_next_accidental()

  def get_next_accidental(self):
    """Predicts the next accidental which would be present in the key signature.

    Cannot predict the next accidental if the key signature is currently empty
    (C major), because the key could contain either sharps or flats.

    Returns:
      The expected y position of the next accidental if possible, or None.
      The expected accidental glyph type, or None.
    """
    # There must already be some accidentals, which are all sharps or all flats.
    # Get the base pitch class for each note that has an accidental.
    pitch_classes = [
        self.clef.y_position_to_midi(position) % _NUM_SEMITONES_PER_OCTAVE
        for position in self._accidentals.keys()
    ]
    # Determine the order of pitch classes (for either all sharps or all flats).
    values = set(self._accidentals.values())
    if len(values) == 1:
      full_key_sig = _KEY_SIGNATURE_PITCH_CLASS_LIST[six.next(iter(values))]
    else:
      # Key signature is empty. Don't know whether to predict a sharp or a flat.
      return None, None

    if len(pitch_classes) == len(full_key_sig):
      # No more accidentals to add.
      return None, None
    elif set(pitch_classes) == set(full_key_sig[:len(pitch_classes)]):
      # Use the next pitch class in the list.
      next_pitch_class = full_key_sig[len(pitch_classes)]
      accidental = six.next(iter(values))
      # The pitch class must match exactly one of the 7 y positions that are
      # allowed for this key signature.
      for y_position in _KEY_SIGNATURE_Y_POSITION_RANGES[self.clef.glyph,
                                                         accidental]:
        if (self.clef.y_position_to_midi(y_position) %
            _NUM_SEMITONES_PER_OCTAVE) == next_pitch_class:
          return y_position, accidental
      raise AssertionError('Failed to find the next accidental y position')
    else:
      # The current key signature is unrecognized.
      return None, None

  def get_type(self):
    """Returns whether this is a sharp, flat, or None (C major) signature."""
    return (six.next(iter(self._accidentals.values()))
            if self._accidentals else None)

  def __len__(self):
    """Returns the number of accidentals in the key signature."""
    return len(self._accidentals)


_NUM_SEMITONES_PER_OCTAVE = 12

# These constants are coincidentally equal.
_NUM_NOTES_IN_DIATONIC_SCALE = 7
_NUM_SEMITONES_IN_PERFECT_FIFTH = 7

# The consecutive base notes of a key signature are each separated by a fifth,
# or 7 semitones.
_CIRCLE_OF_FIFTHS = [
    (i * _NUM_SEMITONES_IN_PERFECT_FIFTH) % _NUM_SEMITONES_PER_OCTAVE
    for i in moves.range(_NUM_SEMITONES_PER_OCTAVE)
]


def _key_sig_pitch_classes(note_name, ascending_fifths):
  first_pitch_class = (
      librosa.note_to_midi(note_name + '0') % _NUM_SEMITONES_PER_OCTAVE)
  # Go through the circle of fifths in ascending or descending order.
  step = 1 if ascending_fifths else -1
  order = _CIRCLE_OF_FIFTHS[::step]
  # Get the start index for the key signature.
  first_pitch_class_ind = order.index(first_pitch_class)
  return list(
      itertools.islice(
          # Create a cycle of the order. We may loop around, e.g. from F back to
          # C.
          itertools.cycle(order),
          # Take the 7 pitch classes from the cycle.
          first_pitch_class_ind,
          first_pitch_class_ind + _NUM_NOTES_IN_DIATONIC_SCALE))


_KEY_SIGNATURE_PITCH_CLASS_LIST = {
    # The sharp key signature starts with F#, and each subsequent note ascends
    # by a fifth.
    Glyph.SHARP:
        _key_sig_pitch_classes('F', ascending_fifths=True),
    # The flat key signature starts with Bb, and each subsequent note descends
    # by a fifth.
    Glyph.FLAT:
        _key_sig_pitch_classes('B', ascending_fifths=False),
}

# Maps the clef and type of accidentals in the key signature to the range of y
# positions where the key signature is shown.
_KEY_SIGNATURE_Y_POSITION_RANGES = {
    (Glyph.CLEF_TREBLE, Glyph.SHARP): range(-1, 6),  # A#4 to G#5
    (Glyph.CLEF_TREBLE, Glyph.FLAT): range(-3, 4),  # Fb4 to Eb5
    (Glyph.CLEF_BASS, Glyph.SHARP): range(-3, 4),  # A#2 to G#3
    (Glyph.CLEF_BASS, Glyph.FLAT): range(-5, 2),  # Fb2 to Eb3
}
