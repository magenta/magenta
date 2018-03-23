"""The score state which is not persisted between measures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import enum

from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.score.elements import key_signature as key_signature_module
from magenta.protobuf import music_pb2

ACCIDENTAL_PITCH_SHIFT_ = {
    musicscore_pb2.Glyph.FLAT: -1,
    musicscore_pb2.Glyph.NATURAL: 0,
    musicscore_pb2.Glyph.NONE: 0,
    musicscore_pb2.Glyph.SHARP: 1,
}


class _KeySignatureState(enum.Enum):

  KEY_SIGNATURE = 1
  ACCIDENTALS = 2


class MeasureState(object):
  """State of a single measure of a staff.

  Attributes:
    clef: The current clef.
    key_signature: The current `KeySignature`.
    chords: A map from stem (tuple `((x0, y0), (x1, y1))`) to the first note
        that was read and is attached to the stem. Subsequent notes attached to
        the same stem will read their start and end time from the first note.
    time: The current time in the measure. Absolute time relative to the start
        of the score. float.
  """

  def __init__(self, start_time, clef, key_signature=None):
    """Initializes a new measure.

    Args:
      start_time: The start time (in quarter notes) of the measure.
      clef: A `Clef`.
      key_signature: The previously detected key signature (optional). If
          present, do not detect a key signature in this measure. This should be
          taken from the previously measure on this staff if this is not the
          first measure. It should not be propagated from one staff to the next,
          because we expect the key signature to be repeated on each staff and
          we will re-detect it.
    """
    self.time = start_time
    self.clef = clef
    self.key_signature = (
        key_signature or key_signature_module.KeySignature(clef))
    self._accidentals = key_signature_module.Accidentals(clef)
    self._key_signature_state = (
        _KeySignatureState.ACCIDENTALS
        if key_signature else _KeySignatureState.KEY_SIGNATURE)
    self.chords = {}

  def new_measure(self, start_time):
    """Constructs a new MeasureState for the next measure.

    Args:
      start_time: The start time of the new measure.

    Returns:
      A new MeasureState object.
    """
    return MeasureState(
        start_time,
        clef=self.clef,
        key_signature=copy.deepcopy(self.key_signature))

  def set_accidental(self, y_position, accidental):
    """Adds a glyph to the key signature or accidentals.

    Args:
      y_position: The position of the accidental.
      accidental: The accidental value.
    """
    if self._key_signature_state == _KeySignatureState.KEY_SIGNATURE:
      if self.key_signature.try_put(y_position, accidental):
        return
      self._key_signature_state = _KeySignatureState.ACCIDENTALS
    self._accidentals.put(y_position, accidental)

  def get_note(self, glyph):
    """Converts a Glyph to a Note.

    Gets the note timing from an existing chord if available, or increments the
    current measure time otherwise.

    Args:
      glyph: A Glyph message. Type must be one of NOTEHEAD_*.

    Returns:
      A Note message.
    """
    accidental = self._accidentals.get_accidental_for_position(glyph.y_position)
    if accidental == musicscore_pb2.Glyph.NONE:
      accidental = self.key_signature.get_accidental_for_position(
          glyph.y_position)
    pitch = (
        self.clef.y_position_to_midi(glyph.y_position) +
        ACCIDENTAL_PITCH_SHIFT_[accidental])
    first_note_in_chord = None
    if glyph.HasField('stem'):
      # Try to get the timing from another note in the same chord.
      stem = ((glyph.stem.start.x, glyph.stem.start.y), (glyph.stem.end.x,
                                                         glyph.stem.end.y))
      if stem in self.chords:
        first_note_in_chord = self.chords[stem]
    else:
      stem = None

    if first_note_in_chord:
      start_time, end_time = (first_note_in_chord.start_time,
                              first_note_in_chord.end_time)
    else:
      # TODO(ringwalt): Check all note durations, not just the first seen in a
      # chord, and use the median detected duration.
      duration = _get_note_duration(glyph)
      start_time, end_time = self.time, self.time + duration
      self.time += duration
    note = music_pb2.NoteSequence.Note(
        pitch=pitch, start_time=start_time, end_time=end_time)
    if stem:
      self.chords[stem] = note
    return note

  def set_clef(self, clef):
    """Sets the clef, and resets the key signature if necessary."""
    if clef != self.clef:
      self._key_signature_state = _KeySignatureState.KEY_SIGNATURE
      self.key_signature = key_signature_module.KeySignature(clef)
      self._accidentals = key_signature_module.Accidentals(clef)
    self.clef = clef

  def on_read_notehead(self):
    """Called after a notehead has been read.

    The key signature should occur before any noteheads in the measure. This
    causes subsequent accidental glyphs to be read as accidentals, and not part
    of the key signature.
    """
    self._key_signature_state = _KeySignatureState.ACCIDENTALS


def _get_note_duration(note):
  """Determines the duration of a notehead glyph.

  This depends on the glyph type, beams (which each halve the duration), and
  dots (which each add a fractional duration). In the future, notes may be
  recognized as a tuplet, which will result in a Fraction duration. For now, the
  duration is a float, because the denominator is always a sum of powers of two.

  Args:
    note: A `Glyph` of a notehead type.

  Returns:
    The float duration of the note, in quarter notes.

  Raises:
    ValueError: If `note` is not a notehead type.
  """
  if note.type == musicscore_pb2.Glyph.NOTEHEAD_FILLED:
    # Quarter note: 2.0 ** 0 == 1
    # Each beam halves the note duration.
    duration = 2.0**-len(note.beam)
  elif note.type == musicscore_pb2.Glyph.NOTEHEAD_EMPTY:
    duration = 2.0
  elif note.type == musicscore_pb2.Glyph.NOTEHEAD_WHOLE:
    duration = 4.0
  else:
    raise ValueError('Expected a notehead, got: %s' % note)
  # The first dot adds half the original duration, and further dots add half the
  # value added by the previous dot.
  dot_value = duration / 2.
  for _ in note.dot:
    duration += dot_value
    dot_value /= 2.
  return duration
