"""Clef logic for OMR.

A Clef object maps y positions on the staff to the MIDI pitch of the natural
note at the y position.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import librosa

from magenta.models.omr.protobuf import musicscore_pb2
from magenta.music import constants


class Clef(object):
  """Represents a clef which maps y positions to MIDI notes.

  Attributes:
    center_line_pitch: A _ScalePitch representing the center line (3rd line of
        the staff).
  """
  center_line_pitch = None

  def y_position_to_midi(self, y_position):
    return (self.center_line_pitch + y_position).midi


class TrebleClef(Clef):
  """Represents a treble clef."""

  def __init__(self):
    self.center_line_pitch = _ScalePitch(constants.MAJOR_SCALE,
                                         librosa.note_to_midi('B4'))
    self.glyph = musicscore_pb2.Glyph.CLEF_TREBLE


class BassClef(Clef):
  """Represents a bass clef."""

  def __init__(self):
    self.center_line_pitch = _ScalePitch(constants.MAJOR_SCALE,
                                         librosa.note_to_midi('D3'))
    self.glyph = musicscore_pb2.Glyph.CLEF_BASS


class _ScalePitch(object):
  """A natural note which can be offset to get another note.

  Attributes:
    scale: The scale which this pitch is based on. A list of MIDI pitch values
        spanning one octave.
    index: The index of the pitch's pitch class within the scale.
    octave: The index of the octave that the pitch is in, relative to the octave
        spanning the scale notes.
  """

  def __init__(self, scale, midi):
    self.scale = scale
    self.index = scale.index(midi % constants.NOTES_PER_OCTAVE)
    self.octave = (midi - scale[0]) // 12

  @property
  def midi(self):
    """The MIDI value for the pitch."""
    return self.scale[self.index] + constants.NOTES_PER_OCTAVE * self.octave

  @property
  def pitch_index(self):
    """The index of the pitch in the C major scale."""
    return self.index + len(self.scale) * self.octave

  def __add__(self, interval):
    """Returns the natural note `interval` away on `self.scale`."""
    pitch = _ScalePitch(self.scale, self.midi)
    pitch_index = self.pitch_index + interval
    pitch.index = pitch_index % len(self.scale)
    pitch.octave = pitch_index // len(self.scale)
    return pitch
