"""Tests for the clefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from absl.testing import absltest
import librosa

from magenta.models.omr.score.elements import clef


class ClefTest(absltest.TestCase):

  def testTrebleClef(self):
    self.assertEqual(clef.TrebleClef().y_position_to_midi(-8),
                     librosa.note_to_midi('A3'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(-6),
                     librosa.note_to_midi('C4'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(0),
                     librosa.note_to_midi('B4'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(1),
                     librosa.note_to_midi('C5'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(3),
                     librosa.note_to_midi('E5'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(4),
                     librosa.note_to_midi('F5'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(14),
                     librosa.note_to_midi('B6'))

  def testBassClef(self):
    self.assertEqual(clef.BassClef().y_position_to_midi(-10),
                     librosa.note_to_midi('A1'))
    self.assertEqual(clef.BassClef().y_position_to_midi(-7),
                     librosa.note_to_midi('D2'))
    self.assertEqual(clef.BassClef().y_position_to_midi(-5),
                     librosa.note_to_midi('F2'))
    self.assertEqual(clef.BassClef().y_position_to_midi(-1),
                     librosa.note_to_midi('C3'))
    self.assertEqual(clef.BassClef().y_position_to_midi(0),
                     librosa.note_to_midi('D3'))
    self.assertEqual(clef.BassClef().y_position_to_midi(6),
                     librosa.note_to_midi('C4'))
    self.assertEqual(clef.BassClef().y_position_to_midi(8),
                     librosa.note_to_midi('E4'))


if __name__ == '__main__':
  absltest.main()
