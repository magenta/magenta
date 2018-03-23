"""Tests for key signature inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from absl.testing import absltest

from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.score.elements import clef
from magenta.models.omr.score.elements import key_signature


class KeySignatureTest(absltest.TestCase):

  def testEmpty_noNextAccidental(self):
    self.assertEqual(
        key_signature.KeySignature(clef.TrebleClef()).get_next_accidental(),
        (None, None))

  def testGMajor(self):
    sig = key_signature.KeySignature(clef.TrebleClef())
    self.assertTrue(sig.try_put(+4, musicscore_pb2.Glyph.SHARP))  # F#
    self.assertEqual(
        sig.get_next_accidental(),
        (+1, musicscore_pb2.Glyph.SHARP))  # C#

  def testGMajor_bassClef(self):
    sig = key_signature.KeySignature(clef.BassClef())
    self.assertTrue(sig.try_put(+2, musicscore_pb2.Glyph.SHARP))  # F#
    self.assertEqual(
        sig.get_next_accidental(),
        (-1, musicscore_pb2.Glyph.SHARP))  # C#

  def testBMajor(self):
    sig = key_signature.KeySignature(clef.TrebleClef())
    self.assertTrue(sig.try_put(+4, musicscore_pb2.Glyph.SHARP))  # F#
    self.assertTrue(sig.try_put(+1, musicscore_pb2.Glyph.SHARP))  # C#
    self.assertTrue(sig.try_put(+5, musicscore_pb2.Glyph.SHARP))  # G#
    self.assertEqual(
        sig.get_next_accidental(),
        (+2, musicscore_pb2.Glyph.SHARP))  # D#

  def testEFlatMajor(self):
    sig = key_signature.KeySignature(clef.TrebleClef())
    self.assertTrue(sig.try_put(0, musicscore_pb2.Glyph.FLAT))  # Bb
    self.assertTrue(sig.try_put(+3, musicscore_pb2.Glyph.FLAT))  # Eb
    self.assertTrue(sig.try_put(-1, musicscore_pb2.Glyph.FLAT))  # Ab
    self.assertEqual(
        sig.get_next_accidental(),
        (+2, musicscore_pb2.Glyph.FLAT))  # Db

  def testEFlatMajor_bassClef(self):
    sig = key_signature.KeySignature(clef.BassClef())
    self.assertTrue(sig.try_put(-2, musicscore_pb2.Glyph.FLAT))  # Bb
    self.assertTrue(sig.try_put(+1, musicscore_pb2.Glyph.FLAT))  # Eb
    self.assertTrue(sig.try_put(-3, musicscore_pb2.Glyph.FLAT))  # Ab
    self.assertEqual(
        sig.get_next_accidental(),
        (0, musicscore_pb2.Glyph.FLAT))  # Db

  def testCFlatMajor_noMoreAccidentals(self):
    sig = key_signature.KeySignature(clef.TrebleClef())
    self.assertTrue(sig.try_put(0, musicscore_pb2.Glyph.FLAT))  # Bb
    self.assertNotEqual(sig.get_next_accidental(), (None, None))
    self.assertTrue(sig.try_put(+3, musicscore_pb2.Glyph.FLAT))  # Eb
    self.assertNotEqual(sig.get_next_accidental(), (None, None))
    self.assertTrue(sig.try_put(-1, musicscore_pb2.Glyph.FLAT))  # Ab
    self.assertNotEqual(sig.get_next_accidental(), (None, None))
    self.assertTrue(sig.try_put(+2, musicscore_pb2.Glyph.FLAT))  # Db
    self.assertNotEqual(sig.get_next_accidental(), (None, None))
    self.assertTrue(sig.try_put(-2, musicscore_pb2.Glyph.FLAT))  # Gb
    self.assertNotEqual(sig.get_next_accidental(), (None, None))
    self.assertTrue(sig.try_put(+1, musicscore_pb2.Glyph.FLAT))  # Cb
    self.assertNotEqual(sig.get_next_accidental(), (None, None))
    self.assertTrue(sig.try_put(-3, musicscore_pb2.Glyph.FLAT))  # Fb
    # Already at Cb major, no more accidentals to add.
    self.assertEqual(sig.get_next_accidental(), (None, None))


if __name__ == '__main__':
  absltest.main()
