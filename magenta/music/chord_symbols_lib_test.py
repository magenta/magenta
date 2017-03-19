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
"""Tests for chord_symbols_lib."""

# internal imports
import tensorflow as tf

from magenta.music import chord_symbols_lib

CHORD_QUALITY_MAJOR = chord_symbols_lib.CHORD_QUALITY_MAJOR
CHORD_QUALITY_MINOR = chord_symbols_lib.CHORD_QUALITY_MINOR
CHORD_QUALITY_AUGMENTED = chord_symbols_lib.CHORD_QUALITY_AUGMENTED
CHORD_QUALITY_DIMINISHED = chord_symbols_lib.CHORD_QUALITY_DIMINISHED
CHORD_QUALITY_OTHER = chord_symbols_lib.CHORD_QUALITY_OTHER


class ChordSymbolFunctionsTest(tf.test.TestCase):

  def setUp(self):
    self.chord_symbol_functions = chord_symbols_lib.ChordSymbolFunctions.get()

  def testTranspose(self):
    # Test basic triads.
    figure = self.chord_symbol_functions.transpose_chord_symbol('C', 2)
    self.assertEqual('D', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('Abm', -3)
    self.assertEqual('Fm', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('F#', 0)
    self.assertEqual('F#', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('Cbb', 6)
    self.assertEqual('Fb', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('C#', -5)
    self.assertEqual('G#', figure)

    # Test more complex chords.
    figure = self.chord_symbol_functions.transpose_chord_symbol('Co7', 7)
    self.assertEqual('Go7', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('D+', -3)
    self.assertEqual('B+', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('Fb9/Ab', 2)
    self.assertEqual('Gb9/Bb', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('A6/9', -7)
    self.assertEqual('D6/9', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('E7(add#9)', 0)
    self.assertEqual('E7(add#9)', figure)

  def testPitches(self):
    pitches = self.chord_symbol_functions.chord_symbol_pitches('Am')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([0, 4, 9]), pitch_classes)
    pitches = self.chord_symbol_functions.chord_symbol_pitches('D7b9')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([0, 2, 3, 6, 9]), pitch_classes)
    pitches = self.chord_symbol_functions.chord_symbol_pitches('F/o')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([3, 5, 8, 11]), pitch_classes)
    pitches = self.chord_symbol_functions.chord_symbol_pitches('C-(M7)')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([0, 3, 7, 11]), pitch_classes)
    pitches = self.chord_symbol_functions.chord_symbol_pitches('E##13')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([1, 3, 4, 6, 8, 10, 11]), pitch_classes)
    pitches = self.chord_symbol_functions.chord_symbol_pitches('G(add2)(#5)')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([3, 7, 9, 11]), pitch_classes)

  def testRoot(self):
    root = self.chord_symbol_functions.chord_symbol_root('Dm9')
    self.assertEqual(2, root)
    root = self.chord_symbol_functions.chord_symbol_root('E/G#')
    self.assertEqual(4, root)
    root = self.chord_symbol_functions.chord_symbol_root('Bsus2')
    self.assertEqual(11, root)
    root = self.chord_symbol_functions.chord_symbol_root('Abmaj7')
    self.assertEqual(8, root)
    root = self.chord_symbol_functions.chord_symbol_root('D##5(add6)')
    self.assertEqual(4, root)
    root = self.chord_symbol_functions.chord_symbol_root('F(b7)(#9)(b13)')
    self.assertEqual(5, root)

  def testBass(self):
    bass = self.chord_symbol_functions.chord_symbol_bass('Dm9')
    self.assertEqual(2, bass)
    bass = self.chord_symbol_functions.chord_symbol_bass('E/G#')
    self.assertEqual(8, bass)
    bass = self.chord_symbol_functions.chord_symbol_bass('Bsus2/A')
    self.assertEqual(9, bass)
    bass = self.chord_symbol_functions.chord_symbol_bass('Abm7/Cb')
    self.assertEqual(11, bass)
    bass = self.chord_symbol_functions.chord_symbol_bass('C#6/9/E#')
    self.assertEqual(5, bass)
    bass = self.chord_symbol_functions.chord_symbol_bass('G/o')
    self.assertEqual(7, bass)

  def testQuality(self):
    # Test major chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('B13')
    self.assertEqual(CHORD_QUALITY_MAJOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('E7#9')
    self.assertEqual(CHORD_QUALITY_MAJOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Fadd2/Eb')
    self.assertEqual(CHORD_QUALITY_MAJOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('C6/9/Bb')
    self.assertEqual(CHORD_QUALITY_MAJOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Gmaj13')
    self.assertEqual(CHORD_QUALITY_MAJOR, quality)

    # Test minor chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('C#-9')
    self.assertEqual(CHORD_QUALITY_MINOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Gm7/Bb')
    self.assertEqual(CHORD_QUALITY_MINOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Cbmmaj7')
    self.assertEqual(CHORD_QUALITY_MINOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('A-(M7)')
    self.assertEqual(CHORD_QUALITY_MINOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Bbmin')
    self.assertEqual(CHORD_QUALITY_MINOR, quality)

    # Test augmented chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('D+/A#')
    self.assertEqual(CHORD_QUALITY_AUGMENTED, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('A+')
    self.assertEqual(CHORD_QUALITY_AUGMENTED, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('G7(#5)')
    self.assertEqual(CHORD_QUALITY_AUGMENTED, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Faug(add2)')
    self.assertEqual(CHORD_QUALITY_AUGMENTED, quality)

    # Test diminished chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('Am7b5')
    self.assertEqual(CHORD_QUALITY_DIMINISHED, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Edim7')
    self.assertEqual(CHORD_QUALITY_DIMINISHED, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Bb/o')
    self.assertEqual(CHORD_QUALITY_DIMINISHED, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Fo')
    self.assertEqual(CHORD_QUALITY_DIMINISHED, quality)

    # Test other chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('G5')
    self.assertEqual(CHORD_QUALITY_OTHER, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Bbsus2')
    self.assertEqual(CHORD_QUALITY_OTHER, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Dsus')
    self.assertEqual(CHORD_QUALITY_OTHER, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('E(no3)')
    self.assertEqual(CHORD_QUALITY_OTHER, quality)


if __name__ == '__main__':
  tf.test.main()
