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
    figure = self.chord_symbol_functions.transpose_chord_symbol('A-m', -3)
    self.assertEqual('Fm', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('F#', 0)
    self.assertEqual('F#', figure)

    # Test more complex chords.
    figure = self.chord_symbol_functions.transpose_chord_symbol('Co7', 7)
    self.assertEqual('Go7', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('D+', -3)
    self.assertEqual('B+', figure)
    figure = self.chord_symbol_functions.transpose_chord_symbol('F-9/A-', 2)
    self.assertEqual('G-9/B-', figure)

  def testMidiPitches(self):
    # Check that pitch classes are correct.
    pitches = self.chord_symbol_functions.chord_symbol_midi_pitches('Am')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([0, 4, 9]), pitch_classes)
    pitches = self.chord_symbol_functions.chord_symbol_midi_pitches('D7b9')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([0, 2, 3, 6, 9]), pitch_classes)
    pitches = self.chord_symbol_functions.chord_symbol_midi_pitches('Fm7b5')
    pitch_classes = set(pitch % 12 for pitch in pitches)
    self.assertEqual(set([3, 5, 8, 11]), pitch_classes)

    # Check that bass notes are correct.
    pitches = self.chord_symbol_functions.chord_symbol_midi_pitches('B-7')
    bass_pitch_class = min(pitches) % 12
    self.assertEqual(10, bass_pitch_class)
    pitches = self.chord_symbol_functions.chord_symbol_midi_pitches('A/G')
    bass_pitch_class = min(pitches) % 12
    self.assertEqual(7, bass_pitch_class)
    pitches = self.chord_symbol_functions.chord_symbol_midi_pitches('F#dim7')
    bass_pitch_class = min(pitches) % 12
    self.assertEqual(6, bass_pitch_class)

  def testRoot(self):
    root = self.chord_symbol_functions.chord_symbol_root('Dm9')
    self.assertEqual(2, root)
    root = self.chord_symbol_functions.chord_symbol_root('E/G#')
    self.assertEqual(4, root)
    root = self.chord_symbol_functions.chord_symbol_root('Bsus2')
    self.assertEqual(11, root)
    root = self.chord_symbol_functions.chord_symbol_root('A-maj7')
    self.assertEqual(8, root)

  def testQuality(self):
    # Test major chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('B13')
    self.assertEqual(CHORD_QUALITY_MAJOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('E7#9')
    self.assertEqual(CHORD_QUALITY_MAJOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Fadd2/E-')
    self.assertEqual(CHORD_QUALITY_MAJOR, quality)

    # Test minor chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('C#m9')
    self.assertEqual(CHORD_QUALITY_MINOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Gm7/B-')
    self.assertEqual(CHORD_QUALITY_MINOR, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('C-mM7')
    self.assertEqual(CHORD_QUALITY_MINOR, quality)

    # Test augmented chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('D+/A#')
    self.assertEqual(CHORD_QUALITY_AUGMENTED, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Aaug')
    self.assertEqual(CHORD_QUALITY_AUGMENTED, quality)

    # Test diminished chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('Am7b5')
    self.assertEqual(CHORD_QUALITY_DIMINISHED, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Edim7')
    self.assertEqual(CHORD_QUALITY_DIMINISHED, quality)

    # Test other chords.
    quality = self.chord_symbol_functions.chord_symbol_quality('G5')
    self.assertEqual(CHORD_QUALITY_OTHER, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('B-sus2')
    self.assertEqual(CHORD_QUALITY_OTHER, quality)
    quality = self.chord_symbol_functions.chord_symbol_quality('Dsus')
    self.assertEqual(CHORD_QUALITY_OTHER, quality)


if __name__ == '__main__':
  tf.test.main()
