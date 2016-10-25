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
"""Tests for chords_lib."""

# internal imports
import tensorflow as tf

from magenta.music import chord_symbols_lib
from magenta.music import chords_lib
from magenta.music import constants
from magenta.music import melodies_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib

NO_CHORD = constants.NO_CHORD


class ChordsLibTest(tf.test.TestCase):

  def setUp(self):
    self.quantized_sequence = sequences_lib.QuantizedSequence()
    self.quantized_sequence.qpm = 60.0
    self.quantized_sequence.steps_per_quarter = 4

  def testTranspose(self):
    # Transpose ChordProgression with basic triads.
    events = ['Cm', 'F', 'B-', 'E-']
    chords = chords_lib.ChordProgression(events)
    chords.transpose(transpose_amount=7)
    expected = ['Gm', 'C', 'F', 'B-']
    self.assertEqual(expected, list(chords))

    # Transpose ChordProgression with more complex chords.
    events = ['Esus2', 'B13', 'A7/B', 'F#dim']
    chords = chords_lib.ChordProgression(events)
    chords.transpose(transpose_amount=-2)
    expected = ['Dsus2', 'A13', 'G7/A', 'Edim']
    self.assertEqual(expected, list(chords))

    # Transpose ChordProgression containing NO_CHORD.
    events = ['C', 'B-', NO_CHORD, 'F', 'C']
    chords = chords_lib.ChordProgression(events)
    chords.transpose(transpose_amount=4)
    expected = ['E', 'D', NO_CHORD, 'A', 'E']
    self.assertEqual(expected, list(chords))

  def testTransposeUnknownChordSymbol(self):
    # Attempt to transpose ChordProgression with unknown chord symbol.
    events = ['Cm', 'G7', 'P#13', 'F']
    chords = chords_lib.ChordProgression(events)
    with self.assertRaises(chord_symbols_lib.ChordSymbolException):
      chords.transpose(transpose_amount=-4)

  def testFromQuantizedSequence(self):
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence,
        [('Am', 4), ('D7', 8), ('G13', 12), ('Csus', 14)])
    chords = chords_lib.ChordProgression()
    chords.from_quantized_sequence(
        self.quantized_sequence, start_step=0, end_step=16)
    expected = [NO_CHORD, NO_CHORD, NO_CHORD, NO_CHORD,
                'Am', 'Am', 'Am', 'Am', 'D7', 'D7', 'D7', 'D7',
                'G13', 'G13', 'Csus', 'Csus']
    self.assertEqual(expected, list(chords))

  def testFromQuantizedSequenceWithinSingleChord(self):
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence, [('F', 0), ('Gm', 8)])
    chords = chords_lib.ChordProgression()
    chords.from_quantized_sequence(
        self.quantized_sequence, start_step=4, end_step=6)
    expected = ['F'] * 2
    self.assertEqual(expected, list(chords))

  def testFromQuantizedSequenceWithNoChords(self):
    chords = chords_lib.ChordProgression()
    chords.from_quantized_sequence(
        self.quantized_sequence, start_step=0, end_step=16)
    expected = [NO_CHORD] * 16
    self.assertEqual(expected, list(chords))

  def testFromQuantizedSequenceWithCoincidentChords(self):
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence,
        [('Am', 4), ('D7', 8), ('G13', 12), ('Csus', 12)])
    chords = chords_lib.ChordProgression()
    with self.assertRaises(chords_lib.CoincidentChordsException):
      chords.from_quantized_sequence(
          self.quantized_sequence, start_step=0, end_step=16)

  def testExtractChords(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence, [('C', 2), ('G7', 6), ('F', 8)])
    self.quantized_sequence.total_steps = 10
    chord_progressions, _ = chords_lib.extract_chords(self.quantized_sequence)
    expected = [[NO_CHORD, NO_CHORD, 'C', 'C', 'C', 'C', 'G7', 'G7', 'F', 'F']]
    self.assertEqual(expected, [list(chords) for chords in chord_progressions])

  def testExtractChordsAllTranspositions(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence, [('C', 1)])
    self.quantized_sequence.total_steps = 2
    chord_progressions, _ = chords_lib.extract_chords(self.quantized_sequence,
                                                      all_transpositions=True)
    expected = zip([NO_CHORD] * 12, ['G-', 'G', 'A-', 'A', 'B-', 'B',
                                     'C', 'D-', 'D', 'E-', 'E', 'F'])
    self.assertEqual(expected, [tuple(chords) for chords in chord_progressions])

  def testExtractChordsForMelodies(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence,
        [('C', 2), ('G7', 6), ('Cmaj7', 33)])
    melodies, _ = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, _ = chords_lib.extract_chords_for_melodies(
        self.quantized_sequence, melodies)
    expected = [[NO_CHORD, NO_CHORD, 'C', 'C', 'C', 'C',
                 'G7', 'G7', 'G7', 'G7', 'G7'],
                [NO_CHORD, NO_CHORD, 'C', 'C', 'C', 'C', 'G7', 'G7'],
                ['G7', 'Cmaj7', 'Cmaj7', 'Cmaj7', 'Cmaj7']]
    self.assertEqual(expected, [list(chords) for chords in chord_progressions])

  def testExtractChordsForMelodiesCoincidentChords(self):
    self.quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 11)])
    testing_lib.add_quantized_track_to_sequence(
        self.quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8),
         (50, 100, 33, 37), (52, 100, 34, 37)])
    testing_lib.add_quantized_chords_to_sequence(
        self.quantized_sequence,
        [('C', 2), ('G7', 6), ('E13', 8), ('Cmaj7', 8)])
    melodies, _ = melodies_lib.extract_melodies(
        self.quantized_sequence, min_bars=1, gap_bars=2, min_unique_pitches=2,
        ignore_polyphonic_notes=True)
    chord_progressions, stats = chords_lib.extract_chords_for_melodies(
        self.quantized_sequence, melodies)
    expected = [[NO_CHORD, NO_CHORD, 'C', 'C', 'C', 'C', 'G7', 'G7'],
                ['Cmaj7', 'Cmaj7', 'Cmaj7', 'Cmaj7', 'Cmaj7']]
    stats_dict = dict([(stat.name, stat) for stat in stats])
    self.assertIsNone(chord_progressions[0])
    self.assertEqual(expected,
                     [list(chords) for chords in chord_progressions[1:]])
    self.assertEqual(stats_dict['coincident_chords'].count, 1)

  def testToSequence(self):
    chords = chords_lib.ChordProgression(
        [NO_CHORD, 'C7', 'C7', 'C7', 'C7', 'Am7b5', 'F6', 'F6', NO_CHORD])
    sequence = chords.to_sequence(sequence_start_time=2, qpm=60.0)

    self.assertProtoEquals(
        'ticks_per_quarter: 220 '
        'tempos < qpm: 60.0 > '
        'text_annotations < '
        '  text: "C7" time: 2.25 annotation_type: CHORD_SYMBOL '
        '> '
        'text_annotations < '
        '  text: "Am7b5" time: 3.25 annotation_type: CHORD_SYMBOL '
        '> '
        'text_annotations < '
        '  text: "F6" time: 3.5 annotation_type: CHORD_SYMBOL '
        '> '
        'text_annotations < '
        '  text: "N.C." time: 4.0 annotation_type: CHORD_SYMBOL '
        '> ',
        sequence)


class MajorMinorEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.encoder_decoder = chords_lib.MajorMinorEncoderDecoder()

  def testEncodeNoChord(self):
    index = self.encoder_decoder.encode_chord(NO_CHORD)
    self.assertEquals(0, index)

  def testEncodeChord(self):
    # major triad
    index = self.encoder_decoder.encode_chord('C')
    self.assertEquals(1, index)

    # minor triad
    index = self.encoder_decoder.encode_chord('Cm')
    self.assertEquals(13, index)

    # dominant 7th
    index = self.encoder_decoder.encode_chord('F7')
    self.assertEquals(6, index)

    # minor 9th
    index = self.encoder_decoder.encode_chord('A-m9')
    self.assertEquals(21, index)

  def testEncodeThirdlessChord(self):
    # suspended chord
    with self.assertRaises(chords_lib.ChordEncodingException):
      self.encoder_decoder.encode_chord('Gsus4')

    # power chord
    with self.assertRaises(chords_lib.ChordEncodingException):
      self.encoder_decoder.encode_chord('B-5')

  def testDecodeNoChord(self):
    figure = self.encoder_decoder.decode_chord(0)
    self.assertEquals(NO_CHORD, figure)

  def testDecodeChord(self):
    # major chord
    figure = self.encoder_decoder.decode_chord(3)
    self.assertEquals('D', figure)

    # minor chord
    figure = self.encoder_decoder.decode_chord(17)
    self.assertEquals('Em', figure)


if __name__ == '__main__':
  tf.test.main()
