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
"""Tests for chords_encoder_decoder."""

import tensorflow as tf

from magenta.music import chords_encoder_decoder
from magenta.music import constants

NO_CHORD = constants.NO_CHORD


class MajorMinorChordOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = chords_encoder_decoder.MajorMinorChordOneHotEncoding()

  def testEncodeNoChord(self):
    index = self.enc.encode_event(NO_CHORD)
    self.assertEqual(0, index)

  def testEncodeChord(self):
    # major triad
    index = self.enc.encode_event('C')
    self.assertEqual(1, index)

    # minor triad
    index = self.enc.encode_event('Cm')
    self.assertEqual(13, index)

    # dominant 7th
    index = self.enc.encode_event('F7')
    self.assertEqual(6, index)

    # minor 9th
    index = self.enc.encode_event('Abm9')
    self.assertEqual(21, index)

  def testEncodeThirdlessChord(self):
    # suspended chord
    with self.assertRaises(chords_encoder_decoder.ChordEncodingException):
      self.enc.encode_event('Gsus4')

    # power chord
    with self.assertRaises(chords_encoder_decoder.ChordEncodingException):
      self.enc.encode_event('Bb5')

  def testDecodeNoChord(self):
    figure = self.enc.decode_event(0)
    self.assertEqual(NO_CHORD, figure)

  def testDecodeChord(self):
    # major chord
    figure = self.enc.decode_event(3)
    self.assertEqual('D', figure)

    # minor chord
    figure = self.enc.decode_event(17)
    self.assertEqual('Em', figure)


class TriadChordOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = chords_encoder_decoder.TriadChordOneHotEncoding()

  def testEncodeNoChord(self):
    index = self.enc.encode_event(NO_CHORD)
    self.assertEqual(0, index)

  def testEncodeChord(self):
    # major triad
    index = self.enc.encode_event('C13')
    self.assertEqual(1, index)

    # minor triad
    index = self.enc.encode_event('Cm(maj7)')
    self.assertEqual(13, index)

    # augmented triad
    index = self.enc.encode_event('Faug7')
    self.assertEqual(30, index)

    # diminished triad
    index = self.enc.encode_event('Abm7b5')
    self.assertEqual(45, index)

  def testEncodeThirdlessChord(self):
    # suspended chord
    with self.assertRaises(chords_encoder_decoder.ChordEncodingException):
      self.enc.encode_event('Gsus4')

    # power chord
    with self.assertRaises(chords_encoder_decoder.ChordEncodingException):
      self.enc.encode_event('Bb5')

  def testDecodeNoChord(self):
    figure = self.enc.decode_event(0)
    self.assertEqual(NO_CHORD, figure)

  def testDecodeChord(self):
    # major chord
    figure = self.enc.decode_event(3)
    self.assertEqual('D', figure)

    # minor chord
    figure = self.enc.decode_event(17)
    self.assertEqual('Em', figure)

    # augmented chord
    figure = self.enc.decode_event(33)
    self.assertEqual('Abaug', figure)

    # diminished chord
    figure = self.enc.decode_event(42)
    self.assertEqual('Fdim', figure)


class PitchChordsEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = chords_encoder_decoder.PitchChordsEncoderDecoder()

  def testInputSize(self):
    self.assertEqual(37, self.enc.input_size)

  def testEncodeNoChord(self):
    input_ = self.enc.events_to_input([NO_CHORD], 0)
    self.assertEqual([1.0] + [0.0] * 36, input_)

  def testEncodeChord(self):
    # major triad
    input_ = self.enc.events_to_input(['C'], 0)
    expected = [0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.assertEqual(expected, input_)

    # minor triad
    input_ = self.enc.events_to_input(['F#m'], 0)
    expected = [0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.assertEqual(expected, input_)

    # major triad with dominant 7th in bass
    input_ = self.enc.events_to_input(['G/F'], 0)
    expected = [0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.assertEqual(expected, input_)

    # 13th chord
    input_ = self.enc.events_to_input(['E13'], 0)
    expected = [0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.assertEqual(expected, input_)

    # minor triad with major 7th
    input_ = self.enc.events_to_input(['Fm(maj7)'], 0)
    expected = [0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.assertEqual(expected, input_)


if __name__ == '__main__':
  tf.test.main()
