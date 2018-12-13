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
"""Tests for audio_io.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import wave

import numpy as np
import scipy
import six
import tensorflow as tf

from magenta.music import audio_io


class AudioIoTest(tf.test.TestCase):

  def setUp(self):
    self.wav_filename = os.path.join(tf.resource_loader.get_data_files_path(),
                                     'testdata/example.wav')
    self.wav_filename_mono = os.path.join(
        tf.resource_loader.get_data_files_path(), 'testdata/example_mono.wav')
    self.wav_data = open(self.wav_filename, 'rb').read()
    self.wav_data_mono = open(self.wav_filename_mono, 'rb').read()

  def testWavDataToSamples(self):
    w = wave.open(self.wav_filename, 'rb')
    w_mono = wave.open(self.wav_filename_mono, 'rb')

    # Check content size.
    y = audio_io.wav_data_to_samples(self.wav_data, sample_rate=16000)
    y_mono = audio_io.wav_data_to_samples(self.wav_data_mono, sample_rate=22050)
    self.assertEqual(
        round(16000.0 * w.getnframes() / w.getframerate()), y.shape[0])
    self.assertEqual(
        round(22050.0 * w_mono.getnframes() / w_mono.getframerate()),
        y_mono.shape[0])

    # Check a few obvious failure modes.
    self.assertLess(0.01, y.std())
    self.assertLess(0.01, y_mono.std())
    self.assertGreater(-0.1, y.min())
    self.assertGreater(-0.1, y_mono.min())
    self.assertLess(0.1, y.max())
    self.assertLess(0.1, y_mono.max())

  def testFloatWavDataToSamples(self):
    y = audio_io.wav_data_to_samples(self.wav_data, sample_rate=16000)
    wav_io = six.BytesIO()
    scipy.io.wavfile.write(wav_io, 16000, y)
    y_from_float = audio_io.wav_data_to_samples(
        wav_io.getvalue(), sample_rate=16000)
    np.testing.assert_array_equal(y, y_from_float)


if __name__ == '__main__':
  tf.test.main()
