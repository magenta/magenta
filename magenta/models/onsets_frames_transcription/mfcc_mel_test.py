# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mfcc_mel."""

from absl.testing import absltest
from magenta.models.onsets_frames_transcription import mfcc_mel
import numpy as np


class MfccMelTest(absltest.TestCase):

  def testMelSpectrumAgreesWithGoldenValues(self):
    # Parallel dsp/mfcc:mel_spectrum_test.
    sample_count = 513
    input_ = np.sqrt(np.arange(1, sample_count + 1))[np.newaxis, :]
    spec_to_mel_matrix = mfcc_mel.SpectrogramToMelMatrix(
        num_spectrogram_bins=sample_count,
        audio_sample_rate=22050,
        num_mel_bins=20,
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0)
    mel_spectrum = np.dot(input_, spec_to_mel_matrix)
    expected = np.array(
        [7.422619, 10.30330648, 13.72703292, 17.24158686, 21.35253118,
         25.77781089, 31.30624108, 37.05877236, 43.9436536, 51.80306637,
         60.79867148, 71.14363376, 82.90910141, 96.50069158, 112.08428368,
         129.96721968, 150.4277597, 173.74997634, 200.86037462, 231.59802942])
    np.testing.assert_array_almost_equal(expected, mel_spectrum[0, :])

  def testSpectrogramToMelMatrixChecksFrequencyBounds(self):
    # Lower edge must be >= 0, but 0 is OK.
    mfcc_mel.SpectrogramToMelMatrix(
        num_spectrogram_bins=513,
        audio_sample_rate=22050,
        num_mel_bins=20,
        lower_edge_hertz=0.0,
        upper_edge_hertz=4000.0)
    with self.assertRaises(ValueError):
      mfcc_mel.SpectrogramToMelMatrix(
          num_spectrogram_bins=513,
          audio_sample_rate=22050,
          num_mel_bins=20,
          lower_edge_hertz=-1.0,
          upper_edge_hertz=4000.0)
    # Upper edge must be <= Nyquist, but Nyquist is OK.
    mfcc_mel.SpectrogramToMelMatrix(
        num_spectrogram_bins=513,
        audio_sample_rate=22050,
        num_mel_bins=20,
        lower_edge_hertz=20.0,
        upper_edge_hertz=11025.0)
    with self.assertRaises(ValueError):
      mfcc_mel.SpectrogramToMelMatrix(
          num_spectrogram_bins=513,
          audio_sample_rate=22050,
          num_mel_bins=20,
          lower_edge_hertz=20.0,
          upper_edge_hertz=16000.0)
    # Must be a positive gap between edges.
    with self.assertRaises(ValueError):
      mfcc_mel.SpectrogramToMelMatrix(
          num_spectrogram_bins=513,
          audio_sample_rate=22050,
          num_mel_bins=20,
          lower_edge_hertz=20.0,
          upper_edge_hertz=20.0)


if __name__ == "__main__":
  absltest.main()
