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

"""Tests for specgrams_helper."""
import os

from absl import flags
from absl.testing import parameterized
from magenta.models.gansynth.lib import specgrams_helper
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

FLAGS = flags.FLAGS


class SpecgramsHelperTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(SpecgramsHelperTest, self).setUp()
    self.data_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../../../testdata/example_nsynth_audio.npy')
    audio = np.load(self.data_filename)
    # Reduce batch size and audio length to speed up test
    self.batch_size = 2
    self.sr = 16000
    self.audio_length = int(self.sr * 0.5)
    self.audio_np = audio[:self.batch_size, :self.audio_length, np.newaxis]
    self.audio = tf.convert_to_tensor(self.audio_np)
    # Test the standard configuration of SpecgramsHelper
    self.spec_shape = [128, 1024]
    self.overlap = 0.75
    self.sh = specgrams_helper.SpecgramsHelper(
        audio_length=self.audio_length,
        spec_shape=tuple(self.spec_shape),
        overlap=self.overlap,
        sample_rate=self.sr,
        mel_downscale=1,
        ifreq=True,
        discard_dc=True)
    self.transform_pairs = {
        'stfts':
            (self.sh.waves_to_stfts, self.sh.stfts_to_waves),
        'specgrams':
            (self.sh.waves_to_specgrams, self.sh.specgrams_to_waves),
        'melspecgrams':
            (self.sh.waves_to_melspecgrams, self.sh.melspecgrams_to_waves)
    }

  @parameterized.parameters(
      ('stfts', 1),
      ('specgrams', 2),
      ('melspecgrams', 2))
  def testShapesAndReconstructions(self, transform_name, target_channels):
    # Transform the data, test shape
    transform = self.transform_pairs[transform_name][0]
    target_shape = tuple([self.batch_size]
                         + self.spec_shape + [target_channels])
    with self.cached_session() as sess:
      spectra_np = sess.run(transform(self.audio))
    self.assertEqual(spectra_np.shape, target_shape)

    # Reconstruct the audio, test shape
    inv_transform = self.transform_pairs[transform_name][1]
    with self.cached_session() as sess:
      recon_np = sess.run(inv_transform(tf.convert_to_tensor(spectra_np)))
    self.assertEqual(recon_np.shape, (self.batch_size, self.audio_length, 1))

    # Test reconstruction error
    # Mel compression adds differences so skip
    if transform_name != 'melspecgrams':
      # Edges have known differences due to windowing
      edge = self.spec_shape[1] * 2
      diff = np.abs(self.audio_np[:, edge:-edge] - recon_np[:, edge:-edge])
      rms = np.mean(diff**2.0)**0.5
      print(transform_name, 'RMS:', rms)
      self.assertLessEqual(rms, 1e-5)


if __name__ == '__main__':
  tf.test.main()
