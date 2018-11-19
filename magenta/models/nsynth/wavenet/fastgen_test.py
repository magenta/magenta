"""Tests for fastgen."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import librosa
import numpy as np
from scipy.io import wavfile
import tensorflow as tf

import fastgen


class FastegenTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {'batch_size': 1},
      {'batch_size': 10})
  def testLoadFastgenNsynth(self, batch_size):
    net = fastgen.load_fastgen_nsynth(batch_size=batch_size)
    with self.test_session() as sess:
      sess.run(net['init_ops'])
      self.assertEqual(net['X'].shape, (batch_size, 1))
      self.assertEqual(net['encoding'].shape, (batch_size, 16))
      self.assertEqual(net['predictions'].shape, (batch_size, 256))

  @parameterized.parameters(
      {'batch_size': 1, 'sample_length': 1024 * 10},
      {'batch_size': 10, 'sample_length': 1024 * 10},
      {'batch_size': 10, 'sample_length': 1024 * 20},
  )
  def testLoadNsynth(self, batch_size, sample_length):
    net = fastgen.load_nsynth(batch_size=batch_size, sample_length=sample_length)
    encodings_length = int(sample_length/512)
    with self.test_session() as sess:
      self.assertEqual(net['X'].shape, (batch_size, sample_length))
      self.assertEqual(net['encoding'].shape, (batch_size, encodings_length, 16))
      self.assertEqual(net['predictions'].shape, (batch_size * sample_length, 256))

  def testLoadBatch(self):
    test_audio = np.random.randn(64000)
    # Make temp dir
    test_dir = tf.test.get_temp_dir()
    tf.gfile.MakeDirs(test_dir)
    # Make wav files
    fname = os.path.join(test_dir, 'test_audio.wav')
    librosa.output.write_wav(fname, test_audio, sr=16000, norm=True)
    fname_2 = os.path.join(test_dir, 'test_audio_2.wav')
    librosa.output.write_wav(fname_2, test_audio, sr=16000, norm=True)
    # Load the files
    files = [fname, fname_2]
    batch_data = fastgen.load_batch(files, sample_length=64000)
    print(batch_data.shape)

if __name__ == '__main__':
  tf.test.main()
