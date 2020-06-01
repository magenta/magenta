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

"""Tests for spectral_ops.

Most tests check for parity with numpy operations.
"""
from absl.testing import parameterized
from magenta.models.gansynth.lib import spectral_ops
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class SpectralOpsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {'freqs': [10, 20, 30, 40]},
      {'freqs': 111.32},
      {'freqs': np.linspace(20, 20000, 100)})
  def testHertzToMel(self, freqs):
    mel = spectral_ops.hertz_to_mel(freqs)
    hz = spectral_ops.mel_to_hertz(mel)
    self.assertAllClose(hz, freqs)

  @parameterized.parameters(
      (32, 200),
      (64, 512),
      (1024, 1024))
  def testLinear2MelMatrix(self, n_mel, n_spec):
    l2m = spectral_ops.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel,
        num_spectrogram_bins=n_spec,
        sample_rate=16000,
        lower_edge_hertz=0.0,
        upper_edge_hertz=8000.0)
    self.assertEqual(l2m.shape, (n_spec, n_mel))

  @parameterized.parameters(
      {'shape': [7], 'axis': 0},
      {'shape': [10, 20], 'axis': 0},
      {'shape': [10, 20], 'axis': 1},
      {'shape': [10, 20, 3], 'axis': 2})
  def testDiff(self, shape, axis):
    x_np = np.random.randn(*shape)
    x_tf = tf.convert_to_tensor(x_np)
    res_np = np.diff(x_np, axis=axis)
    with self.cached_session() as sess:
      res_tf = sess.run(spectral_ops.diff(x_tf, axis=axis))
    self.assertEqual(res_np.shape, res_tf.shape)
    self.assertAllClose(res_np, res_tf)

  @parameterized.parameters(
      {'shape': [7], 'axis': 0},
      {'shape': [10, 20], 'axis': 0},
      {'shape': [10, 20], 'axis': 1},
      {'shape': [10, 20, 3], 'axis': 2})
  def testUnwrap(self, shape, axis):
    x_np = 5 * np.random.randn(*shape)
    x_tf = tf.convert_to_tensor(x_np)
    res_np = np.unwrap(x_np, axis=axis)
    with self.cached_session() as sess:
      res_tf = sess.run(spectral_ops.unwrap(x_tf, axis=axis))
    self.assertEqual(res_np.shape, res_tf.shape)
    self.assertAllClose(res_np, res_tf)

  @parameterized.parameters(
      {'shape': [10, 10]},
      {'shape': [128, 512]})
  def testPolar2Rect(self, shape):
    mag_np = 10 * np.random.rand(*shape)
    phase_np = np.pi * (2 * np.random.rand(*shape) - 1)
    rect_np = mag_np * np.cos(phase_np) + 1.0j * mag_np * np.sin(phase_np)
    mag_tf = tf.convert_to_tensor(mag_np)
    phase_tf = tf.convert_to_tensor(phase_np)
    with self.cached_session() as sess:
      rect_tf = sess.run(spectral_ops.polar2rect(mag_tf, phase_tf))
    self.assertAllClose(rect_np, rect_tf)

  @parameterized.parameters(
      {'shape': [7], 'axis': 0},
      {'shape': [10, 20], 'axis': 0},
      {'shape': [10, 20], 'axis': 1},
      {'shape': [10, 20, 3], 'axis': 2})
  def testInstantaneousFrequency(self, shape, axis):
    # Instantaneous Frequency in numpy
    phase_np = np.pi * (2 * np.random.rand(*shape) - 1)
    unwrapped_np = np.unwrap(phase_np, axis=axis)
    dphase_np = np.diff(unwrapped_np, axis=axis)
    # Append with initial phase
    s = [slice(None),] * unwrapped_np.ndim
    s[axis] = slice(0, 1)
    slice_np = unwrapped_np[s]
    dphase_np = np.concatenate([slice_np, dphase_np], axis=axis) / np.pi

    phase_tf = tf.convert_to_tensor(phase_np)
    with self.cached_session() as sess:
      dphase_tf = sess.run(spectral_ops.instantaneous_frequency(phase_tf,
                                                                time_axis=axis))
    self.assertAllClose(dphase_np, dphase_tf)

if __name__ == '__main__':
  tf.test.main()
