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

"""Helper object for transforming audio to spectra.

Handles transformations between waveforms, stfts, spectrograms,
mel-spectrograms, and instantaneous frequency (specgram).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.gansynth.lib import spectral_ops
import numpy as np
import tensorflow.compat.v1 as tf


class SpecgramsHelper(object):
  """Helper functions to compute specgrams."""

  def __init__(self, audio_length, spec_shape, overlap,
               sample_rate, mel_downscale, ifreq=True, discard_dc=True):
    self._audio_length = audio_length
    self._spec_shape = spec_shape
    self._overlap = overlap
    self._sample_rate = sample_rate
    self._mel_downscale = mel_downscale
    self._ifreq = ifreq
    self._discard_dc = discard_dc

    self._nfft, self._nhop = self._get_nfft_nhop()
    self._pad_l, self._pad_r = self._get_padding()

    self._eps = 1.0e-6

  def _safe_log(self, x):
    return tf.log(x + self._eps)

  def _get_nfft_nhop(self):
    n_freq_bins = self._spec_shape[1]
    # Power of two only has 1 nonzero in binary representation
    is_power_2 = bin(n_freq_bins).count('1') == 1
    if not is_power_2:
      raise ValueError('Wrong spec_shape. Number of frequency bins must be '
                       'a power of 2, not %d' % n_freq_bins)
    nfft = n_freq_bins * 2
    nhop = int((1. - self._overlap) * nfft)
    return (nfft, nhop)

  def _get_padding(self):
    """Infer left and right padding for STFT."""
    n_samps_inv = self._nhop * (self._spec_shape[0] - 1) + self._nfft
    if n_samps_inv < self._audio_length:
      raise ValueError('Wrong audio length. Number of ISTFT samples, %d, should'
                       ' be less than audio lengeth %d' % self._audio_length)

    # For Nsynth dataset, we are putting all padding in the front
    # This causes edge effects in the tail
    padding = n_samps_inv - self._audio_length
    padding_l = padding
    padding_r = padding - padding_l
    return padding_l, padding_r

  def waves_to_stfts(self, waves):
    """Convert from waves to complex stfts.

    Args:
      waves: Tensor of the waveform, shape [batch, time, 1].

    Returns:
      stfts: Complex64 tensor of stft, shape [batch, time, freq, 1].
    """
    waves_padded = tf.pad(waves, [[0, 0], [self._pad_l, self._pad_r], [0, 0]])
    stfts = tf.signal.stft(
        waves_padded[:, :, 0],
        frame_length=self._nfft,
        frame_step=self._nhop,
        fft_length=self._nfft,
        pad_end=False)[:, :, :, tf.newaxis]
    stfts = stfts[:, :, 1:] if self._discard_dc else stfts[:, :, :-1]
    stft_shape = stfts.get_shape().as_list()[1:3]
    if tuple(stft_shape) != tuple(self._spec_shape):
      raise ValueError(
          'Spectrogram returned the wrong shape {}, is not the same as the '
          'constructor spec_shape {}.'.format(stft_shape, self._spec_shape))
    return stfts

  def stfts_to_waves(self, stfts):
    """Convert from complex stfts to waves.

    Args:
      stfts: Complex64 tensor of stft, shape [batch, time, freq, 1].

    Returns:
      waves: Tensor of the waveform, shape [batch, time, 1].
    """
    dc = 1 if self._discard_dc else 0
    nyq = 1 - dc
    stfts = tf.pad(stfts, [[0, 0], [0, 0], [dc, nyq], [0, 0]])
    waves_resyn = tf.signal.inverse_stft(
        stfts=stfts[:, :, :, 0],
        frame_length=self._nfft,
        frame_step=self._nhop,
        fft_length=self._nfft,
        window_fn=tf.signal.inverse_stft_window_fn(
            frame_step=self._nhop))[:, :, tf.newaxis]
    # Python does not allow rslice of -0
    if self._pad_r == 0:
      return waves_resyn[:, self._pad_l:]
    else:
      return waves_resyn[:, self._pad_l:-self._pad_r]

  def stfts_to_specgrams(self, stfts):
    """Converts stfts to specgrams.

    Args:
      stfts: Complex64 tensor of stft, shape [batch, time, freq, 1].

    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, time, freq, 2].
    """
    stfts = stfts[:, :, :, 0]

    logmag = self._safe_log(tf.abs(stfts))

    phase_angle = tf.angle(stfts)
    if self._ifreq:
      p = spectral_ops.instantaneous_frequency(phase_angle)
    else:
      p = phase_angle / np.pi

    return tf.concat(
        [logmag[:, :, :, tf.newaxis], p[:, :, :, tf.newaxis]], axis=-1)

  def specgrams_to_stfts(self, specgrams):
    """Converts specgrams to stfts.

    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, time, freq, 2].

    Returns:
      stfts: Complex64 tensor of stft, shape [batch, time, freq, 1].
    """
    logmag = specgrams[:, :, :, 0]
    p = specgrams[:, :, :, 1]

    mag = tf.exp(logmag)

    if self._ifreq:
      phase_angle = tf.cumsum(p * np.pi, axis=-2)
    else:
      phase_angle = p * np.pi

    return spectral_ops.polar2rect(mag, phase_angle)[:, :, :, tf.newaxis]

  def _linear_to_mel_matrix(self):
    """Get the mel transformation matrix."""
    num_freq_bins = self._nfft // 2
    lower_edge_hertz = 0.0
    upper_edge_hertz = self._sample_rate / 2.0
    num_mel_bins = num_freq_bins // self._mel_downscale
    return spectral_ops.linear_to_mel_weight_matrix(
        num_mel_bins, num_freq_bins, self._sample_rate, lower_edge_hertz,
        upper_edge_hertz)

  def _mel_to_linear_matrix(self):
    """Get the inverse mel transformation matrix."""
    m = self._linear_to_mel_matrix()
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

  def specgrams_to_melspecgrams(self, specgrams):
    """Converts specgrams to melspecgrams.

    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, time, freq, 2].

    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, time, freq, 2], mel scaling of frequencies.
    """
    if self._mel_downscale is None:
      return specgrams

    logmag = specgrams[:, :, :, 0]
    p = specgrams[:, :, :, 1]

    mag2 = tf.exp(2.0 * logmag)
    phase_angle = tf.cumsum(p * np.pi, axis=-2)

    l2mel = tf.to_float(self._linear_to_mel_matrix())
    logmelmag2 = self._safe_log(tf.tensordot(mag2, l2mel, 1))
    mel_phase_angle = tf.tensordot(phase_angle, l2mel, 1)
    mel_p = spectral_ops.instantaneous_frequency(mel_phase_angle)

    return tf.concat(
        [logmelmag2[:, :, :, tf.newaxis], mel_p[:, :, :, tf.newaxis]], axis=-1)

  def melspecgrams_to_specgrams(self, melspecgrams):
    """Converts melspecgrams to specgrams.

    Args:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, time, freq, 2], mel scaling of frequencies.

    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, time, freq, 2].
    """
    if self._mel_downscale is None:
      return melspecgrams

    logmelmag2 = melspecgrams[:, :, :, 0]
    mel_p = melspecgrams[:, :, :, 1]

    mel2l = tf.to_float(self._mel_to_linear_matrix())
    mag2 = tf.tensordot(tf.exp(logmelmag2), mel2l, 1)
    logmag = 0.5 * self._safe_log(mag2)
    mel_phase_angle = tf.cumsum(mel_p * np.pi, axis=-2)
    phase_angle = tf.tensordot(mel_phase_angle, mel2l, 1)
    p = spectral_ops.instantaneous_frequency(phase_angle)

    return tf.concat(
        [logmag[:, :, :, tf.newaxis], p[:, :, :, tf.newaxis]], axis=-1)

  def stfts_to_melspecgrams(self, stfts):
    """Converts stfts to mel-spectrograms."""
    return self.specgrams_to_melspecgrams(self.stfts_to_specgrams(stfts))

  def melspecgrams_to_stfts(self, melspecgrams):
    """Converts mel-spectrograms to stfts."""
    return self.specgrams_to_stfts(self.melspecgrams_to_specgrams(melspecgrams))

  def waves_to_specgrams(self, waves):
    """Converts waves to spectrograms."""
    return self.stfts_to_specgrams(self.waves_to_stfts(waves))

  def specgrams_to_waves(self, specgrams):
    """Converts spectrograms to stfts."""
    return self.stfts_to_waves(self.specgrams_to_stfts(specgrams))

  def waves_to_melspecgrams(self, waves):
    """Converts waves to mel-spectrograms."""
    return self.stfts_to_melspecgrams(self.waves_to_stfts(waves))

  def melspecgrams_to_waves(self, melspecgrams):
    """Converts mel-spectrograms to stfts."""
    return self.stfts_to_waves(self.melspecgrams_to_stfts(melspecgrams))
