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

"""Library of spectral processing functions.

Includes transforming linear to mel frequency scales and phase to instantaneous
frequency.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def mel_to_hertz(mel_values):
  """Converts frequencies in `mel_values` from the mel scale to linear scale."""
  return _MEL_BREAK_FREQUENCY_HERTZ * (
      np.exp(np.array(mel_values) / _MEL_HIGH_FREQUENCY_Q) - 1.0)


def hertz_to_mel(frequencies_hertz):
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (np.array(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ))


def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=16000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0):
  """Returns a matrix to warp linear scale spectrograms to the mel scale.

  Adapted from tf.signal.linear_to_mel_weight_matrix with a minimum
  band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
  we compute the matrix at float64 precision and then cast to `dtype`
  at the end. This function can be constant folded by graph optimization
  since there are no Tensor inputs.

  Args:
    num_mel_bins: Int, number of output frequency dimensions.
    num_spectrogram_bins: Int, number of input frequency dimensions.
    sample_rate: Int, sample rate of the audio.
    lower_edge_hertz: Float, lowest frequency to consider.
    upper_edge_hertz: Float, highest frequency to consider.

  Returns:
    Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].

  Raises:
    ValueError: Input argument in the wrong range.
  """
  # Validate input arguments
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if num_spectrogram_bins <= 0:
    raise ValueError(
        'num_spectrogram_bins must be positive. Got: %s' % num_spectrogram_bins)
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if lower_edge_hertz < 0.0:
    raise ValueError(
        'lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                     'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
                     % (upper_edge_hertz, sample_rate))

  # HTK excludes the spectrogram DC bin.
  bands_to_zero = 1
  nyquist_hertz = sample_rate / 2.0
  linear_frequencies = np.linspace(
      0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:, np.newaxis]
  # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  band_edges_mel = np.linspace(
      hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz),
      num_mel_bins + 2)

  lower_edge_mel = band_edges_mel[0:-2]
  center_mel = band_edges_mel[1:-1]
  upper_edge_mel = band_edges_mel[2:]

  freq_res = nyquist_hertz / float(num_spectrogram_bins)
  freq_th = 1.5 * freq_res
  for i in range(0, num_mel_bins):
    center_hz = mel_to_hertz(center_mel[i])
    lower_hz = mel_to_hertz(lower_edge_mel[i])
    upper_hz = mel_to_hertz(upper_edge_mel[i])
    if upper_hz - lower_hz < freq_th:
      rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
      dm = _MEL_HIGH_FREQUENCY_Q * np.log(rhs + np.sqrt(1.0 + rhs**2))
      lower_edge_mel[i] = center_mel[i] - dm
      upper_edge_mel[i] = center_mel[i] + dm

  lower_edge_hz = mel_to_hertz(lower_edge_mel)[np.newaxis, :]
  center_hz = mel_to_hertz(center_mel)[np.newaxis, :]
  upper_edge_hz = mel_to_hertz(upper_edge_mel)[np.newaxis, :]

  # Calculate lower and upper slopes for every spectrogram bin.
  # Line segments are linear in the mel domain, not Hertz.
  lower_slopes = (linear_frequencies - lower_edge_hz) / (
      center_hz - lower_edge_hz)
  upper_slopes = (upper_edge_hz - linear_frequencies) / (
      upper_edge_hz - center_hz)

  # Intersect the line segments with each other and zero.
  mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

  # Re-add the zeroed lower bins we sliced out above.
  # [freq, mel]
  mel_weights_matrix = np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]],
                              'constant')
  return mel_weights_matrix


def diff(x, axis=-1):
  """Take the finite difference of a tensor along an axis.

  Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.

  Returns:
    d: Tensor with size less than x by 1 along the difference dimension.

  Raises:
    ValueError: Axis out of range for tensor.
  """
  shape = x.get_shape()
  if axis >= len(shape):
    raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                     (axis, len(shape)))

  begin_back = [0 for unused_s in range(len(shape))]
  begin_front = [0 for unused_s in range(len(shape))]
  begin_front[axis] = 1

  size = shape.as_list()
  size[axis] -= 1
  slice_front = tf.slice(x, begin_front, size)
  slice_back = tf.slice(x, begin_back, size)
  d = slice_front - slice_back
  return d


def unwrap(p, discont=np.pi, axis=-1):
  """Unwrap a cyclical phase tensor.

  Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.

  Returns:
    unwrapped: Unwrapped tensor of same size as input.
  """
  dd = diff(p, axis=axis)
  ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
  idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
  ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
  ph_correct = ddmod - dd
  idx = tf.less(tf.abs(dd), discont)
  ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
  ph_cumsum = tf.cumsum(ph_correct, axis=axis)

  shape = p.get_shape().as_list()
  shape[axis] = 1
  ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
  unwrapped = p + ph_cumsum
  return unwrapped


def instantaneous_frequency(phase_angle, time_axis=-2, use_unwrap=True):
  """Transform a fft tensor from phase angle to instantaneous frequency.

  Take the finite difference of the phase. Pad with initial phase to keep the
  tensor the same size.
  Args:
    phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
    time_axis: Axis over which to unwrap and take finite difference.
    use_unwrap: True preserves original GANSynth behavior, whereas False will
        guard against loss of precision.

  Returns:
    dphase: Instantaneous frequency (derivative of phase). Same size as input.
  """
  if use_unwrap:
    # Can lead to loss of precision.
    phase_unwrapped = unwrap(phase_angle, axis=time_axis)
    dphase = diff(phase_unwrapped, axis=time_axis)
  else:
    # Keep dphase bounded. N.B. runs faster than a single mod-2pi expression.
    dphase = diff(phase_angle, axis=time_axis)
    dphase = tf.where(dphase > np.pi, dphase - 2 * np.pi, dphase)
    dphase = tf.where(dphase < -np.pi, dphase + 2 * np.pi, dphase)

  # Add an initial phase to dphase.
  size = phase_angle.get_shape().as_list()
  size[time_axis] = 1
  begin = [0 for unused_s in size]
  phase_slice = tf.slice(phase_angle, begin, size)
  dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
  return dphase


def polar2rect(mag, phase_angle):
  """Convert polar-form complex number to its rectangular form."""
  mag = tf.complex(mag, tf.convert_to_tensor(0.0, dtype=mag.dtype))
  phase = tf.complex(tf.cos(phase_angle), tf.sin(phase_angle))
  return mag * phase


def random_phase_in_radians(shape, dtype):
  return np.pi * (2 * tf.random_uniform(shape, dtype=dtype) - 1.0)


def crop_or_pad(waves, length, channels):
  """Crop or pad wave to have shape [N, length, channels].

  Args:
    waves: A 3D `Tensor` of NLC format.
    length: A Python scalar. The output wave size.
    channels: Number of output waves channels.

  Returns:
    A 3D `Tensor` of NLC format with shape [N, length, channels].
  """
  waves = tf.convert_to_tensor(waves)
  batch_size = int(waves.shape[0])
  waves_shape = tf.shape(waves)

  # Force audio length.
  pad = tf.maximum(0, length - waves_shape[1])
  right_pad = tf.to_int32(tf.to_float(pad) / 2.0)
  left_pad = pad - right_pad
  waves = tf.pad(waves, [[0, 0], [left_pad, right_pad], [0, 0]])
  waves = waves[:, :length, :]

  # Force number of channels.
  num_repeats = tf.to_int32(
      tf.ceil(tf.to_float(channels) / tf.to_float(waves_shape[2])))
  waves = tf.tile(waves, [1, 1, num_repeats])[:, :, :channels]

  waves.set_shape([batch_size, length, channels])
  return waves
