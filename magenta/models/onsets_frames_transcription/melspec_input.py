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

"""Create TF graphs for calculating log-mel-spectral features.

NOTE: This code is very experimental and will likely change, both in interface
and what it outputs.

The single published method is build_mel_calculation_graph, which
will assemble a TF graph from a provided waveform input vector
through to a (num_frames, frame_width, num_mel_bins) tensor of log-
transformed mel spectrogram patches, suitable for feeding the input
to a typical classifier. All the mel calculation parameters
are available as options, but default to their standard values
(e.g. frame_width=96, frame_hop=10). The input waveform can have
size (None,), meaning it will be specified at run-time.

with tflite_compatible=True, the returned graph is constructed only
from tflite-compatible ops (i.e., it uses matmul for the DFT, and
explicitly unrolled framing). In this case, the input waveform tensor
must have an explicit size at graph-building time.
"""

import fractions

import math

from magenta.models.onsets_frames_transcription import mfcc_mel
import numpy as np
import tensorflow.compat.v1 as tf


def _stft_magnitude_full_tf(waveform_input, window_length_samples,
                            hop_length_samples, fft_length):
  """Calculate STFT magnitude (spectrogram) using tf.signal ops."""
  stft_magnitude = tf.abs(
      tf.signal.stft(
          waveform_input,
          frame_length=window_length_samples,
          frame_step=hop_length_samples,
          fft_length=fft_length),
      name='magnitude_spectrogram')
  return stft_magnitude


def _dft_matrix(dft_length):
  """Calculate the full DFT matrix in numpy."""
  omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
  # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
  return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))


def _naive_rdft(signal_tensor, fft_length):
  """Implement real-input Fourier Transform by matmul."""
  # We are right-multiplying by the DFT matrix, and we are keeping
  # only the first half ("positive frequencies").
  # So discard the second half of rows, but transpose the array for
  # right-multiplication.
  # The DFT matrix is symmetric, so we could have done it more
  # directly, but this reflects our intention better.
  complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(
      fft_length // 2 + 1), :].transpose()
  real_dft_tensor = tf.constant(
      np.real(complex_dft_matrix_kept_values).astype(np.float32),
      name='real_dft_matrix')
  imag_dft_tensor = tf.constant(
      np.imag(complex_dft_matrix_kept_values).astype(np.float32),
      name='imaginary_dft_matrix')
  signal_frame_length = int(signal_tensor.shape[-1])
  half_pad = (fft_length - signal_frame_length) // 2
  pad_values = tf.concat([
      tf.zeros([tf.rank(signal_tensor) - 1, 2], tf.int32),
      [[half_pad, fft_length - signal_frame_length - half_pad]]
  ],
                         axis=0)
  padded_signal = tf.pad(signal_tensor, pad_values)
  result_real_part = tf.matmul(padded_signal, real_dft_tensor)
  result_imag_part = tf.matmul(padded_signal, imag_dft_tensor)
  return result_real_part, result_imag_part


def _fixed_frame(signal, frame_length, frame_step, first_axis=False):
  """tflite-compatible tf.signal.frame for fixed-size input.

  Args:
    signal: Tensor containing signal(s).
    frame_length: Number of samples to put in each frame.
    frame_step: Sample advance between successive frames.
    first_axis: If true, framing is applied to first axis of tensor; otherwise,
      it is applied to last axis.

  Returns:
    A new tensor where the last axis (or first, if first_axis) of input
    signal has been replaced by a (num_frames, frame_length) array of individual
    frames where each frame is drawn frame_step samples after the previous one.

  Raises:
    ValueError: if signal has an undefined axis length.  This routine only
      supports framing of signals whose shape is fixed at graph-build time.
  """
  signal_shape = signal.shape.as_list()
  if first_axis:
    length_samples = signal_shape[0]
  else:
    length_samples = signal_shape[-1]
  if length_samples <= 0:
    raise ValueError('fixed framing requires predefined constant signal length')
  num_frames = max(0, 1 + (length_samples - frame_length) // frame_step)
  if first_axis:
    inner_dimensions = signal_shape[1:]
    result_shape = [num_frames, frame_length] + inner_dimensions
    gather_axis = 0
  else:
    outer_dimensions = signal_shape[:-1]
    result_shape = outer_dimensions + [num_frames, frame_length]
    # Currently tflite's gather only supports axis==0, but that may still
    # work if we want the last of 1 axes.
    gather_axis = len(outer_dimensions)

  subframe_length = fractions.gcd(frame_length, frame_step)  # pylint: disable=deprecated-method
  subframes_per_frame = frame_length // subframe_length
  subframes_per_hop = frame_step // subframe_length
  num_subframes = length_samples // subframe_length

  if first_axis:
    trimmed_input_size = [num_subframes * subframe_length] + inner_dimensions
    subframe_shape = [num_subframes, subframe_length] + inner_dimensions
  else:
    trimmed_input_size = outer_dimensions + [num_subframes * subframe_length]
    subframe_shape = outer_dimensions + [num_subframes, subframe_length]
  subframes = tf.reshape(
      tf.slice(
          signal,
          begin=np.zeros(len(signal_shape), np.int32),
          size=trimmed_input_size), subframe_shape)

  # frame_selector is a [num_frames, subframes_per_frame] tensor
  # that indexes into the appropriate frame in subframes. For example:
  # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
  frame_selector = np.reshape(
      np.arange(num_frames) * subframes_per_hop, [num_frames, 1])

  # subframe_selector is a [num_frames, subframes_per_frame] tensor
  # that indexes into the appropriate subframe within a frame. For example:
  # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
  subframe_selector = np.reshape(
      np.arange(subframes_per_frame), [1, subframes_per_frame])

  # Adding the 2 selector tensors together produces a [num_frames,
  # subframes_per_frame] tensor of indices to use with tf.gather to select
  # subframes from subframes. We then reshape the inner-most subframes_per_frame
  # dimension to stitch the subframes together into frames. For example:
  # [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
  selector = frame_selector + subframe_selector
  frames = tf.reshape(
      tf.gather(subframes, selector.astype(np.int32), axis=gather_axis),
      result_shape)
  return frames


def _stft_tflite(signal, frame_length, frame_step, fft_length):
  """tflite-compatible implementation of tf.signal.stft.

  Compute the short-time Fourier transform of a 1D input while avoiding tf ops
  that are not currently supported in tflite (Rfft, Range, SplitV).
  fft_length must be fixed. A Hann window is of frame_length is always
  applied.

  Since fixed (precomputed) framing must be used, signal.shape[-1] must be a
  specific value (so "?"/None is not supported).

  Args:
    signal: 1D tensor containing the time-domain waveform to be transformed.
    frame_length: int, the number of points in each Fourier frame.
    frame_step: int, the number of samples to advance between successive frames.
    fft_length: int, the size of the Fourier transform to apply.

  Returns:
    Two (num_frames, fft_length) tensors containing the real and imaginary parts
    of the short-time Fourier transform of the input signal.
  """
  # Make the window be shape (1, frame_length) instead of just frame_length
  # in an effort to help the tflite broadcast logic.
  window = tf.reshape(
      tf.constant(
          (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
          ).astype(np.float32),
          name='window'), [1, frame_length])
  framed_signal = _fixed_frame(
      signal, frame_length, frame_step, first_axis=False)
  framed_signal *= window
  real_spectrogram, imag_spectrogram = _naive_rdft(framed_signal, fft_length)
  return real_spectrogram, imag_spectrogram


def _stft_magnitude_tflite(waveform_input, window_length_samples,
                           hop_length_samples, fft_length):
  """Calculate spectrogram avoiding tflite incompatible ops."""
  real_stft, imag_stft = _stft_tflite(
      waveform_input,
      frame_length=window_length_samples,
      frame_step=hop_length_samples,
      fft_length=fft_length)
  stft_magnitude = tf.sqrt(
      tf.add(real_stft * real_stft, imag_stft * imag_stft),
      name='magnitude_spectrogram')
  return stft_magnitude


def build_mel_calculation_graph(waveform_input,
                                sample_rate=16000,
                                window_length_seconds=0.025,
                                hop_length_seconds=0.010,
                                num_mel_bins=64,
                                lower_edge_hz=125.0,
                                upper_edge_hz=7500.0,
                                frame_width=96,
                                frame_hop=10,
                                tflite_compatible=False):
  """Build a TF graph to go from waveform to mel spectrum patches.

  Args:
    waveform_input: 1D Tensor which will be filled with 16 kHz waveform as
      tf.float32.
    sample_rate: Scalar giving the sampling rate of the waveform.  Only 16 kHz
      is acceptable at present.
    window_length_seconds: Duration of window used for each Fourier transform.
    hop_length_seconds: Time shift between successive analysis time frames.
    num_mel_bins: The number of mel frequency bins to calculate.
    lower_edge_hz: Frequency boundary at bottom edge of mel mapping.
    upper_edge_hz: Frequency boundary at top edge of mel mapping.
    frame_width: The number of successive time frames to include in each patch.
    frame_hop: The frame advance between successive patches.
    tflite_compatible: Avoid ops not currently supported in tflite.

  Returns:
    Tensor holding [num_patches, frame_width, num_mel_bins] log-mel-spectrogram
    patches.
  """
  # `waveform_input` is a [?] vector as a tensor.
  # `magnitude_spectrogram` is a [?, fft_length/2 + 1] tensor of spectrograms.
  # Derive the dependent parameters.
  window_length_samples = int(round(window_length_seconds * sample_rate))
  hop_length_samples = int(round(hop_length_seconds * sample_rate))
  fft_length = 2**int(
      math.ceil(math.log(window_length_samples) / math.log(2.0)))
  if tflite_compatible:
    magnitude_spectrogram = _stft_magnitude_tflite(
        waveform_input, window_length_samples, hop_length_samples, fft_length)
  else:
    magnitude_spectrogram = _stft_magnitude_full_tf(
        waveform_input, window_length_samples, hop_length_samples, fft_length)

  # Warp the linear-scale, magnitude spectrograms into the mel-scale.
  num_spectrogram_bins = int(magnitude_spectrogram.shape[-1])
  if tflite_compatible:
    linear_to_mel_weight_matrix = tf.constant(
        mfcc_mel.SpectrogramToMelMatrix(num_mel_bins, num_spectrogram_bins,
                                        sample_rate, lower_edge_hz,
                                        upper_edge_hz).astype(np.float32),
        name='linear_to_mel_matrix')
  else:
    # In full tf, the mel weight matrix is calculated at run time within the
    # TF graph.  This avoids including a matrix of 64 x 256 float values (i.e.,
    # 100 kB or more, depending on the representation) in the exported graph.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hz,
        upper_edge_hz)

  mel_spectrogram = tf.matmul(
      magnitude_spectrogram,
      linear_to_mel_weight_matrix,
      name='mel_spectrogram')
  log_offset = 0.001
  log_mel_spectrogram = tf.log(
      mel_spectrogram + log_offset, name='log_mel_spectrogram')
  # log_mel_spectrogram is a [?, num_mel_bins] gram.
  if tflite_compatible:
    features = _fixed_frame(
        log_mel_spectrogram,
        frame_length=frame_width,
        frame_step=frame_hop,
        first_axis=True)
  else:
    features = tf.signal.frame(
        log_mel_spectrogram,
        frame_length=frame_width,
        frame_step=frame_hop,
        axis=0)
  # features is [num_patches, frame_width, num_mel_bins].
  return features
