# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Utility functions for NSynth."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os

# internal imports
import librosa
import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

slim = tf.contrib.slim


def shell_path(path):
  return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


#===============================================================================
# WaveNet Functions
#===============================================================================
def get_module(module_path):
  """Imports module from NSynth directory.

  Args:
    module_path: Path to module separated by dots.
      -> "configs.linear"

  Returns:
    module: Imported module.
  """
  import_path = "magenta.models.nsynth."
  module = importlib.import_module(import_path + module_path)
  return module


def load_audio(path, sample_length=64000, sr=16000):
  """Loading of a wave file.

  Args:
    path: Location of a wave file to load.
    sample_length: The truncated total length of the final wave file.
    sr: Samples per a second.

  Returns:
    out: The audio in samples from -1.0 to 1.0
  """
  audio, _ = librosa.load(path, sr=sr)
  audio = audio[:sample_length]
  return audio


def mu_law(x, mu=255, int8=False):
  """A TF implementation of Mu-Law encoding.

  Args:
    x: The audio samples to encode.
    mu: The Mu to use in our Mu-Law.
    int8: Use int8 encoding.

  Returns:
    out: The Mu-Law encoded int8 data.
  """
  out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
  out = tf.floor(out * 128)
  if int8:
    out = tf.cast(out, tf.int8)
  return out


def inv_mu_law(x, mu=255):
  """A TF implementation of inverse Mu-Law.

  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.

  Returns:
    out: The decoded data.
  """
  x = tf.cast(x, tf.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
  out = tf.where(tf.equal(x, 0), x, out)
  return out


def inv_mu_law_numpy(x, mu=255.0):
  """A numpy implementation of inverse Mu-Law.

  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.

  Returns:
    out: The decoded data.
  """
  x = np.array(x).astype(np.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
  out = np.where(np.equal(x, 0), x, out)
  return out


def trim_for_encoding(wav_data, sample_length, hop_length=512):
  """Make sure audio is a even multiple of hop_size.

  Args:
    wav_data: 1-D or 2-D array of floats.
    sample_length: Max length of audio data.
    hop_length: Pooling size of WaveNet autoencoder.

  Returns:
    wav_data: Trimmed array.
    sample_length: Length of trimmed array.
  """
  if wav_data.ndim == 1:
    # Max sample length is the data length
    if sample_length > wav_data.size:
      sample_length = wav_data.size
    # Multiple of hop_length
    sample_length = (sample_length // hop_length) * hop_length
    # Trim
    wav_data = wav_data[:sample_length]
  # Assume all examples are the same length
  elif wav_data.ndim == 2:
    # Max sample length is the data length
    if sample_length > wav_data[0].size:
      sample_length = wav_data[0].size
    # Multiple of hop_length
    sample_length = (sample_length // hop_length) * hop_length
    # Trim
    wav_data = wav_data[:, :sample_length]

  return wav_data, sample_length


#===============================================================================
# Baseline Functions
#===============================================================================
#---------------------------------------------------
# Pre/Post-processing
#---------------------------------------------------
def get_optimizer(learning_rate, hparams):
  """Get the tf.train.Optimizer for this optimizer string.

  Args:
    learning_rate: The learning_rate tensor.
      hparams: TF.HParams object with the optimizer and momentum values.

  Returns:
    optimizer: The tf.train.Optimizer based on the optimizer string.
  """
  return {
      "rmsprop":
          tf.RMSPropOptimizer(
              learning_rate,
              decay=0.95,
              momentum=hparams.momentum,
              epsilon=1e-4),
      "adam":
          tf.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8),
      "adagrad":
          tf.AdagradOptimizer(learning_rate, initial_accumulator_value=1.0),
      "mom":
          tf.MomentumOptimizer(learning_rate, momentum=hparams.momentum),
      "sgd":
          tf.GradientDescentOptimizer(learning_rate)
  }.get(hparams.optimizer)


def specgram(audio,
             n_fft=512,
             hop_length=None,
             mask=True,
             log_mag=True,
             re_im=False,
             dphase=True,
             mag_only=False):
  """Spectrogram using librosa.

  Args:
    audio: 1-D array of float32 sound samples.
    n_fft: Size of the FFT.
    hop_length: Stride of FFT. Defaults to n_fft/2.
    mask: Mask the phase derivative by the magnitude.
    log_mag: Use the logamplitude.
    re_im: Output Real and Imag. instead of logMag and dPhase.
    dphase: Use derivative of phase instead of phase.
    mag_only: Don't return phase.

  Returns:
    specgram: [n_fft/2 + 1, audio.size / hop_length, 2]. The first channel is
      the logamplitude and the second channel is the derivative of phase.
  """
  if not hop_length:
    hop_length = int(n_fft / 2.)

  fft_config = dict(
      n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)

  spec = librosa.stft(audio, **fft_config)

  if re_im:
    re = spec.real[:, :, np.newaxis]
    im = spec.imag[:, :, np.newaxis]
    spec_real = np.concatenate((re, im), axis=2)

  else:
    mag, phase = librosa.core.magphase(spec)
    phase_angle = np.angle(phase)

    # Magnitudes, scaled 0-1
    if log_mag:
      mag = (librosa.power_to_db(
          mag**2, amin=1e-13, top_db=120., ref=np.max) / 120.) + 1
    else:
      mag /= mag.max()

    if dphase:
      #  Derivative of phase
      phase_unwrapped = np.unwrap(phase_angle)
      p = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
      p = np.concatenate([phase_unwrapped[:, 0:1], p], axis=1) / np.pi
    else:
      # Normal phase
      p = phase_angle / np.pi
    # Mask the phase
    if log_mag and mask:
      p = mag * p
    # Return Mag and Phase
    p = p.astype(np.float32)[:, :, np.newaxis]
    mag = mag.astype(np.float32)[:, :, np.newaxis]
    if mag_only:
      spec_real = mag[:, :, np.newaxis]
    else:
      spec_real = np.concatenate((mag, p), axis=2)
  return spec_real


def inv_magphase(mag, phase_angle):
  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  return mag * phase


def griffin_lim(mag, phase_angle, n_fft, hop, num_iters):
  """Iterative algorithm for phase retrival from a magnitude spectrogram.

  Args:
    mag: Magnitude spectrogram.
    phase_angle: Initial condition for phase.
    n_fft: Size of the FFT.
    hop: Stride of FFT. Defaults to n_fft/2.
    num_iters: Griffin-Lim iterations to perform.

  Returns:
    audio: 1-D array of float32 sound samples.
  """
  fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
  ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
  complex_specgram = inv_magphase(mag, phase_angle)
  for i in range(num_iters):
    audio = librosa.istft(complex_specgram, **ifft_config)
    if i != num_iters - 1:
      complex_specgram = librosa.stft(audio, **fft_config)
      _, phase = librosa.magphase(complex_specgram)
      phase_angle = np.angle(phase)
      complex_specgram = inv_magphase(mag, phase_angle)
  return audio


def ispecgram(spec,
              n_fft=512,
              hop_length=None,
              mask=True,
              log_mag=True,
              re_im=False,
              dphase=True,
              mag_only=True,
              num_iters=1000):
  """Inverse Spectrogram using librosa.

  Args:
    spec: 3-D specgram array [freqs, time, (mag_db, dphase)].
    n_fft: Size of the FFT.
    hop_length: Stride of FFT. Defaults to n_fft/2.
    mask: Reverse the mask of the phase derivative by the magnitude.
    log_mag: Use the logamplitude.
    re_im: Output Real and Imag. instead of logMag and dPhase.
    dphase: Use derivative of phase instead of phase.
    mag_only: Specgram contains no phase.
    num_iters: Number of griffin-lim iterations for mag_only.

  Returns:
    audio: 1-D array of sound samples. Peak normalized to 1.
  """
  if not hop_length:
    hop_length = n_fft // 2

  ifft_config = dict(win_length=n_fft, hop_length=hop_length, center=True)

  if mag_only:
    mag = spec[:, :, 0]
    phase_angle = np.pi * np.random.rand(*mag.shape)
  elif re_im:
    spec_real = spec[:, :, 0] + 1.j * spec[:, :, 1]
  else:
    mag, p = spec[:, :, 0], spec[:, :, 1]
    if mask and log_mag:
      p /= (mag + 1e-13 * np.random.randn(*mag.shape))
    if dphase:
      # Roll up phase
      phase_angle = np.cumsum(p * np.pi, axis=1)
    else:
      phase_angle = p * np.pi

  # Magnitudes
  if log_mag:
    mag = (mag - 1.0) * 120.0
    mag = 10**(mag / 20.0)
  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  spec_real = mag * phase

  if mag_only:
    audio = griffin_lim(
        mag, phase_angle, n_fft, hop_length, num_iters=num_iters)
  else:
    audio = librosa.core.istft(spec_real, **ifft_config)
  return np.squeeze(audio / audio.max())


def batch_specgram(audio,
                   n_fft=512,
                   hop_length=None,
                   mask=True,
                   log_mag=True,
                   re_im=False,
                   dphase=True,
                   mag_only=False):
  assert len(audio.shape) == 2
  batch_size = audio.shape[0]
  res = []
  for b in range(batch_size):
    res.append(
        specgram(audio[b], n_fft, hop_length, mask, log_mag, re_im, dphase,
                 mag_only))
  return np.array(res)


def batch_ispecgram(spec,
                    n_fft=512,
                    hop_length=None,
                    mask=True,
                    log_mag=True,
                    re_im=False,
                    dphase=True,
                    mag_only=False,
                    num_iters=1000):
  assert len(spec.shape) == 4
  batch_size = spec.shape[0]
  res = []
  for b in range(batch_size):
    res.append(
        ispecgram(spec[b, :, :, :], n_fft, hop_length, mask, log_mag, re_im,
                  dphase, mag_only, num_iters))
  return np.array(res)


def tf_specgram(audio,
                n_fft=512,
                hop_length=None,
                mask=True,
                log_mag=True,
                re_im=False,
                dphase=True,
                mag_only=False):
  return tf.py_func(batch_specgram, [
      audio, n_fft, hop_length, mask, log_mag, re_im, dphase, mag_only
  ], tf.float32)


def tf_ispecgram(spec,
                 n_fft=512,
                 hop_length=None,
                 mask=True,
                 pad=True,
                 log_mag=True,
                 re_im=False,
                 dphase=True,
                 mag_only=False,
                 num_iters=1000):
  dims = spec.get_shape().as_list()
  # Add back in nyquist frequency
  x = spec if not pad else tf.concat(
      [spec, tf.zeros([dims[0], 1, dims[2], dims[3]])], 1)
  audio = tf.py_func(batch_ispecgram, [
      x, n_fft, hop_length, mask, log_mag, re_im, dphase, mag_only, num_iters
  ], tf.float32)
  return audio


#---------------------------------------------------
# Summaries
#---------------------------------------------------
def form_image_grid(input_tensor, grid_shape, image_shape, num_channels):
  """Arrange a minibatch of images into a grid to form a single image.

  Args:
    input_tensor: Tensor. Minibatch of images to format, either 4D
        ([batch size, height, width, num_channels]) or flattened
        ([batch size, height * width * num_channels]).
    grid_shape: Sequence of int. The shape of the image grid,
        formatted as [grid_height, grid_width].
    image_shape: Sequence of int. The shape of a single image,
        formatted as [image_height, image_width].
    num_channels: int. The number of channels in an image.

  Returns:
    Tensor representing a single image in which the input images have been
    arranged into a grid.

  Raises:
    ValueError: The grid shape and minibatch size don't match, or the image
        shape and number of channels are incompatible with the input tensor.
  """
  if grid_shape[0] * grid_shape[1] != int(input_tensor.get_shape()[0]):
    raise ValueError("Grid shape incompatible with minibatch size.")
  if len(input_tensor.get_shape()) == 2:
    num_features = image_shape[0] * image_shape[1] * num_channels
    if int(input_tensor.get_shape()[1]) != num_features:
      raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor.")
  elif len(input_tensor.get_shape()) == 4:
    if (int(input_tensor.get_shape()[1]) != image_shape[0] or
        int(input_tensor.get_shape()[2]) != image_shape[1] or
        int(input_tensor.get_shape()[3]) != num_channels):
      raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor.")
  else:
    raise ValueError("Unrecognized input tensor format.")
  height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
  input_tensor = tf.reshape(input_tensor,
                            grid_shape + image_shape + [num_channels])
  input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
  input_tensor = tf.reshape(
      input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
  input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
  input_tensor = tf.reshape(input_tensor, [1, height, width, num_channels])
  return input_tensor


def specgram_summaries(spec,
                       name,
                       hparams,
                       rows=4,
                       columns=4,
                       image=True,
                       phase=True,
                       audio=True):
  """Post summaries of a specgram (Image and Audio).

  For image summaries, creates a rows x columns composite image from the batch.
  Also can create audio summaries for raw audio, but hparams.raw_audio must be
  True.
  Args:
    spec: Batch of spectrograms.
    name: String prepended to summaries.
    hparams: Hyperparamenters.
    rows: Int, number of rows in image.
    columns: Int, number of columns in image.
    image: Bool, create image summary.
    phase: Bool, create image summary from second channel in the batch.
    audio: Bool, create audio summaries for each spectrogram in the batch.
  """
  batch_size, n_freq, n_time, unused_channels = spec.get_shape().as_list()
  # Must divide minibatch evenly
  b = min(batch_size, rows * columns)

  if hparams.raw_audio:
    spec = tf.squeeze(spec)
    spec /= tf.expand_dims(tf.reduce_max(spec, axis=1), axis=1)
    tf.summary.audio(
        name, tf.squeeze(spec), hparams.samples_per_second, max_outputs=b)
  else:
    if image:
      if b % columns != 0:
        rows = np.floor(np.sqrt(b))
        columns = rows
      else:
        rows = b / columns
      tf.summary.image("Mag/%s" % name,
                       form_image_grid(spec[:b, :, :, :1], [rows, columns],
                                       [n_freq, n_time], 1))
      if phase:
        tf.summary.image("Phase/%s" % name,
                         form_image_grid(spec[:b, :, :, 1:], [rows, columns],
                                         [n_freq, n_time], 1))
    if audio:
      tf.summary.audio(
          name,
          tf_ispecgram(
              spec,
              n_fft=hparams.n_fft,
              hop_length=hparams.hop_length,
              mask=hparams.mask,
              log_mag=hparams.log_mag,
              pad=hparams.pad,
              re_im=hparams.re_im,
              dphase=hparams.dphase,
              mag_only=hparams.mag_only),
          hparams.samples_per_second,
          max_outputs=b)


def calculate_softmax_and_summaries(logits, one_hot_labels, name):
  """Calculate the softmax cross entropy loss and associated summaries.

  Args:
    logits: Tensor of logits, first dimension is batch size.
    one_hot_labels: Tensor of one hot encoded categorical labels. First
      dimension is batch size.
    name: Name to use as prefix for summaries.

  Returns:
    loss: Dimensionless tensor representing the mean negative
      log-probability of the true class.
  """
  loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=one_hot_labels)
  loss = tf.reduce_mean(loss)
  softmax_summaries(loss, logits, one_hot_labels, name)
  return loss


def calculate_sparse_softmax_and_summaries(logits, labels, name):
  """Calculate the softmax cross entropy loss and associated summaries.

  Args:
    logits: Tensor of logits, first dimension is batch size.
    labels: Tensor of categorical labels [ints]. First
      dimension is batch size.
    name: Name to use as prefix for summaries.

  Returns:
    loss: Dimensionless tensor representing the mean negative
      log-probability of the true class.
  """
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)
  loss = tf.reduce_mean(loss)
  softmax_summaries(loss, logits, labels, name)
  return loss


def softmax_summaries(loss, logits, one_hot_labels, name="softmax"):
  """Create the softmax summaries for this cross entropy loss.

  Args:
    loss: Cross-entropy loss.
    logits: The [batch_size, classes] float tensor representing the logits.
    one_hot_labels: The float tensor representing actual class ids. If this is
      [batch_size, classes], then we take the argmax of it first.
    name: Prepended to summary scope.
  """
  tf.summary.scalar(name + "_loss", loss)

  one_hot_labels = tf.cond(
      tf.equal(tf.rank(one_hot_labels),
               2), lambda: tf.to_int32(tf.argmax(one_hot_labels, 1)),
      lambda: tf.to_int32(one_hot_labels))

  in_top_1 = tf.nn.in_top_k(logits, one_hot_labels, 1)
  tf.summary.scalar(name + "_precision@1",
                    tf.reduce_mean(tf.to_float(in_top_1)))
  in_top_5 = tf.nn.in_top_k(logits, one_hot_labels, 5)
  tf.summary.scalar(name + "_precision@5",
                    tf.reduce_mean(tf.to_float(in_top_5)))


def calculate_l2_and_summaries(predicted_vectors, true_vectors, name):
  """Calculate L2 loss and associated summaries.

  Args:
    predicted_vectors: Tensor of predictions, first dimension is batch size.
    true_vectors: Tensor of labels, first dimension is batch size.
    name: Name to use as prefix for summaries.

  Returns:
    loss: Dimensionless tensor representing the mean euclidean distance
      between true and predicted.
  """
  loss = tf.reduce_mean((predicted_vectors - true_vectors)**2)
  tf.summary.scalar(name + "_loss", loss, name="loss")
  tf.summary.scalar(
      name + "_prediction_mean_squared_norm",
      tf.reduce_mean(tf.nn.l2_loss(predicted_vectors)),
      name=name + "_prediction_mean_squared_norm")
  tf.summary.scalar(
      name + "_label_mean_squared_norm",
      tf.reduce_mean(tf.nn.l2_loss(true_vectors)),
      name=name + "_label_mean_squared_norm")
  return loss


def frequency_weighted_cost_mask(peak=10.0, hz_flat=1000, sr=16000, n_fft=512):
  """Calculates a mask to weight lower frequencies higher.

  Piecewise linear approximation. Assumes magnitude is in log scale.
  Args:
    peak: Cost increase at 0 Hz.
    hz_flat: Hz at which cost increase is 0.
    sr: Sample rate.
    n_fft: FFT size.

  Returns:
    Constant tensor [1, N_freq, 1] of cost weighting.
  """
  n = int(n_fft / 2)
  cutoff = np.where(
      librosa.core.fft_frequencies(sr=sr, n_fft=n_fft) >= hz_flat)[0][0]
  mask = np.concatenate([np.linspace(peak, 1.0, cutoff), np.ones(n - cutoff)])
  return tf.constant(mask[np.newaxis, :, np.newaxis], dtype=tf.float32)


#---------------------------------------------------
# Neural Nets
#---------------------------------------------------
def pitch_embeddings(batch,
                     timesteps=1,
                     n_pitches=128,
                     dim_embedding=128,
                     reuse=False):
  """Get a embedding of each pitch note.

  Args:
    batch: NSynthDataset batch dictionary.
    timesteps: Number of timesteps to replicate across.
    n_pitches: Number of one-hot embeddings.
    dim_embedding: Dimension of linear projection of one-hot encoding.
    reuse: Reuse variables.

  Returns:
    embedding: A tensor of shape [batch_size, 1, timesteps, dim_embedding].
  """
  batch_size = batch["pitch"].get_shape().as_list()[0]
  with tf.variable_scope("PitchEmbedding", reuse=reuse):
    w = tf.get_variable(
        name="embedding_weights",
        shape=[n_pitches, dim_embedding],
        initializer=tf.random_normal_initializer())
    one_hot_pitch = tf.reshape(batch["pitch"], [batch_size])
    one_hot_pitch = tf.one_hot(one_hot_pitch, depth=n_pitches)
    embedding = tf.matmul(one_hot_pitch, w)
    embedding = tf.reshape(embedding, [batch_size, 1, 1, dim_embedding])
    if timesteps > 1:
      embedding = tf.tile(embedding, [1, 1, timesteps, 1])
    return embedding


def slim_batchnorm_arg_scope(is_training, activation_fn=None):
  """Create a scope for applying BatchNorm in slim.

  This scope also applies Glorot initializiation to convolutional weights.
  Args:
    is_training: Whether this is a training run.
    activation_fn: Whether we apply an activation_fn to the convolution result.

  Returns:
    scope: Use this scope to automatically apply BatchNorm and Xavier Init to
      slim.conv2d and slim.fully_connected.
  """
  batch_norm_params = {
      "is_training": is_training,
      "decay": 0.999,
      "epsilon": 0.001,
      "variables_collections": {
          "beta": None,
          "gamma": None,
          "moving_mean": "moving_vars",
          "moving_variance": "moving_vars",
      }
  }

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
      weights_initializer=slim.initializers.xavier_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params) as scope:
    return scope


def conv2d(x,
           kernel_size,
           stride,
           channels,
           is_training,
           scope="conv2d",
           batch_norm=False,
           residual=False,
           gated=False,
           activation_fn=tf.nn.relu,
           resize=False,
           transpose=False,
           stacked_layers=1):
  """2D-Conv with optional batch_norm, gating, residual.

  Args:
    x: Tensor input [MB, H, W, CH].
    kernel_size: List [H, W].
    stride: List [H, W].
    channels: Int, output channels.
    is_training: Whether to collect stats for BatchNorm.
    scope: Enclosing scope name.
    batch_norm: Apply batch normalization
    residual: Residual connections, have stacked_layers >= 2.
    gated: Gating ala Wavenet.
    activation_fn: Nonlinearity function.
    resize: On transposed convolution, do ImageResize instead of conv_transpose.
    transpose: Use conv_transpose instead of conv.
    stacked_layers: Number of layers before a residual connection.

  Returns:
    x: Tensor output.
  """
  # For residual
  x0 = x
  # Choose convolution function
  conv_fn = slim.conv2d_transpose if transpose else slim.conv2d
  # Double output channels for gates
  num_outputs = channels * 2 if gated else channels
  normalizer_fn = slim.batch_norm if batch_norm else None

  with tf.variable_scope(scope + "_Layer"):
    # Apply a stack of convolutions Before adding residual
    for layer_idx in range(stacked_layers):
      with slim.arg_scope(
          slim_batchnorm_arg_scope(is_training, activation_fn=None)):
        # Use interpolation to upsample instead of conv_transpose
        if transpose and resize:
          unused_mb, h, w, unused_ch = x.get_shape().as_list()
          x = tf.image.resize_images(
              x, size=[h * stride[0], w * stride[1]], method=0)
          stride_conv = [1, 1]
        else:
          stride_conv = stride

        x = conv_fn(
            inputs=x,
            stride=stride_conv,
            kernel_size=kernel_size,
            num_outputs=num_outputs,
            normalizer_fn=normalizer_fn,
            biases_initializer=tf.zeros_initializer(),
            scope=scope)

        if gated:
          with tf.variable_scope("Gated"):
            x1, x2 = x[:, :, :, :channels], x[:, :, :, channels:]
            if activation_fn:
              x1, x2 = activation_fn(x1), tf.sigmoid(x2)
            else:
              x2 = tf.sigmoid(x2)
            x = x1 * x2

        # Apply residual to last layer  before the last nonlinearity
        if residual and (layer_idx == stacked_layers - 1):
          with tf.variable_scope("Residual"):
            # Don't upsample residual in time
            if stride[0] == 1 and stride[1] == 1:
              channels_in = x0.get_shape().as_list()[-1]
              # Make n_channels match for residual
              if channels != channels_in:
                x0 = slim.conv2d(
                    inputs=x0,
                    stride=[1, 1],
                    kernel_size=[1, 1],
                    num_outputs=channels,
                    normalizer_fn=None,
                    activation_fn=None,
                    biases_initializer=tf.zeros_initializer,
                    scope=scope + "_residual")
                x += x0
              else:
                x += x0
        if activation_fn and not gated:
          x = activation_fn(x)
    return x


def leaky_relu(leak=0.1):
  """Leaky ReLU activation function.

  Args:
    leak: float. Slope for the negative part of the leaky ReLU function.
        Defaults to 0.1.

  Returns:
    A lambda computing the leaky ReLU function with the specified slope.
  """
  return lambda x: tf.maximum(x, leak * x)


def causal_linear(x, n_inputs, n_outputs, name, filter_length, rate,
                  batch_size):
  """Applies dilated convolution using queues.

  Assumes a filter_length of 3.

  Args:
    x: The [mb, time, channels] tensor input.
    n_inputs: The input number of channels.
    n_outputs: The output number of channels.
    name: The variable scope to provide to W and biases.
    filter_length: The length of the convolution, assumed to be 3.
    rate: The rate or dilation
    batch_size: Non-symbolic value for batch_size.

  Returns:
    y: The output of the operation
    (init_1, init_2): Initialization operations for the queues
    (push_1, push_2): Push operations for the queues
  """
  assert filter_length == 3

  # create queue
  q_1 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, 1, n_inputs))
  q_2 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, 1, n_inputs))
  init_1 = q_1.enqueue_many(tf.zeros((rate, batch_size, 1, n_inputs)))
  init_2 = q_2.enqueue_many(tf.zeros((rate, batch_size, 1, n_inputs)))
  state_1 = q_1.dequeue()
  push_1 = q_1.enqueue(x)
  state_2 = q_2.dequeue()
  push_2 = q_2.enqueue(state_1)

  # get pretrained weights
  w = tf.get_variable(
      name=name + "/W",
      shape=[1, filter_length, n_inputs, n_outputs],
      dtype=tf.float32)
  b = tf.get_variable(
      name=name + "/biases", shape=[n_outputs], dtype=tf.float32)
  w_q_2 = tf.slice(w, [0, 0, 0, 0], [-1, 1, -1, -1])
  w_q_1 = tf.slice(w, [0, 1, 0, 0], [-1, 1, -1, -1])
  w_x = tf.slice(w, [0, 2, 0, 0], [-1, 1, -1, -1])

  # perform op w/ cached states
  y = tf.nn.bias_add(
      tf.matmul(state_2[:, 0, :], w_q_2[0][0]) + tf.matmul(
          state_1[:, 0, :], w_q_1[0][0]) + tf.matmul(x[:, 0, :], w_x[0][0]), b)

  y = tf.expand_dims(y, 1)
  return y, (init_1, init_2), (push_1, push_2)


def linear(x, n_inputs, n_outputs, name):
  """Simple linear layer.

  Args:
    x: The [mb, time, channels] tensor input.
    n_inputs: The input number of channels.
    n_outputs: The output number of channels.
    name: The variable scope to provide to W and biases.

  Returns:
    y: The output of the operation.
  """
  w = tf.get_variable(
      name=name + "/W", shape=[1, 1, n_inputs, n_outputs], dtype=tf.float32)
  b = tf.get_variable(
      name=name + "/biases", shape=[n_outputs], dtype=tf.float32)
  y = tf.nn.bias_add(tf.matmul(x[:, 0, :], w[0][0]), b)
  y = tf.expand_dims(y, 1)
  return y
