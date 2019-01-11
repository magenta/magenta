# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Utils for classififying pitch, instrument, and qualities for NSynth."""

import collections


import numpy as np
import scipy
import tensorflow as tf

from magenta.models.nsynth import reader
from magenta.models.nsynth import utils

slim = tf.contrib.slim
namedtuple = collections.namedtuple
OrderedDict = collections.OrderedDict

BATCH = namedtuple("batch", ["audio", "spectrogram",])


def _conv2d(x,
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
    preact_x, x: pre and post activation tensor output.
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
          utils.slim_batchnorm_arg_scope(
              is_training, activation_fn=None)):
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
            # Don"t upsample residual in time
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
        preact_x = x
        if activation_fn and not gated:
          x = activation_fn(x)
    return preact_x, x


def _fit(x, hparams, is_training=True, reuse=False):
  """Classification network.

  Args:
    x: Tensor. The observed variables.
    hparams: HParams. Hyperparameters.
    is_training: bool. Whether batch normalization should be computed in
        training mode. Defaults to True.
    reuse: bool. Whether the variable scope should be reused.
        Defaults to False.

  Returns:
    The output of the encoder, i.e. a synthetic z computed from x.
  """
  end_points = OrderedDict()
  with tf.variable_scope("encoder", reuse=reuse):
    preact_h, h = _conv2d(
        x, [5, 5], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="0")
    end_points["preact_layer_0"] = preact_h
    end_points["layer_0"] = h
    preact_h, h = _conv2d(
        h, [4, 4], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="1")
    end_points["preact_layer_1"] = preact_h
    end_points["layer_1"] = h
    preact_h, h = _conv2d(
        h, [4, 4], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="2")
    end_points["preact_layer_2"] = preact_h
    end_points["layer_2"] = h
    preact_h, h = _conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="3")
    end_points["preact_layer_3"] = preact_h
    end_points["layer_3"] = h
    preact_h, h = _conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="4")
    end_points["preact_layer_4"] = preact_h
    end_points["layer_4"] = h
    preact_h, h = _conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="5")
    end_points["preact_layer_5"] = preact_h
    end_points["layer_5"] = h
    preact_h, h = _conv2d(
        h, [4, 4], [2, 2],
        512,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="6")
    end_points["preact_layer_6"] = preact_h
    end_points["layer_6"] = h
    preact_h, h = _conv2d(
        h, [4, 4], [2, 2],
        512,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="7")
    end_points["preact_layer_7"] = preact_h
    end_points["layer_7"] = h
    preact_h, h = _conv2d(
        h, [4, 4], [2, 1],
        512,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="7_1")
    end_points["preact_layer_7_1"] = preact_h
    end_points["layer_7_1"] = h
    preact_h, h = _conv2d(
        h, [1, 1], [1, 1],
        1024,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="8")
    end_points["preact_layer_8"] = preact_h
    end_points["layer_8"] = h

    if hparams.join_family_source:
      total = (hparams.n_pitches + hparams.n_qualities +
               hparams.n_instrument_families * hparams.n_instrument_sources)
    else:
      total = (hparams.n_pitches + hparams.n_qualities +
               hparams.n_instrument_families + hparams.n_instrument_sources)

    _, logits = _conv2d(
        h, [1, 1], [1, 1],
        total,
        is_training,
        activation_fn=None,
        batch_norm=True,
        scope="z")
    logits = tf.reshape(logits, [hparams.batch_size, total])
    end_points["logits"] = logits
  return logits, end_points


def _get_spectrogram(audio,
                     batch_size=16,
                     n_fft=512,
                     hop_length=256,
                     mask=True,
                     log_mag=True,
                     use_cqt=False,
                     re_im=False,
                     dphase=True,
                     mag_only=False,
                     pad=True):
  """Gets a padded spectrograms from the input audio."""
  if hop_length and n_fft:
    specgram = utils.tf_specgram(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        mask=mask,
        log_mag=log_mag,
        re_im=re_im,
        dphase=dphase,
        mag_only=mag_only)
    if use_cqt:
      shape = [batch_size] + [252, 1001, 2]
    else:
      shape = [batch_size] +  reader.SPECGRAM_REGISTRY[(
          n_fft, hop_length)]
      if mag_only:
        shape[-1] = 1
    specgram = tf.reshape(specgram, shape)

    if pad:
      # Pad and crop specgram to 256x256
      num_padding = 2**int(np.ceil(np.log(shape[2]) / np.log(2))) - shape[2]
      specgram = tf.reshape(specgram, shape)
      specgram = tf.pad(specgram, [[0, 0], [0, 0], [0, num_padding], [0, 0]])
      specgram = tf.slice(specgram, [0, 0, 0, 0], [-1, shape[1] - 1, -1, -1])
  return specgram


def _get_batch_audio(audio, hparams):
  """Get batch of audio and spectograms."""
  spectrogram = _get_spectrogram(audio,
                                 hparams.batch_size,
                                 hparams.n_fft,
                                 hparams.hop_length,
                                 hparams.mask,
                                 hparams.log_mag,
                                 hparams.use_cqt,
                                 hparams.re_im,
                                 hparams.dphase,
                                 hparams.mag_only,
                                 hparams.pad)
  batch = BATCH(audio=audio,
                spectrogram=spectrogram)
  return batch


def set_pitch_hparams(batch_size, hparams):
  """Set the hparams for the pitch-detection model."""
  hparams.batch_size = batch_size
  hparams.n_fft = 1024
  hparams.mag_only = True
  hparams.join_family_source = False
  hparams.n_qualities = 10
  hparams.n_instrument_families = 11
  hparams.n_instrument_sources = 3
  return hparams


def get_batch_entropy(samples):
  return [scipy.stats.entropy(sample) for sample in samples]


def get_pitch_accuracy(emp_pitches, real_pitches, threshold=1):
  return np.mean(np.abs(emp_pitches-real_pitches) <= threshold)


def get_pitch_qualities(audio, hparams):
  """For given audio, returns pitch and qualities from a pre-trained model.

  Args:
    audio: Sample for which we need the pitch/qualities.
    hparams: Hyperparameters.

  Returns:
    pitch: Pitch predicted by trained model.
    pitch_logits : Returns the logits corresponding to pitch classifier.
    qualities : Qualities predicted by trained model.
  """
  batch = _get_batch_audio(audio, hparams)

  x = batch.spectrogram
  # Define the model
  with tf.name_scope("Model"):
    logits, _ = _fit(x, hparams, is_training=False, reuse=tf.AUTO_REUSE)

  pitch_logits = logits[:, :hparams.n_pitches]

  qualities_logits = logits[:, hparams.n_pitches:hparams.n_pitches +
                            hparams.n_qualities]

  pitch = tf.argmax(pitch_logits, 1)
  qualities = tf.to_int32(qualities_logits > 0)

  return pitch, pitch_logits, qualities


def get_features(audio, feature_layer, hparams):
  """Returns features for input batch.

  Args:
    audio:
    feature_layer:
    hparams :

  Returns:
    features:
  """
  batch = _get_batch_audio(audio, hparams)

  x = batch.spectrogram
  with tf.name_scope("Model"):
    _, end_points = _fit(x, hparams, is_training=False, reuse=tf.AUTO_REUSE)
  return end_points[feature_layer]
