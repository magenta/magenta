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

"""Autoencoder config.

All configs should have encode() and decode().
"""

from magenta.models.nsynth import utils
import tensorflow.compat.v1 as tf

config_hparams = dict(
    num_latent=1984,
    batch_size=8,
    mag_only=True,
    n_fft=1024,
    fw_loss_coeff=10.0,
    fw_loss_cutoff=4000,)


def encode(x, hparams, is_training=True, reuse=False):
  """Autoencoder encoder network.

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
  with tf.variable_scope("encoder", reuse=reuse):
    h = utils.conv2d(
        x, [5, 5], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="0")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="1")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="2")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="3")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="4")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="5")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        512,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="6")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        512,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="7")
    h = utils.conv2d(
        h, [4, 4], [2, 1],
        512,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="7_1")
    h = utils.conv2d(
        h, [1, 1], [1, 1],
        1024,
        is_training,
        activation_fn=utils.leaky_relu(),
        batch_norm=True,
        scope="8")

    z = utils.conv2d(
        h, [1, 1], [1, 1],
        hparams.num_latent,
        is_training,
        activation_fn=None,
        batch_norm=True,
        scope="z")
  return z


def decode(z, batch, hparams, is_training=True, reuse=False):
  """Autoencoder decoder network.

  Args:
    z: Tensor. The latent variables.
    batch: NSynthReader batch for pitch information.
    hparams: HParams. Hyperparameters (unused).
    is_training: bool. Whether batch normalization should be computed in
        training mode. Defaults to True.
    reuse: bool. Whether the variable scope should be reused.
        Defaults to False.

  Returns:
    The output of the decoder, i.e. a synthetic x computed from z.
  """
  del hparams
  with tf.variable_scope("decoder", reuse=reuse):
    z_pitch = utils.pitch_embeddings(batch, reuse=reuse)
    z = tf.concat([z, z_pitch], 3)

    h = utils.conv2d(
        z, [1, 1], [1, 1],
        1024,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="0")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        512,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="1")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        512,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="2")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="3")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="4")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        256,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="5")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="6")
    h = utils.conv2d(
        h, [4, 4], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="7")
    h = utils.conv2d(
        h, [5, 5], [2, 2],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="8")
    h = utils.conv2d(
        h, [5, 5], [2, 1],
        128,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="8_1")

    xhat = utils.conv2d(
        h, [1, 1], [1, 1],
        1,
        is_training,
        activation_fn=tf.nn.sigmoid,
        batch_norm=False,
        scope="mag")
  return xhat
