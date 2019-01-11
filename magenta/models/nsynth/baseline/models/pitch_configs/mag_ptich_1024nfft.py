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
"""Pitch classification."""

import tensorflow as tf
from magenta.models.nsynth import utils

slim = tf.contrib.slim


def fit(x, hparams, is_training=True, reuse=False):
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

    logits = utils.conv2d(
        h, [1, 1], [1, 1],
        hparams.n_pitches,
        is_training,
        activation_fn=None,
        batch_norm=True,
        scope="z")
    logits = tf.reshape(logits, [hparams.batch_size, hparams.n_pitches])
  return logits
