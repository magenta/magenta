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

# Lint as: python3
"""Nerual network components.

This library containts nerual network components in either raw TF or sonnet
Module.
"""

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


def product_two_guassian_pdfs(mu_1, sigma_1, mu_2, sigma_2):
  """Product of two Guasssian PDF."""
  # https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
  sigma_1_square = tf.square(sigma_1)
  sigma_2_square = tf.square(sigma_2)
  mu = (mu_1 * sigma_2_square + mu_2 * sigma_1_square) / (
      sigma_1_square + sigma_2_square)
  sigma_square = (sigma_1_square * sigma_2_square) / (
      sigma_1_square + sigma_2_square)
  sigma = tf.sqrt(sigma_square)
  return mu, sigma


def tf_batch_image(b, mb=36):
  """Turn a batch of images into a single image mosaic."""
  b_shape = b.get_shape().as_list()
  rows = int(np.ceil(np.sqrt(mb)))
  cols = rows
  diff = rows * cols - mb
  b = tf.concat(
      [b[:mb], tf.zeros([diff, b_shape[1], b_shape[2], b_shape[3]])], axis=0)
  tmp = tf.reshape(b, [-1, cols * b_shape[1], b_shape[2], b_shape[3]])
  img = tf.concat([tmp[i:i + 1] for i in range(rows)], axis=2)
  return img


class EncoderMNIST(snt.AbstractModule):
  """MLP encoder for MNIST."""

  def __init__(self, n_latent=64, layers=(1024,) * 3, name='encoder'):
    super(EncoderMNIST, self).__init__(name=name)
    self.n_latent = n_latent
    self.layers = layers

  def _build(self, x):
    for size in self.layers:
      x = tf.nn.relu(snt.Linear(size)(x))
    pre_z = snt.Linear(2 * self.n_latent)(x)
    mu = pre_z[:, :self.n_latent]
    sigma = tf.nn.softplus(pre_z[:, self.n_latent:])
    return mu, sigma


class DecoderMNIST(snt.AbstractModule):
  """MLP decoder for MNIST."""

  def __init__(self, layers=(1024,) * 3, n_out=784, name='decoder'):
    super(DecoderMNIST, self).__init__(name=name)
    self.layers = layers
    self.n_out = n_out

  def _build(self, x):
    for size in self.layers:
      x = tf.nn.relu(snt.Linear(size)(x))
    logits = snt.Linear(self.n_out)(x)
    return logits


class EncoderConv(snt.AbstractModule):
  """ConvNet encoder for CelebA."""

  def __init__(self,
               n_latent,
               layers=((256, 5, 2), (512, 5, 2), (1024, 3, 2), (2048, 3, 2)),
               padding_linear_layers=None,
               name='encoder'):
    super(EncoderConv, self).__init__(name=name)
    self.n_latent = n_latent
    self.layers = layers
    self.padding_linear_layers = padding_linear_layers or []

  def _build(self, x):
    h = x
    for unused_i, l in enumerate(self.layers):
      h = tf.nn.relu(snt.Conv2D(l[0], l[1], l[2])(h))

    h_shape = h.get_shape().as_list()
    h = tf.reshape(h, [-1, h_shape[1] * h_shape[2] * h_shape[3]])
    for _, l in enumerate(self.padding_linear_layers):
      h = snt.Linear(l)(h)
    pre_z = snt.Linear(2 * self.n_latent)(h)
    mu = pre_z[:, :self.n_latent]
    sigma = tf.nn.softplus(pre_z[:, self.n_latent:])
    return mu, sigma


class DecoderConv(snt.AbstractModule):
  """ConvNet decoder for CelebA."""

  def __init__(self,
               layers=((2048, 4, 4), (1024, 3, 2), (512, 3, 2), (256, 5, 2),
                       (3, 5, 2)),
               padding_linear_layers=None,
               name='decoder'):
    super(DecoderConv, self).__init__(name=name)
    self.layers = layers
    self.padding_linear_layers = padding_linear_layers or []

  def _build(self, x):
    for i, l in enumerate(self.padding_linear_layers):
      x = snt.Linear(l)(x)
    for i, l in enumerate(self.layers):
      if i == 0:
        h = snt.Linear(l[1] * l[2] * l[0])(x)
        h = tf.reshape(h, [-1, l[1], l[2], l[0]])
      elif i == len(self.layers) - 1:
        h = snt.Conv2DTranspose(l[0], None, l[1], l[2])(h)
      else:
        h = tf.nn.relu(snt.Conv2DTranspose(l[0], None, l[1], l[2])(h))
    logits = h
    return logits


class ClassifierConv(snt.AbstractModule):
  """ConvNet classifier."""

  def __init__(self,
               output_size,
               layers=((256, 5, 2), (256, 3, 1), (512, 5, 2), (512, 3, 1),
                       (1024, 3, 2), (2048, 3, 2)),
               name='encoder'):
    super(ClassifierConv, self).__init__(name=name)
    self.output_size = output_size
    self.layers = layers

  def _build(self, x):
    h = x
    for unused_i, l in enumerate(self.layers):
      h = tf.nn.relu(snt.Conv2D(l[0], l[1], l[2])(h))

    h_shape = h.get_shape().as_list()
    h = tf.reshape(h, [-1, h_shape[1] * h_shape[2] * h_shape[3]])
    logits = snt.Linear(self.output_size)(h)
    return logits


class GFull(snt.AbstractModule):
  """MLP (Full layers) generator."""

  def __init__(self, n_latent, layers=(2048,) * 4, name='generator'):
    super(GFull, self).__init__(name=name)
    self.layers = layers
    self.n_latent = n_latent

  def _build(self, z):
    x = z
    for l in self.layers:
      x = tf.nn.relu(snt.Linear(l)(x))
    x = snt.Linear(2 * self.n_latent)(x)
    dz = x[:, :self.n_latent]
    gates = tf.nn.sigmoid(x[:, self.n_latent:])
    z_prime = (1 - gates) * z + gates * dz
    return z_prime


class DFull(snt.AbstractModule):
  """MLP (Full layers) discriminator/classifier."""

  def __init__(self, output_size=1, layers=(2048,) * 4, name='D'):
    super(DFull, self).__init__(name=name)
    self.layers = layers
    self.output_size = output_size

  def _build(self, x):
    for l in self.layers:
      x = tf.nn.relu(snt.Linear(l)(x))
    logits = snt.Linear(self.output_size)(x)
    return logits
