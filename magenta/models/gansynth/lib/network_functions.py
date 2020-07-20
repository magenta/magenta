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

"""Generator and discriminator functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.gansynth.lib import data_normalizer
from magenta.models.gansynth.lib import layers
from magenta.models.gansynth.lib import networks
import tensorflow.compat.v1 as tf


def _num_filters_fn(block_id, **kwargs):
  """Computes number of filters of block `block_id`."""
  return networks.num_filters(block_id, kwargs['fmap_base'],
                              kwargs['fmap_decay'], kwargs['fmap_max'])


def generator_fn_specgram(inputs, **kwargs):
  """Builds generator network."""
  # inputs = (noises, one_hot_labels)
  with tf.variable_scope('generator_cond'):
    z = tf.concat(inputs, axis=1)
  if kwargs['to_rgb_activation'] == 'tanh':
    to_rgb_activation = tf.tanh
  elif kwargs['to_rgb_activation'] == 'linear':
    to_rgb_activation = lambda x: x
  fake_images, end_points = networks.generator(
      z,
      kwargs['progress'],
      lambda block_id: _num_filters_fn(block_id, **kwargs),  # pylint:disable=unnecessary-lambda
      kwargs['resolution_schedule'],
      num_blocks=kwargs['num_blocks'],
      kernel_size=kwargs['kernel_size'],
      colors=2,
      to_rgb_activation=to_rgb_activation,
      simple_arch=kwargs['simple_arch'])
  shape = fake_images.shape
  normalizer = data_normalizer.registry[kwargs['data_normalizer']](kwargs)
  fake_images = normalizer.denormalize_op(fake_images)
  fake_images.set_shape(shape)
  return fake_images, end_points


def discriminator_fn_specgram(images, **kwargs):
  """Builds discriminator network."""
  shape = images.shape
  normalizer = data_normalizer.registry[kwargs['data_normalizer']](kwargs)
  images = normalizer.normalize_op(images)
  images.set_shape(shape)
  logits, end_points = networks.discriminator(
      images,
      kwargs['progress'],
      lambda block_id: _num_filters_fn(block_id, **kwargs),  # pylint:disable=unnecessary-lambda
      kwargs['resolution_schedule'],
      num_blocks=kwargs['num_blocks'],
      kernel_size=kwargs['kernel_size'],
      simple_arch=kwargs['simple_arch'])
  with tf.variable_scope('discriminator_cond'):
    x = tf.layers.flatten(end_points['last_conv'])
    end_points['classification_logits'] = layers.custom_dense(
        x=x, units=kwargs['num_tokens'], scope='classification_logits')
  return logits, end_points


g_fn_registry = {
    'specgram': generator_fn_specgram,
}


d_fn_registry = {
    'specgram': discriminator_fn_specgram,
}
