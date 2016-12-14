# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Learning-related functions for style transfer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports

import numpy as np
import tensorflow as tf

from magenta.models.image_stylization import vgg

slim = tf.contrib.slim


def precompute_gram_matrices(image, final_endpoint='fc8'):
  """Pre-computes the Gram matrices on a given image.

  Args:
    image: 4-D tensor. Input (batch of) image(s).
    final_endpoint: str, name of the final layer to compute Gram matrices for.
        Defaults to 'fc8'.

  Returns:
    dict mapping layer names to their corresponding Gram matrices.
  """
  with tf.Session() as session:
    end_points = vgg.vgg_16(image, final_endpoint=final_endpoint)
    tf.train.Saver(slim.get_variables('vgg_16')).restore(
        session, vgg.checkpoint_file())
    return dict([(key, _gram_matrix(value).eval())
                 for key, value in end_points.iteritems()])


def total_loss(inputs, stylized_inputs, style_gram_matrices, content_weights,
               style_weights, reuse=False):
  """Computes the total loss function.

  The total loss function is composed of a content, a style and a total
  variation term.

  Args:
    inputs: Tensor. The input images.
    stylized_inputs: Tensor. The stylized input images.
    style_gram_matrices: dict mapping layer names to their corresponding
        Gram matrices.
    content_weights: dict mapping layer names to their associated content loss
        weight. Keys that are missing from the dict won't have their content
        loss computed.
    style_weights: dict mapping layer names to their associated style loss
        weight. Keys that are missing from the dict won't have their style
        loss computed.
    reuse: bool. Whether to reuse model parameters. Defaults to False.

  Returns:
    Tensor for the total loss, dict mapping loss names to losses.
  """
  # Propagate the the input and its stylized version through VGG16
  end_points = vgg.vgg_16(inputs, reuse=reuse)
  stylized_end_points = vgg.vgg_16(stylized_inputs, reuse=True)

  # Compute the content loss
  total_content_loss, content_loss_dict = content_loss(
      end_points, stylized_end_points, content_weights)

  # Compute the style loss
  total_style_loss, style_loss_dict = style_loss(
      style_gram_matrices, stylized_end_points, style_weights)

  # Compute the total loss
  loss = total_content_loss + total_style_loss

  loss_dict = {'total_loss': loss}
  loss_dict.update(content_loss_dict)
  loss_dict.update(style_loss_dict)

  return loss, loss_dict


def content_loss(end_points, stylized_end_points, content_weights):
  """Content loss.

  Args:
    end_points: dict mapping VGG16 layer names to their corresponding Tensor
        value for the original input.
    stylized_end_points: dict mapping VGG16 layer names to their corresponding
        Tensor value for the stylized input.
    content_weights: dict mapping layer names to their associated content loss
        weight. Keys that are missing from the dict won't have their content
        loss computed.

  Returns:
    Tensor for the total content loss, dict mapping loss names to losses.
  """
  total_content_loss = np.float32(0.0)
  content_loss_dict = {}

  for name, weight in content_weights.iteritems():
    # Reducing over all but the batch axis before multiplying with the content
    # weights allows to use multiple sets of content weights in a single batch.
    loss = tf.reduce_mean(
        (end_points[name] - stylized_end_points[name]) ** 2,
        [1, 2, 3])
    weighted_loss = tf.reduce_mean(weight * loss)
    loss = tf.reduce_mean(loss)

    content_loss_dict['content_loss/' + name] = loss
    content_loss_dict['weighted_content_loss/' + name] = weighted_loss
    total_content_loss += weighted_loss

  content_loss_dict['total_content_loss'] = total_content_loss

  return total_content_loss, content_loss_dict


def style_loss(style_gram_matrices, end_points, style_weights):
  """Style loss.

  Args:
    style_gram_matrices: dict mapping VGG16 layer names to their corresponding
        gram matrix for the style image.
    end_points: dict mapping VGG16 layer names to their corresponding
        Tensor value for the stylized input.
    style_weights: dict mapping layer names to their associated style loss
        weight. Keys that are missing from the dict won't have their style
        loss computed.

  Returns:
    Tensor for the total style loss, dict mapping loss names to losses.
  """
  total_style_loss = np.float32(0.0)
  style_loss_dict = {}

  for name, weight in style_weights.iteritems():
    # Reducing over all but the batch axis before multiplying with the style
    # weights allows to use multiple sets of style weights in a single batch.
    loss = tf.reduce_mean(
        (_gram_matrix(end_points[name]) - style_gram_matrices[name]) ** 2,
        [1, 2])
    weighted_style_loss = tf.reduce_mean(weight * loss)
    loss = tf.reduce_mean(loss)

    style_loss_dict['style_loss/' + name] = loss
    style_loss_dict['weighted_style_loss/' + name] = weighted_style_loss
    total_style_loss += weighted_style_loss

  style_loss_dict['total_style_loss'] = total_style_loss

  return total_style_loss, style_loss_dict


def _gram_matrix(feature_maps):
  """Computes the Gram matrix for a set of feature maps."""
  batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
  denominator = tf.to_float(height * width)
  feature_maps = tf.reshape(
      feature_maps, tf.stack([batch_size, height * width, channels]))
  matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
  return matrix / denominator
