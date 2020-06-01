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

"""Loss methods for real-time arbitrary image stylization model."""
from magenta.models.image_stylization import learning as learning_utils
from magenta.models.image_stylization import vgg
import numpy as np
import tensorflow.compat.v1 as tf


def total_loss(content_inputs, style_inputs, stylized_inputs, content_weights,
               style_weights, total_variation_weight, reuse=False):
  """Computes the total loss function.

  The total loss function is composed of a content, a style and a total
  variation term.

  Args:
    content_inputs: Tensor. The input images.
    style_inputs: Tensor. The input images.
    stylized_inputs: Tensor. The stylized input images.
    content_weights: dict mapping layer names to their associated content loss
        weight. Keys that are missing from the dict won't have their content
        loss computed.
    style_weights: dict mapping layer names to their associated style loss
        weight. Keys that are missing from the dict won't have their style
        loss computed.
    total_variation_weight: float. Coefficient for the total variation part of
        the loss.
    reuse: bool. Whether to reuse model parameters. Defaults to False.

  Returns:
    Tensor for the total loss, dict mapping loss names to losses.
  """
  # Propagate the input and its stylized version through VGG16.
  with tf.name_scope('content_endpoints'):
    content_end_points = vgg.vgg_16(content_inputs, reuse=reuse)
  with tf.name_scope('style_endpoints'):
    style_end_points = vgg.vgg_16(style_inputs, reuse=True)
  with tf.name_scope('stylized_endpoints'):
    stylized_end_points = vgg.vgg_16(stylized_inputs, reuse=True)

  # Compute the content loss
  with tf.name_scope('content_loss'):
    total_content_loss, content_loss_dict = content_loss(
        content_end_points, stylized_end_points, content_weights)

  # Compute the style loss
  with tf.name_scope('style_loss'):
    total_style_loss, style_loss_dict = style_loss(
        style_end_points, stylized_end_points, style_weights)

  # Compute the total variation loss
  with tf.name_scope('total_variation_loss'):
    tv_loss, total_variation_loss_dict = learning_utils.total_variation_loss(
        stylized_inputs, total_variation_weight)

  # Compute the total loss
  with tf.name_scope('total_loss'):
    loss = total_content_loss + total_style_loss + tv_loss

  loss_dict = {'total_loss': loss}
  loss_dict.update(content_loss_dict)
  loss_dict.update(style_loss_dict)
  loss_dict.update(total_variation_loss_dict)

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

  for name, weight in content_weights.items():
    loss = tf.reduce_mean(
        (end_points[name] - stylized_end_points[name]) ** 2)
    weighted_loss = weight * loss

    content_loss_dict['content_loss/' + name] = loss
    content_loss_dict['weighted_content_loss/' + name] = weighted_loss
    total_content_loss += weighted_loss

  content_loss_dict['total_content_loss'] = total_content_loss

  return total_content_loss, content_loss_dict


def style_loss(style_end_points, stylized_end_points, style_weights):
  """Style loss.

  Args:
    style_end_points: dict mapping VGG16 layer names to their corresponding
        Tensor value for the style input.
    stylized_end_points: dict mapping VGG16 layer names to their corresponding
        Tensor value for the stylized input.
    style_weights: dict mapping layer names to their associated style loss
        weight. Keys that are missing from the dict won't have their style
        loss computed.

  Returns:
    Tensor for the total style loss, dict mapping loss names to losses.
  """
  total_style_loss = np.float32(0.0)
  style_loss_dict = {}

  for name, weight in style_weights.items():
    loss = tf.reduce_mean(
        (learning_utils.gram_matrix(stylized_end_points[name]) -
         learning_utils.gram_matrix(style_end_points[name])) ** 2)
    weighted_loss = weight * loss

    style_loss_dict['style_loss/' + name] = loss
    style_loss_dict['weighted_style_loss/' + name] = weighted_loss
    total_style_loss += weighted_loss

  style_loss_dict['total_style_loss'] = total_style_loss

  return total_style_loss, style_loss_dict
