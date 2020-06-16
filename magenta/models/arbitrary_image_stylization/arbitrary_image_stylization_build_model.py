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
"""Methods for building real-time arbitrary image stylization model."""

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_losses as losses
from magenta.models.arbitrary_image_stylization import nza_model as transformer_model
from magenta.models.image_stylization import ops
import tensorflow.compat.v1 as tf
import tf_slim as slim
from tf_slim.nets import inception_v3


def build_model(content_input_,
                style_input_,
                trainable,
                is_training,
                reuse=None,
                inception_end_point='Mixed_6e',
                style_prediction_bottleneck=100,
                adds_losses=True,
                content_weights=None,
                style_weights=None,
                total_variation_weight=None):
  """The image stylize function.

  Args:
    content_input_: Tensor. Batch of content input images.
    style_input_: Tensor. Batch of style input images.
    trainable: bool. Should the parameters be marked as trainable?
    is_training: bool. Is it training phase or not?
    reuse: bool. Whether to reuse model parameters. Defaults to False.
    inception_end_point: string. Specifies the endpoint to construct the
        inception_v3 network up to. This network is used for style prediction.
    style_prediction_bottleneck: int. Specifies the bottleneck size in the
        number of parameters of the style embedding.
    adds_losses: wheather or not to add objectives to the model.
    content_weights: dict mapping layer names to their associated content loss
        weight. Keys that are missing from the dict won't have their content
        loss computed.
    style_weights: dict mapping layer names to their associated style loss
        weight. Keys that are missing from the dict won't have their style
        loss computed.
    total_variation_weight: float. Coefficient for the total variation part of
        the loss.

  Returns:
    Tensor for the output of the transformer network, Tensor for the total loss,
    dict mapping loss names to losses, Tensor for the bottleneck activations of
    the style prediction network.
  """
  # Gets scope name and shape of the activations of transformer network which
  # will be used to apply style.
  [activation_names,
   activation_depths] = transformer_model.style_normalization_activations()

  # Defines the style prediction network.
  style_params, bottleneck_feat = style_prediction(
      style_input_,
      activation_names,
      activation_depths,
      is_training=is_training,
      trainable=trainable,
      inception_end_point=inception_end_point,
      style_prediction_bottleneck=style_prediction_bottleneck,
      reuse=reuse)

  # Defines the style transformer network.
  stylized_images = transformer_model.transform(
      content_input_,
      normalizer_fn=ops.conditional_style_norm,
      reuse=reuse,
      trainable=trainable,
      is_training=is_training,
      normalizer_params={'style_params': style_params})

  # Adds losses.
  loss_dict = {}
  total_loss = []
  if adds_losses:
    total_loss, loss_dict = losses.total_loss(
        content_input_,
        style_input_,
        stylized_images,
        content_weights=content_weights,
        style_weights=style_weights,
        total_variation_weight=total_variation_weight)

  return stylized_images, total_loss, loss_dict, bottleneck_feat


def style_prediction(style_input_,
                     activation_names,
                     activation_depths,
                     is_training=True,
                     trainable=True,
                     inception_end_point='Mixed_6e',
                     style_prediction_bottleneck=100,
                     reuse=None):
  """Maps style images to the style embeddings (beta and gamma parameters).

  Args:
    style_input_: Tensor. Batch of style input images.
    activation_names: string. Scope names of the activations of the transformer
        network which are used to apply style normalization.
    activation_depths: Shapes of the activations of the transformer network
        which are used to apply style normalization.
    is_training: bool. Is it training phase or not?
    trainable: bool. Should the parameters be marked as trainable?
    inception_end_point: string. Specifies the endpoint to construct the
        inception_v3 network up to. This network is part of the style prediction
        network.
    style_prediction_bottleneck: int. Specifies the bottleneck size in the
        number of parameters of the style embedding.
    reuse: bool. Whether to reuse model parameters. Defaults to False.

  Returns:
    Tensor for the output of the style prediction network, Tensor for the
        bottleneck of style parameters of the style prediction network.
  """
  with tf.name_scope('style_prediction') and tf.variable_scope(
      tf.get_variable_scope(), reuse=reuse):
    with slim.arg_scope(_inception_v3_arg_scope(is_training=is_training)):
      with slim.arg_scope(
          [slim.conv2d, slim.fully_connected, slim.batch_norm],
          trainable=trainable):
        with slim.arg_scope(
            [slim.batch_norm, slim.dropout], is_training=is_training):
          _, end_points = inception_v3.inception_v3_base(
              style_input_,
              scope='InceptionV3',
              final_endpoint=inception_end_point)

    # Shape of feat_convlayer is (batch_size, ?, ?, depth).
    # For Mixed_6e end point, depth is 768, for input image size of 256x265
    # width and height are 14x14.
    feat_convlayer = end_points[inception_end_point]
    with tf.name_scope('bottleneck'):
      # (batch_size, 1, 1, depth).
      bottleneck_feat = tf.reduce_mean(
          feat_convlayer, axis=[1, 2], keep_dims=True)

    if style_prediction_bottleneck > 0:
      with slim.arg_scope(
          [slim.conv2d],
          activation_fn=None,
          normalizer_fn=None,
          trainable=trainable):
        # (batch_size, 1, 1, style_prediction_bottleneck).
        bottleneck_feat = slim.conv2d(bottleneck_feat,
                                      style_prediction_bottleneck, [1, 1])

    style_params = {}
    with tf.variable_scope('style_params'):
      for i in range(len(activation_depths)):
        with tf.variable_scope(activation_names[i], reuse=reuse):
          with slim.arg_scope(
              [slim.conv2d],
              activation_fn=None,
              normalizer_fn=None,
              trainable=trainable):

            # Computing beta parameter of the style normalization for the
            # activation_names[i] layer of the style transformer network.
            # (batch_size, 1, 1, activation_depths[i])
            beta = slim.conv2d(bottleneck_feat, activation_depths[i], [1, 1])
            # (batch_size, activation_depths[i])
            beta = tf.squeeze(beta, [1, 2], name='SpatialSqueeze')
            style_params['{}/beta'.format(activation_names[i])] = beta

            # Computing gamma parameter of the style normalization for the
            # activation_names[i] layer of the style transformer network.
            # (batch_size, 1, 1, activation_depths[i])
            gamma = slim.conv2d(bottleneck_feat, activation_depths[i], [1, 1])
            # (batch_size, activation_depths[i])
            gamma = tf.squeeze(gamma, [1, 2], name='SpatialSqueeze')
            style_params['{}/gamma'.format(activation_names[i])] = gamma

  return style_params, bottleneck_feat


def _inception_v3_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.1,
                            batch_norm_var_collection='moving_vars'):
  """Defines the default InceptionV3 arg scope.

  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    batch_norm_var_collection: The name of the collection for the batch norm
      variables.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      'is_training': is_training,
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }
  normalizer_fn = slim.batch_norm

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        activation_fn=tf.nn.relu6,
        normalizer_fn=normalizer_fn,
        normalizer_params=batch_norm_params) as sc:
      return sc
