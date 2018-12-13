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
"""
Methods for building real-time arbitrary image stylization model
using MobileNetV2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_losses as losses
from magenta.models.arbitrary_image_stylization import nza_model as transformer_model
from magenta.models.image_stylization import ops

try:
  from nets.mobilenet import mobilenet_v2, mobilenet
except ImportError:
  print("Cannot import MobileNet model. Make sure to install slim "
        "models library described "
        "in https://github.com/tensorflow/models/tree/master/research/slim")
  raise

slim = tf.contrib.slim


def build_mobilenet_model(content_input_,
                          style_input_,
                          mobilenet_trainable=True,
                          style_params_trainable=False,
                          transformer_trainable=False,
                          reuse=None,
                          mobilenet_end_point='layer_19',
                          style_prediction_bottleneck=100,
                          adds_losses=True,
                          content_weights=None,
                          style_weights=None,
                          total_variation_weight=None):
  [activation_names,
   activation_depths] = transformer_model.style_normalization_activations()

  # Defines the style prediction network.
  style_params, bottleneck_feat = style_prediction_mobilenet(
      style_input_,
      activation_names,
      activation_depths,
      mobilenet_end_point=mobilenet_end_point,
      mobilenet_trainable=mobilenet_trainable,
      style_params_trainable=style_params_trainable,
      style_prediction_bottleneck=style_prediction_bottleneck,
      reuse=reuse
  )

  # Defines the style transformer network
  stylized_images = transformer_model.transform(
      content_input_,
      normalizer_fn=ops.conditional_style_norm,
      reuse=reuse,
      trainable=transformer_trainable,
      is_training=transformer_trainable,
      normalizer_params={'style_params': style_params}
  )

  # Adds losses
  loss_dict = {}
  total_loss = []
  if adds_losses:
    total_loss, loss_dict = losses.total_loss(
        content_input_,
        style_input_,
        stylized_images,
        content_weights=content_weights,
        style_weights=style_weights,
        total_variation_weight=total_variation_weight
    )

  return stylized_images, total_loss, loss_dict, bottleneck_feat


def style_prediction_mobilenet(style_input_,
                               activation_names,
                               activation_depths,
                               mobilenet_end_point='layer_19',
                               mobilenet_trainable=True,
                               style_params_trainable=False,
                               style_prediction_bottleneck=100,
                               reuse=None):
  with tf.name_scope('style_prediction_mobilenet') and tf.variable_scope(
      tf.get_variable_scope(), reuse=reuse):
    with slim.arg_scope(mobilenet_v2.training_scope(
        is_training=mobilenet_trainable)):
      _, end_points = mobilenet.mobilenet_base(
          style_input_,
          conv_defs=mobilenet_v2.V2_DEF,
          final_endpoint=mobilenet_end_point,
          scope='MobilenetV2'
      )

    feat_convlayer = end_points[mobilenet_end_point]
    with tf.name_scope('bottleneck'):
      # (batch_size, 1, 1, depth).
      bottleneck_feat = tf.reduce_mean(
          feat_convlayer, axis=[1, 2], keep_dims=True)

    if style_prediction_bottleneck > 0:
      with tf.variable_scope('mobilenet_conv'):
        with slim.arg_scope(
            [slim.conv2d],
            activation_fn=None,
            normalizer_fn=None,
            trainable=mobilenet_trainable):
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
              trainable=style_params_trainable):
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
