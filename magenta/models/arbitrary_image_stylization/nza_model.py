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

"""Style transfer network code.

This model does not apply styles in the encoding
layers. Encoding layers (contract) use batch norm as the normalization function.
"""
from magenta.models.image_stylization import model as model_util
import tensorflow.compat.v1 as tf
import tf_slim as slim


def transform(input_,
              normalizer_fn=None,
              normalizer_params=None,
              reuse=False,
              trainable=True,
              is_training=True,
              alpha=1.0):
  """Maps content images to stylized images.

  Args:
    input_: Tensor. Batch of input images.
    normalizer_fn: normalization layer function for applying style
      normalization.
    normalizer_params: dict of parameters to pass to the style normalization op.
    reuse: bool. Whether to reuse model parameters. Defaults to False.
    trainable: bool. Should the parameters be marked as trainable?
    is_training: bool. Is it training phase or not?
    alpha: float. Width multiplier to reduce the number of filters used in the
      model and slim it down. Defaults to 1.0, which results
      in the hyper-parameters used in the published paper.


  Returns:
    Tensor. The output of the transformer network.
  """
  with tf.variable_scope('transformer', reuse=reuse):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_initializer=tf.random_normal_initializer(
                            0.0, 0.01),
                        biases_initializer=tf.constant_initializer(0.0),
                        trainable=trainable):
      with slim.arg_scope([slim.conv2d],
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=None,
                          trainable=trainable):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training,
                            trainable=trainable):
          with tf.variable_scope('contract'):
            h = model_util.conv2d(input_, 9, 1, int(alpha * 32), 'conv1')
            h = model_util.conv2d(h, 3, 2, int(alpha * 64), 'conv2')
            h = model_util.conv2d(h, 3, 2, int(alpha * 128), 'conv3')
      with tf.variable_scope('residual'):
        h = model_util.residual_block(h, 3, 'residual1')
        h = model_util.residual_block(h, 3, 'residual2')
        h = model_util.residual_block(h, 3, 'residual3')
        h = model_util.residual_block(h, 3, 'residual4')
        h = model_util.residual_block(h, 3, 'residual5')
      with tf.variable_scope('expand'):
        h = model_util.upsampling(h, 3, 2, int(alpha * 64), 'conv1')
        h = model_util.upsampling(h, 3, 2, int(alpha * 32), 'conv2')
        return model_util.upsampling(
            h, 9, 1, 3, 'conv3', activation_fn=tf.nn.sigmoid)


def style_normalization_activations(pre_name='transformer',
                                    post_name='StyleNorm',
                                    alpha=1.0):
  """Returns scope name and depths of the style normalization activations.

  Args:
    pre_name: string. Prepends this name to the scope names.
    post_name: string. Appends this name to the scope names.
    alpha: float. Width multiplier to reduce the number of filters used in the
      model and slim it down.. Defaults to 1.0, which results
      in the hyper-parameters used in the published paper.

  Returns:
    string. Scope names of the activations of the transformer network which are
        used to apply style normalization.
    int[]. Depths of the activations of the transformer network which are used
        to apply style normalization.
  """

  scope_names = [
      'residual/residual1/conv1', 'residual/residual1/conv2',
      'residual/residual2/conv1', 'residual/residual2/conv2',
      'residual/residual3/conv1', 'residual/residual3/conv2',
      'residual/residual4/conv1', 'residual/residual4/conv2',
      'residual/residual5/conv1', 'residual/residual5/conv2',
      'expand/conv1/conv', 'expand/conv2/conv', 'expand/conv3/conv'
  ]
  scope_names = [
      '{}/{}/{}'.format(pre_name, name, post_name) for name in scope_names
  ]
  # 10 convolution layers of 'residual/residual*/conv*' have the same depth.
  depths = [int(alpha * 128)] * 10 + [int(alpha * 64), int(alpha * 32), 3]

  return scope_names, depths
