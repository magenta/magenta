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

"""Style transfer network code."""
from magenta.models.image_stylization import ops
import tensorflow.compat.v1 as tf
import tf_slim as slim


def transform(input_,
              alpha=1.0,
              normalizer_fn=ops.conditional_instance_norm,
              normalizer_params=None,
              reuse=False):
  """Maps content images to stylized images.

  Args:
    input_: Tensor. Batch of input images.
    alpha: Float. Width multiplier to reduce the number of filters in the model
        and slim it down. Defaults to 1.0, which results in the hyper-parameters
        used in the published paper.
    normalizer_fn: normalization layer function.  Defaults to
        ops.conditional_instance_norm.
    normalizer_params: dict of parameters to pass to the conditional instance
        normalization op.
    reuse: bool. Whether to reuse model parameters. Defaults to False.

  Returns:
    Tensor. The output of the transformer network.
  """
  if normalizer_params is None:
    normalizer_params = {'center': True, 'scale': True}
  with tf.variable_scope('transformer', reuse=reuse):
    with slim.arg_scope(
        [slim.conv2d],
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params,
        weights_initializer=tf.random_normal_initializer(0.0, 0.01),
        biases_initializer=tf.constant_initializer(0.0)):
      with tf.variable_scope('contract'):
        h = conv2d(input_, 9, 1, int(alpha * 32), 'conv1')
        h = conv2d(h, 3, 2, int(alpha * 64), 'conv2')
        h = conv2d(h, 3, 2, int(alpha * 128), 'conv3')
      with tf.variable_scope('residual'):
        h = residual_block(h, 3, 'residual1')
        h = residual_block(h, 3, 'residual2')
        h = residual_block(h, 3, 'residual3')
        h = residual_block(h, 3, 'residual4')
        h = residual_block(h, 3, 'residual5')
      with tf.variable_scope('expand'):
        h = upsampling(h, 3, 2, int(alpha * 64), 'conv1')
        h = upsampling(h, 3, 2, int(alpha * 32), 'conv2')
        return upsampling(h, 9, 1, 3, 'conv3', activation_fn=tf.nn.sigmoid)


def conv2d(input_,
           kernel_size,
           stride,
           num_outputs,
           scope,
           activation_fn=tf.nn.relu):
  """Same-padded convolution with mirror padding instead of zero-padding.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  padding = kernel_size // 2
  padded_input = tf.pad(
      input_, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')
  return slim.conv2d(
      padded_input,
      padding='VALID',
      kernel_size=kernel_size,
      stride=stride,
      num_outputs=num_outputs,
      activation_fn=activation_fn,
      scope=scope)


def upsampling(input_,
               kernel_size,
               stride,
               num_outputs,
               scope,
               activation_fn=tf.nn.relu):
  """A smooth replacement of a same-padded transposed convolution.

  This function first computes a nearest-neighbor upsampling of the input by a
  factor of `stride`, then applies a mirror-padded, same-padded convolution.

  It expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    shape = tf.shape(input_)
    height = shape[1]
    width = shape[2]
    upsampled_input = tf.image.resize_nearest_neighbor(
        input_, [stride * height, stride * width])
    return conv2d(
        upsampled_input,
        kernel_size,
        1,
        num_outputs,
        'conv',
        activation_fn=activation_fn)


def residual_block(input_, kernel_size, scope, activation_fn=tf.nn.relu):
  """A residual block made of two mirror-padded, same-padded convolutions.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor, the input.
    kernel_size: int (odd-valued) representing the kernel size.
    scope: str, scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor, the output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    num_outputs = int(input_.get_shape()[-1])
    h_1 = conv2d(input_, kernel_size, 1, num_outputs, 'conv1', activation_fn)
    h_2 = conv2d(h_1, kernel_size, 1, num_outputs, 'conv2', None)
    return input_ + h_2
