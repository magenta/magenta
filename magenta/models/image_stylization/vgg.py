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

"""Implementation of the VGG-16 network.

In this specific implementation, max-pooling operations are replaced with
average-pooling operations.
"""
import os

import tensorflow.compat.v1 as tf
import tf_slim as slim

flags = tf.app.flags
flags.DEFINE_string('vgg_checkpoint', None, 'Path to VGG16 checkpoint file.')
FLAGS = flags.FLAGS


def checkpoint_file():
  """Get the path to the VGG16 checkpoint file from flags.

  Returns:
    Path to the VGG checkpoint.
  Raises:
    ValueError: checkpoint is null.
  """
  if FLAGS.vgg_checkpoint is None:
    raise ValueError('VGG checkpoint is None.')

  return os.path.expanduser(FLAGS.vgg_checkpoint)


def vgg_16(inputs, reuse=False, pooling='avg', final_endpoint='fc8'):
  """VGG-16 implementation intended for test-time use.

  It takes inputs with values in [0, 1] and preprocesses them (scaling,
  mean-centering) before feeding them to the VGG-16 network.

  Args:
    inputs: A 4-D tensor of shape [batch_size, image_size, image_size, 3]
        and dtype float32, with values in [0, 1].
    reuse: bool. Whether to reuse model parameters. Defaults to False.
    pooling: str in {'avg', 'max'}, which pooling operation to use. Defaults
        to 'avg'.
    final_endpoint: str, specifies the endpoint to construct the network up to.
        Defaults to 'fc8'.

  Returns:
    A dict mapping end-point names to their corresponding Tensor.

  Raises:
    ValueError: the final_endpoint argument is not recognized.
  """
  inputs *= 255.0
  inputs -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

  pooling_fns = {'avg': slim.avg_pool2d, 'max': slim.max_pool2d}
  pooling_fn = pooling_fns[pooling]

  with tf.variable_scope('vgg_16', [inputs], reuse=reuse) as sc:
    end_points = {}

    def add_and_check_is_final(layer_name, net):
      end_points['%s/%s' % (sc.name, layer_name)] = net
      return layer_name == final_endpoint

    with slim.arg_scope([slim.conv2d], trainable=False):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      if add_and_check_is_final('conv1', net): return end_points
      net = pooling_fn(net, [2, 2], scope='pool1')
      if add_and_check_is_final('pool1', net): return end_points
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      if add_and_check_is_final('conv2', net): return end_points
      net = pooling_fn(net, [2, 2], scope='pool2')
      if add_and_check_is_final('pool2', net): return end_points
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      if add_and_check_is_final('conv3', net): return end_points
      net = pooling_fn(net, [2, 2], scope='pool3')
      if add_and_check_is_final('pool3', net): return end_points
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      if add_and_check_is_final('conv4', net): return end_points
      net = pooling_fn(net, [2, 2], scope='pool4')
      if add_and_check_is_final('pool4', net): return end_points
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      if add_and_check_is_final('conv5', net): return end_points
      net = pooling_fn(net, [2, 2], scope='pool5')
      if add_and_check_is_final('pool5', net): return end_points
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      if add_and_check_is_final('fc6', net): return end_points
      net = slim.dropout(net, 0.5, is_training=False, scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      if add_and_check_is_final('fc7', net): return end_points
      net = slim.dropout(net, 0.5, is_training=False, scope='dropout7')
      net = slim.conv2d(net, 1000, [1, 1], activation_fn=None,
                        scope='fc8')
      end_points[sc.name + '/predictions'] = slim.softmax(net)
      if add_and_check_is_final('fc8', net): return end_points

    raise ValueError('final_endpoint (%s) not recognized' % final_endpoint)
