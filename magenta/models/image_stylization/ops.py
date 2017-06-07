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
"""Compound TensorFlow operations for style transfer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

slim = tf.contrib.slim


@slim.add_arg_scope
def conditional_instance_norm(inputs,
                              labels,
                              num_categories,
                              center=True,
                              scale=True,
                              activation_fn=None,
                              reuse=None,
                              variables_collections=None,
                              outputs_collections=None,
                              trainable=True,
                              scope=None):
  """Conditional instance normalization from TODO(vdumoulin): add link.

    "A Learned Representation for Artistic Style"

    Vincent Dumoulin, Jon Shlens, Manjunath Kudlur

  Can be used as a normalizer function for conv2d.

  Args:
    inputs: a tensor with 4 dimensions. The normalization occurs over height
        and width.
    labels: tensor, style labels to condition on.
    num_categories: int, total number of styles being modeled.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: Optional activation function.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined, or if the
        input doesn't have 4 dimensions.
  """
  with tf.variable_scope(scope, 'InstanceNorm', [inputs],
                         reuse=reuse) as sc:
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    if inputs_rank != 4:
      raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = [1, 2]
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))

    def _label_conditioned_variable(name, initializer, labels, num_categories):
      """Label conditioning."""
      shape = tf.TensorShape([num_categories]).concatenate(params_shape)
      var_collections = slim.utils.get_variable_collections(
          variables_collections, name)
      var = slim.model_variable(name,
                                shape=shape,
                                dtype=dtype,
                                initializer=initializer,
                                collections=var_collections,
                                trainable=trainable)
      conditioned_var = tf.gather(var, labels)
      conditioned_var = tf.expand_dims(tf.expand_dims(conditioned_var, 1), 1)
      return conditioned_var

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta = _label_conditioned_variable(
          'beta', tf.zeros_initializer(), labels, num_categories)
    if scale:
      gamma = _label_conditioned_variable(
          'gamma', tf.ones_initializer(), labels, num_categories)
    # Calculate the moments on the last axis (instance activations).
    mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1E-5
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn:
      outputs = activation_fn(outputs)
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            outputs)


@slim.add_arg_scope
def weighted_instance_norm(inputs,
                           weights,
                           num_categories,
                           center=True,
                           scale=True,
                           activation_fn=None,
                           reuse=None,
                           variables_collections=None,
                           outputs_collections=None,
                           trainable=True,
                           scope=None):
  """Weighted instance normalization.

  Can be used as a normalizer function for conv2d.

  Args:
    inputs: a tensor with 4 dimensions. The normalization occurs over height
        and width.
    weights: 1D tensor.
    num_categories: int, total number of styles being modeled.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: Optional activation function.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined, or if the
        input doesn't have 4 dimensions.
  """
  with tf.variable_scope(scope, 'InstanceNorm', [inputs],
                         reuse=reuse) as sc:
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    if inputs_rank != 4:
      raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = [1, 2]
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))

    def _weighted_variable(name, initializer, weights, num_categories):
      """Weighting."""
      shape = tf.TensorShape([num_categories]).concatenate(params_shape)
      var_collections = slim.utils.get_variable_collections(
          variables_collections, name)
      var = slim.model_variable(name,
                                shape=shape,
                                dtype=dtype,
                                initializer=initializer,
                                collections=var_collections,
                                trainable=trainable)
      weights = tf.reshape(
          weights,
          weights.get_shape().concatenate([1] * params_shape.ndims))
      conditioned_var = weights * var
      conditioned_var = tf.reduce_sum(conditioned_var, 0, keep_dims=True)
      conditioned_var = tf.expand_dims(tf.expand_dims(conditioned_var, 1), 1)
      return conditioned_var

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta = _weighted_variable(
          'beta', tf.zeros_initializer(), weights, num_categories)
    if scale:
      gamma = _weighted_variable(
          'gamma', tf.ones_initializer(), weights, num_categories)
    # Calculate the moments on the last axis (instance activations).
    mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1E-5
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn:
      outputs = activation_fn(outputs)
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            outputs)
