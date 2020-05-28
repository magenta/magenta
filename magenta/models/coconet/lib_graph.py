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

"""Defines the graph for a convolutional net designed for music autofill."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from magenta.models.coconet import lib_hparams
from magenta.models.coconet import lib_tfutil
import tensorflow.compat.v1 as tf


class CoconetGraph(object):
  """Model for predicting autofills given context."""

  def __init__(self,
               is_training,
               hparams,
               placeholders=None,
               direct_inputs=None,
               use_placeholders=True):
    self.hparams = hparams
    self.batch_size = hparams.batch_size
    self.num_pitches = hparams.num_pitches
    self.num_instruments = hparams.num_instruments
    self.is_training = is_training
    self.placeholders = placeholders
    self._direct_inputs = direct_inputs
    self._use_placeholders = use_placeholders
    self.hiddens = []
    self.popstats_by_batchstat = collections.OrderedDict()
    self.build()

  @property
  def use_placeholders(self):
    return self._use_placeholders

  @use_placeholders.setter
  def use_placeholders(self, use_placeholders):
    self._use_placeholders = use_placeholders

  @property
  def inputs(self):
    if self.use_placeholders:
      return self.placeholders
    else:
      return self.direct_inputs

  @property
  def direct_inputs(self):
    return self._direct_inputs

  @direct_inputs.setter
  def direct_inputs(self, direct_inputs):
    if set(direct_inputs.keys()) != set(self.placeholders.keys()):
      raise AttributeError('Need to have pianorolls, masks, lengths.')
    self._direct_inputs = direct_inputs

  @property
  def pianorolls(self):
    return self.inputs['pianorolls']

  @property
  def masks(self):
    return self.inputs['masks']

  @property
  def lengths(self):
    return self.inputs['lengths']

  def build(self):
    """Builds the graph."""
    featuremaps = self.get_convnet_input()
    self.residual_init()

    layers = self.hparams.get_conv_arch().layers
    n = len(layers)
    for i, layer in enumerate(layers):
      with tf.variable_scope('conv%d' % i):
        self.residual_counter += 1
        self.residual_save(featuremaps)

        featuremaps = self.apply_convolution(featuremaps, layer, i)
        featuremaps = self.apply_residual(
            featuremaps, is_first=i == 0, is_last=i == n - 1)
        featuremaps = self.apply_activation(featuremaps, layer)
        featuremaps = self.apply_pooling(featuremaps, layer)

        self.hiddens.append(featuremaps)

    self.logits = featuremaps
    self.predictions = self.compute_predictions(logits=self.logits)
    self.cross_entropy = self.compute_cross_entropy(
        logits=self.logits, labels=self.pianorolls)

    self.compute_loss(self.cross_entropy)
    self.setup_optimizer()

    for var in tf.trainable_variables():
      tf.logging.info('%s_%r', var.name, var.get_shape().as_list())

  def get_convnet_input(self):
    """Returns concatenates masked out pianorolls with their masks."""
    # pianorolls, masks = self.inputs['pianorolls'], self.inputs[
    #     'masks']
    pianorolls, masks = self.pianorolls, self.masks
    pianorolls *= 1. - masks
    if self.hparams.mask_indicates_context:
      # flip meaning of mask for convnet purposes: after flipping, mask is hot
      # where values are known. this makes more sense in light of padding done
      # by convolution operations: the padded area will have zero mask,
      # indicating no information to rely on.
      masks = 1. - masks
    return tf.concat([pianorolls, masks], axis=3)

  def setup_optimizer(self):
    """Instantiates learning rate, decay op and train_op among others."""
    # If not training, don't need to add optimizer to the graph.
    if not self.is_training:
      self.train_op = tf.no_op()
      self.learning_rate = tf.no_op()
      return

    self.learning_rate = tf.Variable(
        self.hparams.learning_rate,
        name='learning_rate',
        trainable=False,
        dtype=tf.float32)

    # FIXME 0.5 -> hparams.decay_rate
    self.decay_op = tf.assign(self.learning_rate, 0.5 * self.learning_rate)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)

  def compute_predictions(self, logits):
    if self.hparams.use_softmax_loss:
      return tf.nn.softmax(logits, dim=2)
    return tf.nn.sigmoid(logits)

  def compute_cross_entropy(self, logits, labels):
    if self.hparams.use_softmax_loss:
      # don't use tf.nn.softmax_cross_entropy because we need the shape to
      # remain constant
      return -tf.nn.log_softmax(logits, dim=2) * labels
    else:
      return tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=labels)

  def compute_loss(self, unreduced_loss):
    """Computes scaled loss based on mask out size."""
    # construct mask to identify zero padding that was introduced to
    # make the batch rectangular
    batch_duration = tf.shape(self.pianorolls)[1]
    indices = tf.to_float(tf.range(batch_duration))
    pad_mask = tf.to_float(
        indices[None, :, None, None] < self.lengths[:, None, None, None])

    # construct mask and its complement, respecting pad mask
    mask = pad_mask * self.masks
    unmask = pad_mask * (1. - self.masks)

    # Compute numbers of variables
    # #timesteps * #variables per timestep
    variable_axis = 3 if self.hparams.use_softmax_loss else 2
    dd = (
        self.lengths[:, None, None, None] * tf.to_float(
            tf.shape(self.pianorolls)[variable_axis]))
    reduced_dd = tf.reduce_sum(dd)

    # Compute numbers of variables to be predicted/conditioned on
    mask_size = tf.reduce_sum(mask, axis=[1, variable_axis], keep_dims=True)
    unmask_size = tf.reduce_sum(unmask, axis=[1, variable_axis], keep_dims=True)

    unreduced_loss *= pad_mask
    if self.hparams.rescale_loss:
      unreduced_loss *= dd / mask_size

    # Compute average loss over entire set of variables
    self.loss_total = tf.reduce_sum(unreduced_loss) / reduced_dd

    # Compute separate losses for masked/unmasked variables
    # NOTE: indexing the pitch dimension with 0 because the mask is constant
    # across pitch. Except in the sigmoid case, but then the pitch dimension
    # will have been reduced over.
    self.reduced_mask_size = tf.reduce_sum(mask_size[:, :, 0, :])
    self.reduced_unmask_size = tf.reduce_sum(unmask_size[:, :, 0, :])

    assert_partition_op = tf.group(
        tf.assert_equal(tf.reduce_sum(mask * unmask), 0.),
        tf.assert_equal(self.reduced_mask_size + self.reduced_unmask_size,
                        reduced_dd))
    with tf.control_dependencies([assert_partition_op]):
      self.loss_mask = (
          tf.reduce_sum(mask * unreduced_loss) / self.reduced_mask_size)
      self.loss_unmask = (
          tf.reduce_sum(unmask * unreduced_loss) / self.reduced_unmask_size)

    # Check which loss to use as objective function.
    self.loss = (
        self.loss_mask if self.hparams.optimize_mask_only else self.loss_total)

  def residual_init(self):
    if not self.hparams.use_residual:
      return
    self.residual_period = 2
    self.output_for_residual = None
    self.residual_counter = -1

  def residual_reset(self):
    self.output_for_residual = None
    self.residual_counter = 0

  def residual_save(self, x):
    if not self.hparams.use_residual:
      return
    if self.residual_counter % self.residual_period == 1:
      self.output_for_residual = x

  def apply_residual(self, x, is_first, is_last):
    """Adds output saved from earlier layer to x if at residual period."""
    if not self.hparams.use_residual:
      return x
    if self.output_for_residual is None:
      return x
    if self.output_for_residual.get_shape()[-1] != x.get_shape()[-1]:
      # shape mismatch; e.g. change in number of filters
      self.residual_reset()
      return x
    if self.residual_counter % self.residual_period == 0:
      if not is_first and not is_last:
        x += self.output_for_residual
    return x

  def apply_convolution(self, x, layer, layer_idx):
    """Adds convolution and batch norm layers if hparam.batch_norm is True."""
    if 'filters' not in layer:
      return x

    filter_shape = layer['filters']
    # Instantiate or retrieve filter weights.
    fanin = tf.to_float(tf.reduce_prod(filter_shape[:-1]))
    stddev = tf.sqrt(tf.div(2.0, fanin))
    initializer = tf.random_normal_initializer(
        0.0, stddev)
    regular_convs = (not self.hparams.use_sep_conv or
                     layer_idx < self.hparams.num_initial_regular_conv_layers)
    if regular_convs:
      dilation_rates = layer.get('dilation_rate', 1)
      if isinstance(dilation_rates, int):
        dilation_rates = [dilation_rates] * 2
      weights = tf.get_variable(
          'weights',
          filter_shape,
          initializer=initializer if self.is_training else None)
      stride = layer.get('conv_stride', 1)
      conv = tf.nn.conv2d(
          x,
          weights,
          strides=[1, stride, stride, 1],
          padding=layer.get('conv_pad', 'SAME'),
          dilations=[1] + dilation_rates + [1])
    else:
      num_outputs = filter_shape[-1]
      num_splits = layer.get('num_pointwise_splits', 1)
      tf.logging.info('num_splits %d', num_splits)
      if num_splits > 1:
        num_outputs = None
      conv = tf.layers.separable_conv2d(
          x,
          num_outputs,
          filter_shape[:2],
          depth_multiplier=self.hparams.sep_conv_depth_multiplier,
          stride=layer.get('conv_stride', 1),
          padding=layer.get('conv_pad', 'SAME'),
          rate=layer.get('dilation_rate', 1),
          activation_fn=None,
          weights_initializer=initializer if self.is_training else None)
      if num_splits > 1:
        splits = tf.split(conv, num_splits, -1)
        print(len(splits), splits[0].shape)
        # TODO(annahuang): support non equal splits.
        pointwise_splits = [
            tf.layers.dense(splits[i], filter_shape[3]/num_splits,
                            name='split_%d_%d' % (layer_idx, i))
            for i in range(num_splits)]
        conv = tf.concat((pointwise_splits), axis=-1)

    # Compute batch normalization or add biases.
    if self.hparams.batch_norm:
      y = self.apply_batchnorm(conv)
    else:
      biases = tf.get_variable(
          'bias', [conv.get_shape()[-1]],
          initializer=tf.constant_initializer(0.0))
      y = tf.nn.bias_add(conv, biases)
    return y

  def apply_batchnorm(self, x):
    """Normalizes batch w/ moving population stats for training, o/w batch."""
    output_dim = x.get_shape()[-1]
    gammas = tf.get_variable(
        'gamma', [1, 1, 1, output_dim],
        initializer=tf.constant_initializer(0.1))
    betas = tf.get_variable(
        'beta', [output_dim], initializer=tf.constant_initializer(0.))

    popmean = tf.get_variable(
        'popmean',
        shape=[1, 1, 1, output_dim],
        trainable=False,
        collections=[
            tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES
        ],
        initializer=tf.constant_initializer(0.0))
    popvariance = tf.get_variable(
        'popvariance',
        shape=[1, 1, 1, output_dim],
        trainable=False,
        collections=[
            tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES
        ],
        initializer=tf.constant_initializer(1.0))

    decay = 0.01
    if self.is_training:
      batchmean, batchvariance = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
      mean, variance = batchmean, batchvariance
      updates = [
          popmean.assign_sub(decay * (popmean - mean)),
          popvariance.assign_sub(decay * (popvariance - variance))
      ]
      # make update happen when mean/variance are used
      with tf.control_dependencies(updates):
        mean, variance = tf.identity(mean), tf.identity(variance)
      self.popstats_by_batchstat[batchmean] = popmean
      self.popstats_by_batchstat[batchvariance] = popvariance
    else:
      mean, variance = popmean, popvariance

    return tf.nn.batch_normalization(x, mean, variance, betas, gammas,
                                     self.hparams.batch_norm_variance_epsilon)

  def apply_activation(self, x, layer):
    activation_func = layer.get('activation', tf.nn.relu)
    return activation_func(x)

  def apply_pooling(self, x, layer):
    if 'pooling' not in layer:
      return x
    pooling = layer['pooling']
    return tf.nn.max_pool(
        x,
        ksize=[1, pooling[0], pooling[1], 1],
        strides=[1, pooling[0], pooling[1], 1],
        padding=layer['pool_pad'])


def get_placeholders(hparams):
  return dict(
      pianorolls=tf.placeholder(
          tf.float32,
          [None, None, hparams.num_pitches, hparams.num_instruments]),
      masks=tf.placeholder(
          tf.float32,
          [None, None, hparams.num_pitches, hparams.num_instruments]),
      lengths=tf.placeholder(tf.float32, [None]))


def build_graph(is_training,
                hparams,
                placeholders=None,
                direct_inputs=None,
                use_placeholders=True):
  """Builds the model graph."""
  if placeholders is None and use_placeholders:
    placeholders = get_placeholders(hparams)
  initializer = tf.random_uniform_initializer(-hparams.init_scale,
                                              hparams.init_scale)
  with tf.variable_scope('model', reuse=None, initializer=initializer):
    graph = CoconetGraph(
        is_training=is_training,
        hparams=hparams,
        placeholders=placeholders,
        direct_inputs=direct_inputs,
        use_placeholders=use_placeholders)
  return graph


def load_checkpoint(path, instantiate_sess=True):
  """Builds graph, loads checkpoint, and returns wrapped model."""
  tf.logging.info('Loading checkpoint from %s', path)
  hparams = lib_hparams.load_hparams(path)
  model = build_graph(is_training=False, hparams=hparams)
  wmodel = lib_tfutil.WrappedModel(model, model.loss.graph, hparams)
  if not instantiate_sess:
    return wmodel
  with wmodel.graph.as_default():
    wmodel.sess = tf.Session()
    saver = tf.train.Saver()
    tf.logging.info('loading checkpoint %s', path)
    chkpt_path = os.path.join(path, 'best_model.ckpt')
    saver.restore(wmodel.sess, chkpt_path)
  return wmodel
