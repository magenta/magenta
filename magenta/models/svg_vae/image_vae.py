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

"""Defines the ImageVAE model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.models.image_stylization import ops
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.log(tf.math.expm1(x))


@registry.register_model
class ImageVAE(t2t_model.T2TModel):
  """Defines the ImageVAE model."""

  def bottom(self, features):
    # inputs and targets should all be images, no preprocessing needed.
    # but we do need to resize them to 64x64.
    transformed_features = collections.OrderedDict()
    transformed_features['targets'] = features['targets']
    transformed_features['inputs'] = features['inputs']
    transformed_features['cls'] = features['targets_cls']
    if 'bottleneck' in features:
      transformed_features['bottleneck'] = features['bottleneck']
    return transformed_features

  def body(self, features):
    train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
    return self.vae_internal(features, self._hparams, train)

  def top(self, body_output, features):
    # body_output should be a dict with 'outputs', which will be an image.
    # no postprocessing needed.
    return body_output

  def loss(self, logits, features):
    # logits should be dict with 'outputs', which is image.
    targets = tf.reshape(features['targets'], [-1, 64, 64, 1])
    weights = common_layers.weights_all(targets)
    loss_num = tf.pow(logits - targets, 2)
    return tf.reduce_sum(loss_num * weights), tf.reduce_sum(weights)

  def vae_internal(self, features, hparams, train):
    # inputs and targets should both be images with dims [batch, 64, 64, 1]
    inputs, targets = features['inputs'], features['targets']
    inputs = tf.reshape(inputs, [-1, 64, 64, 1])
    targets = tf.reshape(targets, [-1, 64, 64, 1])

    clss = features['cls']

    with tf.variable_scope('vae_internal', reuse=tf.AUTO_REUSE):
      # encoder
      enc_out = self.visual_encoder(inputs, clss, hparams, train)
      enc_out = tf.reshape(enc_out, [-1, 2 * hparams.bottleneck_bits])

      # bottleneck
      sampled_bottleneck, b_loss = self.bottleneck(enc_out)
      losses = {'bottleneck_kl': tf.reduce_mean(b_loss)}

      if 'bottleneck' in features:
        if common_layers.shape_list(features['bottleneck'])[0] == 0:
          # return bottleneck for interpolation
          # set losses['training'] = 0 so top() isn't called on it
          # potential todo: use losses dict so we have kl_loss here for non stop
          # gradient models
          return sampled_bottleneck, {'training': 0.0}
        else:
          # we want to use the given bottleneck
          sampled_bottleneck = features['bottleneck']

      # finalize bottleneck
      unbottleneck = sampled_bottleneck

      # decoder.
      dec_out = self.visual_decoder(unbottleneck, clss, hparams)

      # calculate training loss here lol
      rec_loss = -dec_out.log_prob(inputs)
      elbo = tf.reduce_mean(-(b_loss + rec_loss))
      losses['rec_loss'] = tf.reduce_mean(rec_loss)
      losses['training'] = -elbo

      if (not hasattr(self, 'summarized_imgs')
          and self._hparams.mode != tf.estimator.ModeKeys.PREDICT):
        self.summarized_imgs = True
        with tf.name_scope(None), tf.name_scope('train' if train else 'test'):
          tf.summary.image('rendered_out', dec_out.mean())
          tf.summary.image('rendered_og', inputs)

    return dec_out.mean(), losses

  def bottleneck(self, x):
    z_size = self.hparams.bottleneck_bits
    x_shape = common_layers.shape_list(x)
    with tf.variable_scope('bottleneck', reuse=tf.AUTO_REUSE):
      mu = x[..., :self.hparams.bottleneck_bits]
      if self.hparams.mode != tf.estimator.ModeKeys.TRAIN:
        return mu, 0.0  # No sampling or kl loss on eval.
      log_sigma = x[..., self.hparams.bottleneck_bits:]
      epsilon = tf.random_normal(x_shape[:-1] + [z_size])
      z = mu + tf.exp(log_sigma / 2) * epsilon
      kl = 0.5 * tf.reduce_mean(
          tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma, axis=-1)
      # This is the 'free bits' trick mentioned in Kingma et al. (2016)
      free_bits = self.hparams.free_bits
      kl_loss = tf.reduce_mean(tf.maximum(kl - free_bits, 0.0))
    return z, kl_loss * self.hparams.kl_beta

  def visual_encoder(self, inputs, clss, hparams, train):
    del train
    # goes from [batch, 64, 64, 1] to [batch, hidden_size]
    with tf.variable_scope('visual_encoder', reuse=tf.AUTO_REUSE):
      ret = inputs
      clss = tf.reshape(clss, [-1])

      # conv layer, followed by instance norm + FiLM
      ret = tf.layers.Conv2D(hparams.base_depth, 5, 1,
                             padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2D(hparams.base_depth, 5, 2,
                             padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2D(2 * hparams.base_depth, 5, 1,
                             padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2D(2 * hparams.base_depth, 5, 2,
                             padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      # new conv layer, to bring shape down
      ret = tf.layers.Conv2D(2 * hparams.bottleneck_bits, 4, 2,
                             padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      # new conv layer, to bring shape down
      ret = tf.layers.Conv2D(2 * hparams.bottleneck_bits, 4, 2,
                             padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      # ret has 1024
      ret = tf.layers.flatten(ret)
      ret = tf.layers.dense(ret, 2 * hparams.bottleneck_bits, activation=None)

    return ret

  def visual_decoder(self, bottleneck, clss, hparams):
    # goes from [batch, bottleneck_bits] to [batch, 64, 64, 1]
    with tf.variable_scope('visual_decoder', reuse=tf.AUTO_REUSE):
      # unbottleneck
      ret = tf.layers.dense(bottleneck, 1024, activation=None)
      ret = tf.reshape(ret, [-1, 4, 4, 64])
      clss = tf.reshape(clss, [-1])

      # new deconv to bring shape up
      ret = tf.layers.Conv2DTranspose(2 * hparams.base_depth, 4, 2,
                                      padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      # new deconv to bring shape up
      ret = tf.layers.Conv2DTranspose(2 * hparams.base_depth, 4, 2,
                                      padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2DTranspose(2 * hparams.base_depth, 5, padding='SAME',
                                      activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2DTranspose(2 * hparams.base_depth, 5, 2,
                                      padding='SAME', activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2DTranspose(hparams.base_depth, 5, padding='SAME',
                                      activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2DTranspose(hparams.base_depth, 5, 2, padding='SAME',
                                      activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2DTranspose(hparams.base_depth, 5, padding='SAME',
                                      activation=None)(ret)
      ret = ops.conditional_instance_norm(ret, clss, hparams.num_categories)
      ret = tf.nn.relu(ret)

      ret = tf.layers.Conv2D(1, 5, padding='SAME', activation=None)(ret)

      ret = tfd.Independent(tfd.Bernoulli(logits=ret),
                            reinterpreted_batch_ndims=3,
                            name='image')
    return ret


@registry.register_hparams
def image_vae():
  """Basic Image VAE model hparams."""
  hparams = common_hparams.basic_params1()
  hparams.daisy_chain_variables = False
  hparams.batch_size = 64
  hparams.hidden_size = 32
  hparams.initializer = 'uniform_unit_scaling'
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0

  # VAE hparams
  hparams.add_hparam('base_depth', 32)
  hparams.add_hparam('bottleneck_bits', 32)

  # loss hparams
  hparams.add_hparam('kl_beta', 300)
  hparams.add_hparam('free_bits_div', 4)
  hparams.add_hparam('free_bits', 0.15)

  # data format hparams
  hparams.add_hparam('num_categories', 62)

  # problem hparams (required, don't modify)
  hparams.add_hparam('absolute', False)
  hparams.add_hparam('just_render', True)
  hparams.add_hparam('plus_render', False)

  return hparams
