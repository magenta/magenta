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

"""Model in the dapaspace (e.g. pre-trained VAE).

The whole experiment handles transfer between latent space
of generative models that model the data. This file defines models
that explicitly model the data (x) in the latent space (z) and provide
mechanism of encoding (x->z) and decoding (z->x).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.latent_transfer.common import dataset_is_mnist_family
import numpy as np
from six import iteritems
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

ds = tfp.distributions


class Model(snt.AbstractModule):
  """VAE for MNIST or CelebA dataset."""

  def __init__(self, config, name=''):
    super(Model, self).__init__(name=name)
    self.config = config

  def _build(self, unused_input=None):
    # pylint:disable=unused-variable,possibly-unused-variable
    # Reason:
    #   All endpoints are stored as attribute at the end of `_build`.
    #   Pylint cannot infer this case so it emits false alarm of
    #   unused-variable if we do not disable this warning.

    config = self.config

    # Constants
    batch_size = config['batch_size']
    n_latent = config['n_latent']
    img_width = config['img_width']

    # ---------------------------------------------------------------------
    # ## Placeholders
    # ---------------------------------------------------------------------
    # Image data
    if dataset_is_mnist_family(config['dataset']):
      n_labels = 10
      x = tf.placeholder(
          tf.float32, shape=(None, img_width * img_width), name='x')
      attr_loss_fn = tf.losses.softmax_cross_entropy
      attr_pred_fn = tf.nn.softmax
      attr_weights = tf.constant(np.ones([1]).astype(np.float32))
      # p_x_fn = lambda logits: ds.Bernoulli(logits=logits)
      x_sigma = tf.constant(config['x_sigma'])
      p_x_fn = (lambda logs: ds.Normal(loc=tf.nn.sigmoid(logs), scale=x_sigma)
               )  # noqa

    elif config['dataset'] == 'CELEBA':
      n_labels = 10
      x = tf.placeholder(
          tf.float32, shape=(None, img_width, img_width, 3), name='x')
      attr_loss_fn = tf.losses.sigmoid_cross_entropy
      attr_pred_fn = tf.nn.sigmoid
      attr_weights = tf.constant(np.ones([1, n_labels]).astype(np.float32))
      x_sigma = tf.constant(config['x_sigma'])
      p_x_fn = (lambda logs: ds.Normal(loc=tf.nn.sigmoid(logs), scale=x_sigma)
               )  # noqa

    # Attributes
    labels = tf.placeholder(tf.int32, shape=(None, n_labels), name='labels')
    # Real / fake label reward
    r = tf.placeholder(tf.float32, shape=(None, 1), name='D_label')
    # Transform through optimization
    z0 = tf.placeholder(tf.float32, shape=(None, n_latent), name='z0')

    # ---------------------------------------------------------------------
    # ## Modules with parameters
    # ---------------------------------------------------------------------
    # Abstract Modules.
    # Variable that is class has name consider to be invalid by pylint so we
    # disable the warning.
    # pylint:disable=invalid-name
    Encoder = config['Encoder']
    Decoder = config['Decoder']
    Classifier = config['Classifier']
    # pylint:enable=invalid-name

    encoder = Encoder(name='encoder')
    decoder = Decoder(name='decoder')
    classifier = Classifier(output_size=n_labels, name='classifier')

    # ---------------------------------------------------------------------
    # ## Classify Attributes from pixels
    # ---------------------------------------------------------------------
    logits_classifier = classifier(x)
    pred_classifier = attr_pred_fn(logits_classifier)
    classifier_loss = attr_loss_fn(labels, logits=logits_classifier)

    # ---------------------------------------------------------------------
    # ## VAE
    # ---------------------------------------------------------------------
    # Encode
    mu, sigma = encoder(x)
    q_z = ds.Normal(loc=mu, scale=sigma)

    # Optimize / Amortize or feedthrough
    q_z_sample = q_z.sample()

    z = q_z_sample

    # Decode
    logits = decoder(z)
    p_x = p_x_fn(logits)
    x_mean = p_x.mean()

    # Reconstruction Loss
    if config['dataset'] == 'CELEBA':
      recons = tf.reduce_sum(p_x.log_prob(x), axis=[1, 2, 3])
    else:
      recons = tf.reduce_sum(p_x.log_prob(x), axis=[-1])

    mean_recons = tf.reduce_mean(recons)

    # Prior
    p_z = ds.Normal(loc=0., scale=1.)
    prior_sample = p_z.sample(sample_shape=[batch_size, n_latent])

    # KL Loss.
    # We use `KL` in variable name for naming consistency with math.
    # pylint:disable=invalid-name
    if config['beta'] == 0:
      mean_KL = tf.constant(0.0)
    else:
      KL_qp = ds.kl_divergence(q_z, p_z)
      KL = tf.reduce_sum(KL_qp, axis=-1)
      mean_KL = tf.reduce_mean(KL)
    # pylint:enable=invalid-name

    # VAE Loss
    beta = tf.constant(config['beta'])
    vae_loss = -mean_recons + mean_KL * beta

    # ---------------------------------------------------------------------
    # ## Training
    # ---------------------------------------------------------------------
    # Learning rates
    vae_lr = tf.constant(3e-4)
    classifier_lr = tf.constant(3e-4)

    # Training Ops
    vae_vars = list(encoder.get_variables())
    vae_vars.extend(decoder.get_variables())
    train_vae = tf.train.AdamOptimizer(learning_rate=vae_lr).minimize(
        vae_loss, var_list=vae_vars)

    classifier_vars = classifier.get_variables()
    train_classifier = tf.train.AdamOptimizer(
        learning_rate=classifier_lr).minimize(
            classifier_loss, var_list=classifier_vars)

    # Savers
    vae_saver = tf.train.Saver(vae_vars, max_to_keep=100)
    classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1000)

    # Add all endpoints as object attributes
    for k, v in iteritems(locals()):
      self.__dict__[k] = v

    # pylint:enable=unused-variable,possibly-unused-variable
