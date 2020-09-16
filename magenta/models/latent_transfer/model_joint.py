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

"""The joint transfer model that bridges latent spaces of dataspace models.

The whole experiment handles transfer between latent space
of generative models that model the data. This file defines the joint model
that models the transfer between latent spaces (z1, z2) of models on dataspace.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.latent_transfer import nn
from six import iteritems
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

ds = tfp.distributions


def affine(x, output_size, z=None, residual=False, softplus=False):
  """Make an affine layer with optional residual link and softplus activation.

  Args:
    x: An TF tensor which is the input.
    output_size: The size of output, e.g. the dimension of this affine layer.
    z: An TF tensor which is added when residual link is enabled.
    residual: A boolean indicating whether to enable residual link.
    softplus: Whether to apply softplus activation at the end.

  Returns:
    The output tensor.
  """
  if residual:
    x = snt.Linear(2 * output_size)(x)
    z = snt.Linear(output_size)(z)
    dz = x[:, :output_size]
    gates = tf.nn.sigmoid(x[:, output_size:])
    output = (1 - gates) * z + gates * dz
  else:
    output = snt.Linear(output_size)(x)

  if softplus:
    output = tf.nn.softplus(output)

  return output


class EncoderLatentFull(snt.AbstractModule):
  """An MLP (Full layers) encoder for modeling latent space."""

  def __init__(self,
               input_size,
               output_size,
               layers=(2048,) * 4,
               name='EncoderLatentFull',
               residual=True):
    super(EncoderLatentFull, self).__init__(name=name)
    self.layers = layers
    self.input_size = input_size
    self.output_size = output_size
    self.residual = residual

  def _build(self, z):
    x = z
    for l in self.layers:
      x = tf.nn.relu(snt.Linear(l)(x))

    mu = affine(x, self.output_size, z, residual=self.residual, softplus=False)
    sigma = affine(
        x, self.output_size, z, residual=self.residual, softplus=True)
    return mu, sigma


class DecoderLatentFull(snt.AbstractModule):
  """An MLP (Full layers) decoder for modeling latent space."""

  def __init__(self,
               input_size,
               output_size,
               layers=(2048,) * 4,
               name='DecoderLatentFull',
               residual=True):
    super(DecoderLatentFull, self).__init__(name=name)
    self.layers = layers
    self.input_size = input_size
    self.output_size = output_size
    self.residual = residual

  def _build(self, z):
    x = z
    for l in self.layers:
      x = tf.nn.relu(snt.Linear(l)(x))

    mu = affine(x, self.output_size, z, residual=self.residual, softplus=False)
    return mu


class VAE(snt.AbstractModule):
  """VAE for modling latant space."""

  def __init__(self, config, name=''):
    super(VAE, self).__init__(name=name)
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
    n_latent_shared = config['n_latent_shared']

    # ---------------------------------------------------------------------
    # ## Placeholders
    # ---------------------------------------------------------------------

    x = tf.placeholder(tf.float32, shape=(None, n_latent))

    # ---------------------------------------------------------------------
    # ## Modules with parameters
    # ---------------------------------------------------------------------
    # Variable that is class has name consider to be invalid by pylint so we
    # disable the warning.
    # pylint:disable=invalid-name
    Encoder = config['Encoder']
    Decoder = config['Decoder']
    encoder = Encoder(name='encoder')
    decoder = Decoder(name='decoder')
    # pylint:enable=invalid-name

    # ---------------------------------------------------------------------
    # ## Placeholders
    # ---------------------------------------------------------------------
    mu, sigma = encoder(x)
    mean_abs_mu, mean_abs_sigma = tf.reduce_mean(tf.abs(mu)), tf.reduce_mean(
        tf.abs(sigma))  # for summary only
    q_z = ds.Normal(loc=mu, scale=sigma)
    q_z_sample = q_z.sample()

    # Decode
    x_prime = decoder(q_z_sample)

    # Reconstruction Loss
    # Don't use log_prob from tf.ds (larger = better)
    # Instead, we use L2 norm (smaller = better)
    # # recons = tf.reduce_sum(p_x.log_prob(x), axis=[-1])
    recons = tf.reduce_mean(tf.square(x_prime - x))
    mean_recons = tf.reduce_mean(recons)

    # Prior
    p_z = ds.Normal(loc=0., scale=1.)
    p_z_sample = p_z.sample(sample_shape=[batch_size, n_latent_shared])
    x_from_prior = decoder(p_z_sample)

    # Space filling

    # We use `KL` in variable name for naming consistency with math.
    # pylint:disable=invalid-name
    beta = config['prior_loss_beta']
    if beta == 0:
      prior_loss = tf.constant(0.0)
    else:
      if config['prior_loss'].lower() == 'KL'.lower():
        KL_qp = ds.kl_divergence(ds.Normal(loc=mu, scale=sigma), p_z)
        KL = tf.reduce_sum(KL_qp, axis=-1)
        mean_KL = tf.reduce_mean(KL)
        prior_loss = mean_KL
      else:
        raise NotImplementedError()
    # pylint:enable=invalid-name

    # VAE Loss
    beta = tf.constant(config['prior_loss_beta'])
    scaled_prior_loss = prior_loss * beta
    vae_loss = mean_recons + scaled_prior_loss

    # ---------------------------------------------------------------------
    # ## Training
    # ---------------------------------------------------------------------
    # Learning rates
    vae_lr = tf.constant(3e-4)
    # Training Ops
    vae_vars = list(encoder.get_variables())
    vae_vars.extend(decoder.get_variables())

    if vae_vars:
      # Here, if we use identity transferm, there is no var to optimize,
      # so in this case we shall avoid building optimizer and saver,
      # otherwise there would be
      # "No variables to optimize." / "No variables to save" error.

      # Optimizer
      train_vae = tf.train.AdamOptimizer(learning_rate=vae_lr).minimize(
          vae_loss, var_list=vae_vars)

      # Savers
      vae_saver = tf.train.Saver(vae_vars, max_to_keep=100)

    # Add all endpoints as object attributes
    for k, v in iteritems(locals()):
      self.__dict__[k] = v

    # pylint:enable=unused-variable,possibly-unused-variable


class Model(snt.AbstractModule):
  """A joint model with two VAEs for latent spaces and ops for transfer.

  This model containts two VAEs to model two latant spaces individually,
  as well as extra Baysian Inference in training to enable transfer.
  """

  def __init__(self, config, name=''):
    super(Model, self).__init__(name=name)
    self.config = config

  def _build(self, unused_input=None):
    # pylint:disable=unused-variable,possibly-unused-variable
    # Reason:
    #   All endpoints are stored as attribute at the end of `_build`.
    #   Pylint cannot infer this case so it emits false alarm of
    #   unused-variable if we do not disable this warning.

    # pylint:disable=invalid-name
    # Reason:
    #   Following variables have their name consider to be invalid by pylint so
    #   we disable the warning.
    #   - Variable that is class
    #   - Variable that in its name has A or B indicating their belonging of
    #     one side of data.

    # ---------------------------------------------------------------------
    # ## Extract parameters from config
    # ---------------------------------------------------------------------

    config = self.config
    lr = config.get('lr', 3e-4)
    n_latent_shared = config['n_latent_shared']

    if 'n_latent' in config:
      n_latent_A = n_latent_B = config['n_latent']
    else:
      n_latent_A = config['vae_A']['n_latent']
      n_latent_B = config['vae_B']['n_latent']

    # ---------------------------------------------------------------------
    # ## VAE containing Modules with parameters
    # ---------------------------------------------------------------------
    vae_A = VAE(config['vae_A'], name='vae_A')
    vae_A()
    vae_B = VAE(config['vae_B'], name='vae_B')
    vae_B()

    vae_lr = tf.constant(lr)
    vae_vars = vae_A.vae_vars + vae_B.vae_vars
    vae_loss = vae_A.vae_loss + vae_B.vae_loss
    train_vae = tf.train.AdamOptimizer(learning_rate=vae_lr).minimize(
        vae_loss, var_list=vae_vars)
    vae_saver = tf.train.Saver(vae_vars, max_to_keep=100)

    # ---------------------------------------------------------------------
    # ## Computation Flow
    # ---------------------------------------------------------------------

    # Tensor Endpoints
    x_A = vae_A.x
    x_B = vae_B.x
    q_z_sample_A = vae_A.q_z_sample
    q_z_sample_B = vae_B.q_z_sample
    mu_A, sigma_A = vae_A.mu, vae_A.sigma
    mu_B, sigma_B = vae_B.mu, vae_B.sigma
    x_prime_A = vae_A.x_prime
    x_prime_B = vae_B.x_prime
    x_from_prior_A = vae_A.x_from_prior
    x_from_prior_B = vae_B.x_from_prior
    x_A_to_B = vae_B.decoder(q_z_sample_A)
    x_B_to_A = vae_A.decoder(q_z_sample_B)
    x_A_to_B_direct = vae_B.decoder(mu_A)
    x_B_to_A_direct = vae_A.decoder(mu_B)
    z_hat = tf.placeholder(tf.float32, shape=(None, n_latent_shared))
    x_joint_A = vae_A.decoder(z_hat)
    x_joint_B = vae_B.decoder(z_hat)

    vae_loss_A = vae_A.vae_loss
    vae_loss_B = vae_B.vae_loss

    x_align_A = tf.placeholder(tf.float32, shape=(None, n_latent_A))
    x_align_B = tf.placeholder(tf.float32, shape=(None, n_latent_B))
    mu_align_A, sigma_align_A = vae_A.encoder(x_align_A)
    mu_align_B, sigma_align_B = vae_B.encoder(x_align_B)
    q_z_align_A = ds.Normal(loc=mu_align_A, scale=sigma_align_A)
    q_z_align_B = ds.Normal(loc=mu_align_B, scale=sigma_align_B)

    # VI in joint space

    mu_align, sigma_align = nn.product_two_guassian_pdfs(
        mu_align_A, sigma_align_A, mu_align_B, sigma_align_B)
    q_z_align = ds.Normal(loc=mu_align, scale=sigma_align)
    p_z_align = ds.Normal(loc=0., scale=1.)

    # - KL
    KL_qp_align = ds.kl_divergence(q_z_align, p_z_align)
    KL_align = tf.reduce_sum(KL_qp_align, axis=-1)
    mean_KL_align = tf.reduce_mean(KL_align)
    prior_loss_align = mean_KL_align
    prior_loss_align_beta = config.get('prior_loss_align_beta', 0.0)
    scaled_prior_loss_align = prior_loss_align * prior_loss_align_beta

    # - Reconstruction (from joint Gussian)
    q_z_sample_align = q_z_align.sample()
    x_prime_A_align = vae_A.decoder(q_z_sample_align)
    x_prime_B_align = vae_B.decoder(q_z_sample_align)

    mean_recons_A_align = tf.reduce_mean(tf.square(x_prime_A_align - x_align_A))
    mean_recons_B_align = tf.reduce_mean(tf.square(x_prime_B_align - x_align_B))
    mean_recons_A_align_beta = config.get('mean_recons_A_align_beta', 0.0)
    scaled_mean_recons_A_align = mean_recons_A_align * mean_recons_A_align_beta
    mean_recons_B_align_beta = config.get('mean_recons_B_align_beta', 0.0)
    scaled_mean_recons_B_align = mean_recons_B_align * mean_recons_B_align_beta
    scaled_mean_recons_align = (
        scaled_mean_recons_A_align + scaled_mean_recons_B_align)

    # - Reconstruction (from transfer)
    q_z_align_A_sample = q_z_align_A.sample()
    q_z_align_B_sample = q_z_align_B.sample()
    x_A_to_B_align = vae_B.decoder(q_z_align_A_sample)
    x_B_to_A_align = vae_A.decoder(q_z_align_B_sample)
    mean_recons_A_to_B_align = tf.reduce_mean(
        tf.square(x_A_to_B_align - x_align_B))
    mean_recons_B_to_A_align = tf.reduce_mean(
        tf.square(x_B_to_A_align - x_align_A))
    mean_recons_A_to_B_align_beta = config.get('mean_recons_A_to_B_align_beta',
                                               0.0)
    scaled_mean_recons_A_to_B_align = (
        mean_recons_A_to_B_align * mean_recons_A_to_B_align_beta)
    mean_recons_B_to_A_align_beta = config.get('mean_recons_B_to_A_align_beta',
                                               0.0)
    scaled_mean_recons_B_to_A_align = (
        mean_recons_B_to_A_align * mean_recons_B_to_A_align_beta)
    scaled_mean_recons_cross_A_B_align = (
        scaled_mean_recons_A_to_B_align + scaled_mean_recons_B_to_A_align)

    # Full loss
    full_loss = (vae_loss_A + vae_loss_B + scaled_mean_recons_align +
                 scaled_mean_recons_cross_A_B_align)

    # train op
    full_lr = tf.constant(lr)
    train_full = tf.train.AdamOptimizer(learning_rate=full_lr).minimize(
        full_loss, var_list=vae_vars)

    # Add all endpoints as object attributes
    for k, v in iteritems(locals()):
      self.__dict__[k] = v

    # pylint:enable=unused-variable,possibly-unused-variable
    # pylint:enable=invalid-name

  def get_summary_kv_dict(self):
    m = self
    return {
        'm.vae_A.mean_recons':
        m.vae_A.mean_recons,
        'm.vae_A.prior_loss':
        m.vae_A.prior_loss,
        'm.vae_A.scaled_prior_loss':
        m.vae_A.scaled_prior_loss,
        'm.vae_A.vae_loss':
        m.vae_A.vae_loss,
        'm.vae_B.mean_recons':
        m.vae_B.mean_recons,
        'm.vae_A.mean_abs_mu':
        m.vae_A.mean_abs_mu,
        'm.vae_A.mean_abs_sigma':
        m.vae_A.mean_abs_sigma,
        'm.vae_B.prior_loss':
        m.vae_B.prior_loss,
        'm.vae_B.scaled_prior_loss':
        m.vae_B.scaled_prior_loss,
        'm.vae_B.vae_loss':
        m.vae_B.vae_loss,
        'm.vae_B.mean_abs_mu':
        m.vae_B.mean_abs_mu,
        'm.vae_B.mean_abs_sigma':
        m.vae_B.mean_abs_sigma,
        'm.vae_loss_A':
        m.vae_loss_A,
        'm.vae_loss_B':
        m.vae_loss_B,
        'm.prior_loss_align':
        m.prior_loss_align,
        'm.scaled_prior_loss_align':
        m.scaled_prior_loss_align,
        'm.mean_recons_A_align':
        m.mean_recons_A_align,
        'm.mean_recons_B_align':
        m.mean_recons_B_align,
        'm.scaled_mean_recons_A_align':
        m.scaled_mean_recons_A_align,
        'm.scaled_mean_recons_B_align':
        m.scaled_mean_recons_B_align,
        'm.scaled_mean_recons_align':
        m.scaled_mean_recons_align,
        'm.mean_recons_A_to_B_align':
        m.mean_recons_A_to_B_align,
        'm.mean_recons_B_to_A_align':
        m.mean_recons_B_to_A_align,
        'm.scaled_mean_recons_A_to_B_align':
        m.scaled_mean_recons_A_to_B_align,
        'm.scaled_mean_recons_B_to_A_align':
        m.scaled_mean_recons_B_to_A_align,
        'm.scaled_mean_recons_cross_A_B_align':
        m.scaled_mean_recons_cross_A_B_align,
        'm.full_loss':
        m.full_loss
    }
