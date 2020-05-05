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

"""Config for MNIST <> WaveGAN transfer.
"""

# pylint:disable=invalid-name

import functools

from magenta.models.latent_transfer import model_joint
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

n_latent_A = 100
n_latent_B = 100
n_latent_shared = FLAGS.n_latent_shared
layers = (128,) * 4
layers_B = (2048,) * 8
batch_size = 128

Encoder = functools.partial(
    model_joint.EncoderLatentFull,
    input_size=n_latent_A,
    output_size=n_latent_shared,
    layers=layers)

Decoder = functools.partial(
    model_joint.DecoderLatentFull,
    input_size=n_latent_shared,
    output_size=n_latent_A,
    layers=layers)

vae_config_A = {
    'Encoder': Encoder,
    'Decoder': Decoder,
    'prior_loss_beta': FLAGS.prior_loss_beta_A,
    'prior_loss': 'KL',
    'batch_size': batch_size,
    'n_latent': n_latent_A,
    'n_latent_shared': n_latent_shared,
}


def make_Encoder_B(n_latent):
  return functools.partial(
      model_joint.EncoderLatentFull,
      input_size=n_latent,
      output_size=n_latent_shared,
      layers=layers_B,
  )


def make_Decoder_B(n_latent):
  return functools.partial(
      model_joint.DecoderLatentFull,
      input_size=n_latent_shared,
      output_size=n_latent,
      layers=layers_B,
  )


wavegan_config_B = {
    'Encoder': make_Encoder_B(n_latent_B),
    'Decoder': make_Decoder_B(n_latent_B),
    'prior_loss_beta': FLAGS.prior_loss_beta_B,
    'prior_loss': 'KL',
    'batch_size': batch_size,
    'n_latent': n_latent_B,
    'n_latent_shared': n_latent_shared,
}

config = {
    'vae_A': vae_config_A,
    'vae_B': wavegan_config_B,
    'config_A': 'mnist_0_nlatent100',
    'config_B': 'wavegan',
    'config_classifier_A': 'mnist_classifier_0',
    'config_classifier_B': '<unused>',
    # model
    'prior_loss_align_beta': FLAGS.prior_loss_align_beta,
    'mean_recons_A_align_beta': FLAGS.mean_recons_A_align_beta,
    'mean_recons_B_align_beta': FLAGS.mean_recons_B_align_beta,
    'mean_recons_A_to_B_align_beta': FLAGS.mean_recons_A_to_B_align_beta,
    'mean_recons_B_to_A_align_beta': FLAGS.mean_recons_B_to_A_align_beta,
    'pairing_number': FLAGS.pairing_number,
    # training dynamics
    'batch_size': batch_size,
    'n_latent_shared': n_latent_shared,
}
