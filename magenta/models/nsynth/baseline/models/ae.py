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

# Lint as: python2, python3
"""Autoencoder model for training on spectrograms."""
from magenta.contrib import training as contrib_training
from magenta.models.nsynth import utils
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim


def get_hparams(config_name):
  """Set hyperparameters.

  Args:
    config_name: Name of config module to use.

  Returns:
    A HParams object (magenta) with defaults.
  """
  hparams = contrib_training.HParams(
      # Optimization
      batch_size=16,
      learning_rate=1e-4,
      adam_beta=0.5,
      max_steps=6000 * 50000,
      samples_per_second=16000,
      num_samples=64000,
      # Preprocessing
      n_fft=1024,
      hop_length=256,
      mask=True,
      log_mag=True,
      use_cqt=False,
      re_im=False,
      dphase=True,
      mag_only=False,
      pad=True,
      mu_law_num=0,
      raw_audio=False,
      # Graph
      num_latent=64,  # dimension of z.
      cost_phase_mask=False,
      phase_loss_coeff=1.0,
      fw_loss_coeff=1.0,  # Frequency weighted cost
      fw_loss_cutoff=1000,
  )
  # Set values from a dictionary in the config
  config = utils.get_module("baseline.models.ae_configs.%s" % config_name)
  if hasattr(config, "config_hparams"):
    config_hparams = config.config_hparams
    hparams.update(config_hparams)
  return hparams


def compute_mse_loss(x, xhat, hparams):
  """MSE loss function.

  Args:
    x: Input data tensor.
    xhat: Reconstruction tensor.
    hparams: Hyperparameters.

  Returns:
    total_loss: MSE loss scalar.
  """
  with tf.name_scope("Losses"):
    if hparams.raw_audio:
      total_loss = tf.reduce_mean((x - xhat)**2)
    else:
      # Magnitude
      m = x[:, :, :, 0] if hparams.cost_phase_mask else 1.0
      fm = utils.frequency_weighted_cost_mask(
          hparams.fw_loss_coeff,
          hz_flat=hparams.fw_loss_cutoff,
          n_fft=hparams.n_fft)
      mag_loss = tf.reduce_mean(fm * (x[:, :, :, 0] - xhat[:, :, :, 0])**2)
      if hparams.mag_only:
        total_loss = mag_loss
      else:
        # Phase
        if hparams.dphase:
          phase_loss = tf.reduce_mean(fm * m *
                                      (x[:, :, :, 1] - xhat[:, :, :, 1])**2)
        else:
          # Von Mises Distribution "Circular Normal"
          # Added constant to keep positive (Same Probability) range [0, 2]
          phase_loss = 1 - tf.reduce_mean(fm * m * tf.cos(
              (x[:, :, :, 1] - xhat[:, :, :, 1]) * np.pi))
        total_loss = mag_loss + hparams.phase_loss_coeff * phase_loss
        tf.summary.scalar("Loss/Mag", mag_loss)
        tf.summary.scalar("Loss/Phase", phase_loss)
    tf.summary.scalar("Loss/Total", total_loss)
  return total_loss


def train_op(batch, hparams, config_name):
  """Define a training op, including summaries and optimization.

  Args:
    batch: Dictionary produced by NSynthDataset.
    hparams: Hyperparameters dictionary.
    config_name: Name of config module.

  Returns:
    train_op: A complete iteration of training with summaries.
  """
  config = utils.get_module("baseline.models.ae_configs.%s" % config_name)

  if hparams.raw_audio:
    x = batch["audio"]
    # Add height and channel dims
    x = tf.expand_dims(tf.expand_dims(x, 1), -1)
  else:
    x = batch["spectrogram"]

  # Define the model
  with tf.name_scope("Model"):
    z = config.encode(x, hparams)
    xhat = config.decode(z, batch, hparams)

  # For interpolation
  tf.add_to_collection("x", x)
  tf.add_to_collection("pitch", batch["pitch"])
  tf.add_to_collection("z", z)
  tf.add_to_collection("xhat", xhat)

  # Compute losses
  total_loss = compute_mse_loss(x, xhat, hparams)

  # Apply optimizer
  with tf.name_scope("Optimizer"):
    global_step = tf.get_variable(
        "global_step", [],
        tf.int64,
        initializer=tf.constant_initializer(0),
        trainable=False)
    optimizer = tf.train.AdamOptimizer(hparams.learning_rate, hparams.adam_beta)
    train_step = slim.learning.create_train_op(total_loss,
                                               optimizer,
                                               global_step=global_step)

  return train_step


def eval_op(batch, hparams, config_name):
  """Define a evaluation op.

  Args:
    batch: Batch produced by NSynthReader.
    hparams: Hyperparameters.
    config_name: Name of config module.

  Returns:
    eval_op: A complete evaluation op with summaries.
  """
  phase = not (hparams.mag_only or hparams.raw_audio)

  config = utils.get_module("baseline.models.ae_configs.%s" % config_name)
  if hparams.raw_audio:
    x = batch["audio"]
    # Add height and channel dims
    x = tf.expand_dims(tf.expand_dims(x, 1), -1)
  else:
    x = batch["spectrogram"]

  # Define the model
  with tf.name_scope("Model"):
    z = config.encode(x, hparams, is_training=False)
    xhat = config.decode(z, batch, hparams, is_training=False)

  # For interpolation
  tf.add_to_collection("x", x)
  tf.add_to_collection("pitch", batch["pitch"])
  tf.add_to_collection("z", z)
  tf.add_to_collection("xhat", xhat)

  total_loss = compute_mse_loss(x, xhat, hparams)

  # Define the metrics:
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      "Loss": slim.metrics.mean(total_loss),
  })

  # Define the summaries
  for name, value in names_to_values.items():
    slim.summaries.add_scalar_summary(value, name, print_summary=True)

  # Interpolate
  with tf.name_scope("Interpolation"):
    xhat = config.decode(z, batch, hparams, reuse=True, is_training=False)

    # Linear interpolation
    z_shift_one_example = tf.concat([z[1:], z[:1]], 0)
    z_linear_half = (z + z_shift_one_example) / 2.0
    xhat_linear_half = config.decode(z_linear_half, batch, hparams, reuse=True,
                                     is_training=False)

    # Pitch shift

    pitch_plus_2 = tf.clip_by_value(batch["pitch"] + 2, 0, 127)
    pitch_minus_2 = tf.clip_by_value(batch["pitch"] - 2, 0, 127)

    batch["pitch"] = pitch_minus_2
    xhat_pitch_minus_2 = config.decode(z, batch, hparams,
                                       reuse=True, is_training=False)
    batch["pitch"] = pitch_plus_2
    xhat_pitch_plus_2 = config.decode(z, batch, hparams,
                                      reuse=True, is_training=False)

  utils.specgram_summaries(x, "Training Examples", hparams, phase=phase)
  utils.specgram_summaries(xhat, "Reconstructions", hparams, phase=phase)
  utils.specgram_summaries(
      x - xhat, "Difference", hparams, audio=False, phase=phase)
  utils.specgram_summaries(
      xhat_linear_half, "Linear Interp. 0.5", hparams, phase=phase)
  utils.specgram_summaries(xhat_pitch_plus_2, "Pitch +2", hparams, phase=phase)
  utils.specgram_summaries(xhat_pitch_minus_2, "Pitch -2", hparams, phase=phase)

  return list(names_to_updates.values())
