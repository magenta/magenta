# Copyright 2019 Google Inc. All Rights Reserved.
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
"""A model for classification of pitch from log magnitude and phase."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from magenta.models.nsynth import utils

slim = tf.contrib.slim


def get_hparams():
  return tf.HParams(
      # Optimization
      batch_size=16,
      learning_rate=1e-4,
      adam_beta=0.5,
      max_steps=6000 * 50000,
      samples_per_second=16000,
      num_samples=64000,
      # Preprocessing
      n_fft=512,
      hop_length=256,
      mask=True,
      log_mag=True,
      use_cqt=False,
      re_im=False,
      dphase=True,
      mag_only=False,
      pad=True,
      mu_law_num=0,
      orig_audio=False,
      # Graph
      n_pitches=128,
      n_instrument_families=12,
      n_instrument_sources=3,
      n_qualities=11,
      label_type="pitch",
      join_family_source=False,)


def calc_loss(logits, batch, hparams):
  """Softmax cross-entropy loss for pitch and instrument.

  Args:
    logits: Tensor of network outputs.
    batch: Input data from reader.
    hparams: Hyperparameters.

  Returns:
    total_loss: Scalar.
  """
  if hparams.label_type == "pitch":
    labels = batch.pitch
  if hparams.label_type == "instrument":
    labels = batch.instrument_family * (batch.instrument_source + 1)

  with tf.name_scope("Losses"):
    one_hot_labels = tf.reshape(labels, [hparams.batch_size])
    one_hot_labels = tf.one_hot(one_hot_labels, depth=hparams.n_pitches)
    total_loss = utils.calculate_softmax_and_summaries(logits, one_hot_labels,
                                                       "CE")
  return total_loss


def calc_all_loss(logits, batch, hparams):
  """Softmax cross-entropy loss for pitch, quality, and instrument.

  Args:
    logits: Tensor of network outputs.
    batch: Input data from reader.
    hparams: Hyperparameters.

  Returns:
    loss: Scalar.
  """
  pitch_logits = logits[:, :hparams.n_pitches]
  qualities_logits = logits[:, hparams.n_pitches:hparams.n_pitches +
                            hparams.n_qualities]

  pitch_labels = batch.pitch
  qualities_labels = batch.qualities

  with tf.name_scope("PitchLosses"):
    one_hot_pitch = tf.reshape(pitch_labels, [hparams.batch_size])
    one_hot_pitch = tf.one_hot(one_hot_pitch, depth=hparams.n_pitches)
    pitch_loss = utils.calculate_softmax_and_summaries(pitch_logits,
                                                       one_hot_pitch, "CE")

  with tf.name_scope("QualitiesLosses"):
    qualities_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=qualities_logits, labels=tf.to_float(qualities_labels))
    qualities_loss = tf.reduce_mean(qualities_loss)
    qualities_acc = tf.reduce_mean(
        tf.to_float(
            tf.equal(
                tf.to_int32(qualities_logits > 0), tf.to_int32(
                    qualities_labels))))
    tf.summary.scalar("Qualities Accuracy", qualities_acc)
    tf.summary.scalar("Qualities Loss", qualities_loss)

  if hparams.join_family_source:
    instrument_logits = logits[:, hparams.n_pitches + hparams.n_qualities:]
    instrument_labels = batch.instrument_family * (batch.instrument_source + 1)

    with tf.name_scope("InstrumentLosses"):
      one_hot_instrument = tf.reshape(instrument_labels, [hparams.batch_size])
      one_hot_instrument = tf.one_hot(
          one_hot_instrument,
          depth=hparams.n_instrument_families * hparams.n_instrument_sources)
      instrument_loss = utils.calculate_softmax_and_summaries(
          instrument_logits, one_hot_instrument, "CE")

    # Compute losses
    total_loss = pitch_loss + instrument_loss + qualities_loss

  else:
    instrument_family_logits = logits[:, hparams.n_pitches +
                                      hparams.n_qualities:hparams.n_pitches +
                                      hparams.n_qualities +
                                      hparams.n_instrument_families]
    instrument_source_logits = logits[:, hparams.n_pitches + hparams.n_qualities
                                      + hparams.n_instrument_families:]

    instrument_family_labels = batch.instrument_family
    instrument_source_labels = batch.instrument_source

    with tf.name_scope("InstrumentFamilyLosses"):
      one_hot_instrument_family = tf.reshape(instrument_family_labels,
                                             [hparams.batch_size])
      one_hot_instrument_family = tf.one_hot(
          one_hot_instrument_family, depth=hparams.n_instrument_families)
      instrument_family_loss = utils.calculate_softmax_and_summaries(
          instrument_family_logits, one_hot_instrument_family, "CE")

    with tf.name_scope("InstrumentSourceLosses"):
      one_hot_instrument_source = tf.reshape(instrument_source_labels,
                                             [hparams.batch_size])
      one_hot_instrument_source = tf.one_hot(
          one_hot_instrument_source, depth=hparams.n_instrument_sources)
      instrument_source_loss = utils.calculate_softmax_and_summaries(
          instrument_source_logits, one_hot_instrument_source, "CE")

    # Compute losses
    total_loss = (pitch_loss + instrument_source_loss + instrument_family_loss +
                  qualities_loss)

  return total_loss


def train_op(batch, hparams, config_name):
  """Define a training op, including summaries and optimization.

  Args:
    batch: Batch produced by NSynthReader.
    hparams: Hyperparameters.
    config_name: Name of config module.

  Returns:
    train_op: A complete iteration of training with summaries.
  """
  config = utils.get_module("models.pitch_configs.%s" % config_name)

  x = batch.spectrogram
  # Define the model
  with tf.name_scope("Model"):
    logits = config.fit(x, hparams)

  # Compute losses
  #   total_loss = calc_loss(logits, batch, hparams)
  total_loss = calc_all_loss(logits, batch, hparams)

  # Apply optimizer
  with tf.name_scope("Optimizer"):
    unused_global_step = slim.variables.get_or_create_global_step()
    optimizer = tf.AdamOptimizer(hparams.learning_rate, hparams.adam_beta)
    train_step = slim.learning.create_train_op(total_loss, optimizer)

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
  config = utils.get_module("models.pitch_configs.%s" % config_name)

  x = batch.spectrogram
  # Define the model
  with tf.name_scope("Model"):
    logits = config.fit(x, hparams)

  # Compute losses
  #   total_loss = calc_loss(logits, batch, hparams)
  total_loss = calc_all_loss(logits, batch, hparams)

  # Get other metrics (Accuracy)
  pitch_labels = batch.pitch
  qualities_labels = tf.to_int32(batch.qualities)

  pitch_logits = logits[:, :hparams.n_pitches]
  qualities_logits = logits[:, hparams.n_pitches:hparams.n_pitches +
                            hparams.n_qualities]

  pitch_preds = tf.argmax(pitch_logits, 1)
  qualities_preds = tf.to_int32(qualities_logits > 0)

  # Define the metrics:
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      "Loss":
          slim.metrics.mean(total_loss),
      "Pitch_Accuracy":
          slim.metrics.accuracy(pitch_preds, pitch_labels),
      # "Instrument_Accuracy":slim.metrics.accuracy(qualities_preds,
      #                                             qualities_labels),
      "Qualities_Accuracy":
          slim.metrics.accuracy(qualities_preds, qualities_labels),
  })

  # Define the summaries
  for name, value in names_to_values.iteritems():
    slim.summaries.add_scalar_summary(value, name, print_summary=True)

  return names_to_updates.values()
