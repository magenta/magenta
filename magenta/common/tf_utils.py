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

"""Tensorflow-related utilities."""

from magenta.contrib import training as contrib_training
import tensorflow.compat.v1 as tf


def merge_hparams(hparams_1, hparams_2):
  """Merge hyperparameters from two tf.contrib.training.HParams objects.

  If the same key is present in both HParams objects, the value from `hparams_2`
  will be used.

  Args:
    hparams_1: The first tf.contrib.training.HParams object to merge.
    hparams_2: The second tf.contrib.training.HParams object to merge.

  Returns:
    A merged tf.contrib.training.HParams object with the hyperparameters from
    both `hparams_1` and `hparams_2`.
  """
  hparams_map = hparams_1.values()
  hparams_map.update(hparams_2.values())
  return contrib_training.HParams(**hparams_map)


def log_loss(labels, predictions, epsilon=1e-7, scope=None, weights=None):
  """Calculate log losses.

  Same as tf.losses.log_loss except that this returns the individual losses
  instead of passing them into compute_weighted_loss and returning their
  weighted mean. This is useful for eval jobs that report the mean loss. By
  returning individual losses, that mean loss can be the same regardless of
  batch size.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.
    weights: Weights to apply to labels.

  Returns:
    A `Tensor` representing the loss values.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels`.
  """
  with tf.name_scope(scope, "log_loss", (predictions, labels)):
    predictions = tf.to_float(predictions)
    labels = tf.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = -tf.multiply(labels, tf.log(predictions + epsilon)) - tf.multiply(
        (1 - labels), tf.log(1 - predictions + epsilon))
    if weights is not None:
      losses = tf.multiply(losses, weights)

    return losses
