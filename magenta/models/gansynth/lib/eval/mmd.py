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
"""Utility function for computing MMD between samples from two models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from magenta.models.gansynth.lib.eval import pitch_classifier_utils
from magenta.models.nsynth import utils


def energy_kernel(x, y):
  return -tf.sqrt(
      tf.reduce_sum(
          tf.squared_difference(x[None, :, :], y[:, None, :]),
          axis=2
      )
  )


def gaussian_kernel(x, y, gamma=0.1):
  return tf.exp(
      -gamma * tf.reduce_sum(
          tf.squared_difference(x[None, :, :], y[:, None, :]),
          axis=2
      )
  )


def rq_kernel(x, y, alpha=0.0):
  return tf.pow(
      1 + (tf.reduce_sum(
          tf.squared_difference(x[None, :, :], y[:, None, :]),
          axis=2
      ) / (2*alpha)),
      -alpha
  )


def dot_kernel(x, y):
  return tf.reduce_sum(x[None, :, :] * y[:, None, :], axis=2)


def rq_star_kernel(x, y):
  return (rq_kernel(x, y, 0.2) +
          rq_kernel(x, y, 0.5) +
          rq_kernel(x, y, 1.0) +
          rq_kernel(x, y, 2.0) +
          rq_kernel(x, y, 5.0) +
          dot_kernel(x, y))


def polynomial_kernel(x, y, degree=3):
  dot_product = x[None, :, :] * y[:, None, :]
  kernel = tf.pow(
      1 + tf.reduce_mean(dot_product, axis=2), degree
  )
  return kernel


def e_kernel(kernel, batch_size, x, y, mtype="triangular"):
  """E_{x~X,y~Y}[kernel(x, y)]."""
  mask = np.zeros((batch_size, batch_size), dtype="float32")
  for i in xrange(batch_size):
    if mtype == "triangular":
      mask[i, :i] = 1.
    elif mtype == "none":
      mask[i, :] = 1.
    elif mtype == "diagonal":
      mask[i, :] = 1.
      mask[i, i] = 0.
    else:
      raise Exception("invalid mask")
  n = np.sum(mask)
  mask = tf.constant(mask)
  return tf.reduce_sum(mask * kernel(x, y)) / n


def get_mmd(kernel, batch_size, x_real, x_fake, mask="triangular"):
  """Get MMD^2 between two x, y."""
  return (e_kernel(kernel, batch_size, x_real, x_real, mtype=mask) +
          e_kernel(kernel, batch_size, x_fake, x_fake, mtype=mask) -
          (2 * e_kernel(kernel, batch_size, x_real, x_fake, mtype="none")))


def run(flags, real_samples, fake_samples):
  """Returns MMD for model estimated on `num_samples`.

  Args:
    flags:
    real_samples:
    fake_samples:
  Returns:
    mmd:
  """

  kernel_map = {"gaussian": gaussian_kernel,
                "energy": energy_kernel,
                "rq": rq_kernel,
                "dot": dot_kernel,
                "rq_star": rq_star_kernel,
                "polynomial": polynomial_kernel}

  num_samples = flags["mmd_num_samples"]
  kernel = kernel_map[flags["mmd_kernel"]]

  batch_size = flags["eval_batch_size"]
  num_batches = num_samples // batch_size

  classifier_model = utils.get_module("models.pitch")
  hparams = classifier_model.get_hparams()
  hparams = pitch_classifier_utils.set_pitch_hparams(
      batch_size, hparams)
  classifier_layer = flags["classifier_layer"]
  pitch_checkpoint_dir = flags["pitch_checkpoint_dir"]

  batch_mmd = []

  with tf.Graph().as_default() as graph:
    real_batch_tf = tf.placeholder(name="real", dtype=tf.float32)
    fake_batch_tf = tf.placeholder(name="fake", dtype=tf.float32)
    real_features_op = pitch_classifier_utils.get_features(
        real_batch_tf, classifier_layer, hparams)
    real_features_op = tf.reshape(real_features_op, [batch_size, -1])
    fake_features_op = pitch_classifier_utils.get_features(
        fake_batch_tf, classifier_layer, hparams)
    fake_features_op = tf.reshape(fake_features_op, [batch_size, -1])
    mmd_op = get_mmd(kernel, batch_size, real_features_op, fake_features_op)

    with tf.Session(graph=graph) as sess:
      tf.train.Saver().restore(
          sess, tf.train.latest_checkpoint(pitch_checkpoint_dir))
      print("Loaded pitch-detection parameters")
      batch_idx = 0
      print("Starting MMD computation")
      for idx in range(num_batches):
        print("processing batch : %d / %d" % (idx, num_batches))
        real_batch = real_samples[batch_idx:batch_idx+batch_size, :]
        fake_batch = fake_samples[batch_idx:batch_idx+batch_size, :]
        mmd_val = sess.run(mmd_op, feed_dict={real_batch_tf: real_batch,
                                              fake_batch_tf: fake_batch})
        batch_mmd.append(mmd_val)
        batch_idx += batch_size
    mmd = np.mean(batch_mmd)
    return {"mmd_score": mmd}
