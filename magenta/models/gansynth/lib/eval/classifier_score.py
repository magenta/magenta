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
"""Inception Score for audio samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from magenta.models.gansynth.lib.eval import pitch_classifier_utils
from magenta.models.nsynth import utils

tfgan_eval = tf.contrib.gan.eval


def kl_divergence(p, p_logits, q):
  return tf.reduce_sum(
      p * (tf.nn.log_softmax(p_logits) - tf.log(q)), axis=1)


def classifier_score_from_logits(logits, q):
  p = tf.nn.softmax(logits)
  kl = kl_divergence(p, logits, q)
  kl.shape.assert_has_rank(1)
  log_score = tf.reduce_mean(kl)
  return log_score


def run(flags, all_samples):
  """Compute Classifier (inception-style) score using TF.Gan library."""

  # Defining the pitch detection model
  with tf.Graph().as_default() as graph:
    batch_size = flags['eval_batch_size']
    classifier_model = utils.get_module('models.pitch')
    hparams = classifier_model.get_hparams()
    hparams = pitch_classifier_utils.set_pitch_hparams(
        batch_size, hparams)
    classifier_layer = 'logits'

    audio_placeholder = tf.placeholder(name='x', dtype=tf.float32)
    audio_logits_op = pitch_classifier_utils.get_features(
        audio_placeholder, classifier_layer, hparams)
    pitch_logits_op = audio_logits_op[:, :hparams.n_pitches]

    pitch_dist_op = tf.nn.softmax(pitch_logits_op, axis=1)
    pitch_distributions = []
    with tf.Session(graph=graph) as sess:
      saver = tf.train.Saver()
      saver.restore(sess, tf.train.latest_checkpoint(
          flags.pitch_checkpoint_dir))
      print('Loaded pitch-detection params')
      num_batches = all_samples.shape[0] // batch_size
      for batch_idx in range(num_batches):
        print('processing batch %s / %s' %(batch_idx, num_batches))
        audio_samples = all_samples[batch_idx*batch_size:(batch_idx+1)*batch_size, :]  # pylint: disable=line-too-long
        pitch_dist = sess.run(pitch_dist_op,
                              feed_dict={audio_placeholder: audio_samples})
        pitch_distributions.append(np.mean(pitch_dist, axis=0))

    marginal = np.mean(pitch_distributions, axis=0)
    classifier_score_op = classifier_score_from_logits(pitch_logits_op,
                                                       marginal)

    with tf.Session(graph=graph) as sess:
      saver = tf.train.Saver()
      saver.restore(sess, tf.train.latest_checkpoint(
          flags.pitch_checkpoint_dir))
      print('Loaded pitch-detection parameters')

      batch_classifier_score = []
      num_batches = all_samples.shape[0] // batch_size
      for batch_idx in range(num_batches):
        print('processing batch %s / %s' %(batch_idx, num_batches))
        audio_samples = all_samples[batch_idx*batch_size:(batch_idx+1)*batch_size, :]  # pylint: disable=line-too-long
        classifier_score = sess.run(classifier_score_op,
                                    feed_dict={audio_placeholder:
                                               audio_samples})
        batch_classifier_score.append(classifier_score)
    mean_classifier_score = np.exp(np.mean(batch_classifier_score))
  return {'classifier_score': mean_classifier_score}
