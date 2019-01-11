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
"""Frechet Inception Distance for audio samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from magenta.models.gansynth.lib.eval import pitch_classifier_utils
from magenta.models.nsynth import utils

tfgan_eval = tf.contrib.gan.eval


def run(flags, real_samples, fake_samples):
  """Compute FID score using tf.gan library."""
  assert real_samples.shape == fake_samples.shape

  ## Set hparams for pitch-classifier model
  num_samples = flags['fid_num_samples']
  batch_size = flags['eval_batch_size']
  num_batches = num_samples // batch_size
  classifier_model = utils.get_module('models.pitch')
  hparams = classifier_model.get_hparams()
  hparams = pitch_classifier_utils.set_pitch_hparams(
      batch_size, hparams)
  pitch_checkpoint_dir = flags['pitch_checkpoint_dir']
  classifier_layer = flags['classifier_layer']

  ## Defining ops for computing FID between two models.
  with tf.Graph().as_default() as graph:
    real_batch_tf = tf.placeholder(name='real', dtype=tf.float32)
    fake_batch_tf = tf.placeholder(name='fake', dtype=tf.float32)
    real_features_tf = tf.placeholder(name='real_ft', dtype=tf.float32)
    fake_features_tf = tf.placeholder(name='fake_ft', dtype=tf.float32)
    real_features_op = pitch_classifier_utils.get_features(
        real_batch_tf, classifier_layer, hparams)
    real_features_op = tf.reshape(real_features_op, [batch_size, -1])
    fake_features_op = pitch_classifier_utils.get_features(
        fake_batch_tf, classifier_layer, hparams)
    fake_features_op = tf.reshape(fake_features_op, [batch_size, -1])
    fid_score_op = tfgan_eval.frechet_classifier_distance_from_activations(
        real_features_tf, fake_features_tf)

    with tf.Session(graph=graph) as sess:
      tf.train.Saver().restore(
          sess, tf.train.latest_checkpoint(pitch_checkpoint_dir))
      print('Loaded pitch-detection parameters')
      batch_idx = 0
      for idx in range(num_batches):
        print('processing batch %s / %s' %(idx, num_batches))
        real_batch = real_samples[batch_idx:batch_idx+batch_size, :]
        fake_batch = fake_samples[batch_idx:batch_idx+batch_size, :]
        real_features_np, fake_features_np = sess.run(
            [real_features_op, fake_features_op],
            feed_dict={real_batch_tf: real_batch,
                       fake_batch_tf: fake_batch})
        if idx == 0:
          real_features_all = real_features_np
          fake_features_all = fake_features_np
        else:
          np.concatenate([real_features_all, real_features_np], axis=0)
          np.concatenate([fake_features_all, fake_features_np], axis=0)
        batch_idx += batch_size
      fid_score = sess.run(fid_score_op,
                           feed_dict={real_features_tf: real_features_all,
                                      fake_features_tf: fake_features_all})

  return {'fid_score': fid_score}
