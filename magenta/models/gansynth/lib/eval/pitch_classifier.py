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
"""Evaluates metrics based on pitch classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from magenta.models.gansynth.lib.eval import pitch_classifier_utils
from magenta.models.nsynth.baseline.models import pitch as classifier_model


def run(flags, all_samples, all_pitches):
  """Evaluate the generated dataset to get pitch distribution.

  Saves the predicted-pitch and entropy for samples in the generated-dataset.

  Args:
    flags: Dictionary of evaluation flags
    all_samples: Batch of audio samples to be classified.
    all_pitches: Batch of pitch-labels corresponding to audio-samples.
  Returns:
    mean_accuracy: Accuracy of the model for each value of pitch.
    mean_entropy: Entropy of logit distributions.
  """
  # Defining the pitch detection model
  with tf.Graph().as_default() as graph:
    batch_size = flags["eval_batch_size"]
    threshold = flags["threshold"]
    hparams = classifier_model.get_hparams()
    hparams = pitch_classifier_utils.set_pitch_hparams(
        batch_size, hparams)

    # placeholder for a batch of the input audio
    audio_placeholder = tf.placeholder(name="audio", dtype=tf.float32)

    # Defines tf-op for getting pitch, pitch_logits, quality from given audio
    # Uses config=mag_all_1024nfft
    pitch_op, pitch_logits_op, _ = pitch_classifier_utils.get_pitch_qualities(  # pylint: disable=line-too-long
        audio_placeholder, hparams)
    pitch_distribution_op = tf.nn.softmax(pitch_logits_op)

    with tf.Session(graph=graph) as sess:
      saver = tf.train.Saver()
      # Load checkpoint for the Pitch Detection Model
      saver.restore(sess, tf.train.latest_checkpoint(
          flags.pitch_checkpoint_dir))
      predicted_pitches = []
      predicted_entropy = []
      num_batches = all_samples.shape[0] // batch_size
      for batch_idx in range(num_batches):
        print("processing batch %s / %s" %(batch_idx, num_batches))
        audio_samples = all_samples[batch_idx*batch_size:(batch_idx+1)*batch_size, :]  # pylint: disable=line-too-long
        batch_pitch, pitch_distribution = sess.run(
            [pitch_op, pitch_distribution_op], feed_dict={audio_placeholder:
                                                          audio_samples})
        batch_entropy = pitch_classifier_utils.get_batch_entropy(
            pitch_distribution)
        predicted_pitches = np.concatenate((predicted_pitches, batch_pitch))
        predicted_entropy = np.concatenate((predicted_entropy, batch_entropy))

    pitch_accuracy = pitch_classifier_utils.get_pitch_accuracy(
        predicted_pitches, all_pitches, threshold=threshold)
    mean_accuracy = np.mean(pitch_accuracy)
    mean_entropy = np.mean(predicted_entropy)
    return {"mean_pitch_accuracy": mean_accuracy,
            "mean_pitch_entropy": mean_entropy}
