# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Evaluates a trained ALI model.

See the README.md file for compilation and running instructions.
"""
sfrom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# internal imports
import tensorflow as tf
from magenta.models.nsynth.baseline import datasets
from magenta.models.nsynth.baseline import reader
from magenta.models.nsynth.baseline import utils

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("master",
                           "local",
                           "BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_string("checkpoint_dir", "/tmp/baseline/train",
                           "Directory where the model was written to.")
tf.app.flags.DEFINE_string("eval_dir", "/tmp/baseline/eval",
                           "Directory where the results are saved to.")
tf.app.flags.DEFINE_string("config_hparams",
                           "",
                           "Comma-delineated string of hyperparameters.")
tf.app.flags.DEFINE_string("model", "ae", "Which model to use in models/")
tf.app.flags.DEFINE_string("config",
                           "same_conv",
                           "Which model to use in configs/")
tf.app.flags.DEFINE_integer("eval_interval_secs",
                            300,
                            "Frequency, in seconds, at which evaluation is "
                            "run.")
tf.app.flags.DEFINE_string("dataset",
                           "FGP88",
                           "Which dataset to use. The data provider will "
                           "automatically retrieve the associated spectrogram "
                           "dataset.")
tf.app.flags.DEFINE_string("n_eval",
                           12000,
                           "How many data points to evaluate on.")


def main(unused_argv):
  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)

  with tf.Graph().as_default():
    # If ps_tasks is 0, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    dataset = datasets.get_dataset(FLAGS.dataset, None)
    model = utils.get_module("models.%s" % FLAGS.model)

    hparams = model.get_hparams()
    hparams.parse(FLAGS.config_hparams)
    hparams.parse("samples_per_second=%d" % dataset.samples_per_second)
    hparams.parse("num_samples=%d" % dataset.num_samples)

    with tf.name_scope("Reader"):
      batch = reader.NSynthReader(
          dataset, hparams, is_training=False).get_batch()

    eval_op = model.eval_op(batch, hparams, FLAGS.config)

    # This ensures that we make a single pass over all of the data.
    if FLAGS.dataset == "HANS_SPLIT_2048":
      n_eval = 2048
    elif FLAGS.dataset == "HANS_SPLIT_4096":
      n_eval = 4096
    else:
      n_eval = int(FLAGS.n_eval)

    num_batches = math.floor(n_eval / float(hparams.batch_size))

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_op,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == "__main__":
  tf.app.run()
