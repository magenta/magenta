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
"""Trains model using tf.slim.

See the README.md file for compilation and running instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
tf.app.flags.DEFINE_string("train_dir", "/tmp/baseline/train",
                           "Directory where to write event logs.")
tf.app.flags.DEFINE_string("dataset",
                           "FGP88",
                           "Which dataset to use. The data provider will "
                           "automatically retrieve the associated spectrogram "
                           "dataset.")
tf.app.flags.DEFINE_string("config_hparams",
                           "",
                           "Comma-delineated string of hyperparameters.")
tf.app.flags.DEFINE_string("model", "ae", "Which model to use in models/")
tf.app.flags.DEFINE_string("config",
                           "same_conv",
                           "Which model to use in configs/")
tf.app.flags.DEFINE_integer("save_summaries_secs",
                            15,
                            "Frequency at which summaries are saved, in "
                            "seconds.")
tf.app.flags.DEFINE_integer("save_interval_secs",
                            15,
                            "Frequency at which the model is saved, in "
                            "seconds.")
tf.app.flags.DEFINE_integer("ps_tasks",
                            0,
                            "Number of parameter servers. If 0, parameters "
                            "are handled locally by the worker.")
tf.app.flags.DEFINE_integer("task",
                            0,
                            "Task ID. Used when training with multiple "
                            "workers to identify each worker.")


def main(unused_argv):
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

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

    # Run the Reader on the CPU
    cpu_device = ("/job:worker/cpu:0" if FLAGS.ps_tasks else
                  "/job:localhost/replica:0/task:0/cpu:0")

    with tf.device(cpu_device):
      with tf.name_scope("Reader"):
        batch = reader.NSynthReader(
            dataset, hparams, is_training=True).get_batch()

    with tf.device(tf.ReplicaDeviceSetter(ps_tasks=FLAGS.ps_tasks)):
      train_op = model.train_op(batch, hparams, FLAGS.config)

      # Run training
      slim.learning.train(
          train_op=train_op,
          logdir=FLAGS.train_dir,
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          number_of_steps=hparams.max_steps,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


if __name__ == "__main__":
  tf.app.run()
