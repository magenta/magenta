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

"""Trains model using tf.slim."""
from magenta.models.nsynth import reader
from magenta.models.nsynth import utils
import tensorflow.compat.v1 as tf
import tf_slim as slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("master",
                           "",
                           "BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_string("logdir", "/tmp/baseline/train",
                           "Directory where to write event logs.")
tf.app.flags.DEFINE_string("train_path",
                           "",
                           "Path the nsynth-train.tfrecord.")
tf.app.flags.DEFINE_string("model", "ae", "Which model to use in models/")
tf.app.flags.DEFINE_string("config",
                           "nfft_1024",
                           "Which config to use in models/configs/")
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
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  if not tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.MakeDirs(FLAGS.logdir)

  with tf.Graph().as_default():

    # If ps_tasks is 0, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    model = utils.get_module("baseline.models.%s" % FLAGS.model)
    hparams = model.get_hparams(FLAGS.config)

    # Run the Reader on the CPU
    if FLAGS.ps_tasks:
      cpu_device = "/job:worker/cpu:0"
    else:
      cpu_device = "/job:localhost/replica:0/task:0/cpu:0"

    with tf.device(cpu_device):
      with tf.name_scope("Reader"):
        batch = reader.NSynthDataset(
            FLAGS.train_path, is_training=True).get_baseline_batch(hparams)

    with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks)):
      train_op = model.train_op(batch, hparams, FLAGS.config)

      # Run training
      slim.learning.train(
          train_op=train_op,
          logdir=FLAGS.logdir,
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          number_of_steps=hparams.max_steps,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.app.run()
