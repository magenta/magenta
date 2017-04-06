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
"""Run a pretrained autoencoder model on an entire dataset, saving encodings.
"""

import getpass
import os
import sys

import numpy as np
import tensorflow as tf

from magenta.models.nsynth import reader
from magenta.models.nsynth import utils

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("master", "",
                           "BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_string("model", "ae", "Which model to use in models/")
tf.app.flags.DEFINE_string("config", "1024_nfft",
                           "Which model to use in configs/")
tf.app.flags.DEFINE_string("expdir", "",
                           "The log directory for this experiment.")
tf.app.flags.DEFINE_string("tfrecord_path", "",
                           "Path to nsynth-{train, valid, test}.tfrecord.")
tf.app.flags.DEFINE_string("savedir", "", "Where to save the embeddings.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")


def save_arrays(savedir, hparams, z_val):
  """Save arrays as npy files.

  Args:
    savedir: Directory where arrays are saved.
    hparams: Hyperparameters.
    z_val: Array to save.
  """
  z_save_val = np.array(z_val).reshape(-1, hparams.num_latent)

  save_name = os.path.join(savedir, "{}_%s.npy".format(FLAGS.dataset_split))
  with tf.gfile.Open(save_name % "z", "w") as f:
    np.save(f, z_save_val)

  tf.logging.info("Z_Save:{}".format(z_save_val.shape))
  tf.logging.info("Successfully saved to {}".format(save_name % ""))


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)


  # Make some directories
  expdir = FLAGS.expdir
  assert tf.gfile.Exists(expdir), "Can't find directory %s" % expdir
  tf.logging.info("CHECKPOINT_DIR: {}".format(expdir))
  checkpoint_path = tf.train.latest_checkpoint(expdir)
  tf.logging.info("CHECKPOINT_PATH: {}".format(checkpoint_path))
  if not checkpoint_path:
    tf.logging.info("There was a problem determining the latest checkpoint.")
    sys.exit(1)

  savedir = FLAGS.savedir
  if not tf.gfile.Exists(savedir):
    tf.gfile.MakeDirs(savedir)

  # Make the graph
  with tf.Graph().as_default():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      model = utils.get_module("models.%s" % FLAGS.model)
      hparams = model.get_hparams(FLAGS.config)

      # Load the trained model with is_training=False
      with tf.name_scope("Reader"):
        batch = reader.NSynthDataset(
            FLAGS.tfrecord_path,
            is_training=False).get_baseline_batch(hparams)

      _ = model.eval_op(batch, hparams, FLAGS.config)
      z = tf.get_collection("z")[0]

      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      sess.run(init_op)

      # Add ops to save and restore all the variables.
      # Restore variables from disk.
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Model restored.")

      # Start up some threads
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      i = 0
      z_val = []
      key_val = []
      pitch_val = []
      instrument_val = []
      instrument_family_val = []
      instrument_source_val = []
      qualities_val = []
      try:
        while True:
          if coord.should_stop():
            break
          res_val = sess.run([z])
          z_val.append(res_val[0])
          tf.logging.info("Iter: %d" % i)
          tf.logging.info("Z:{}".format(res_val[0].shape))
          i += 1
          if i + 1 % 100 == 0:
            save_arrays(savedir, hparams, z_val)
      # Report all exceptions to the coordinator, pylint: disable=broad-except
      except Exception, e:
        coord.request_stop(e)
      # pylint: enable=broad-except
      finally:
        save_arrays(savedir, hparams, z_val)
        # Terminate as usual.  It is innocuous to request stop twice.
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
  tf.app.run()
