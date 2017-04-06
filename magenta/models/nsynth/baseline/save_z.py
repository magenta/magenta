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

from magenta.models.nsynth.baseline import datasets
from magenta.models.nsynth.baseline import reader
from magenta.models.nsynth.baseline import utils

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("master", "local",
                           "BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_string("job_name", "43_mag0_hans_fm10_4000",
                           "Name of the run, determines checkpoint_dir, "
                           "save_dir")
tf.app.flags.DEFINE_string("model", "ae", "Which model to use in models/")
tf.app.flags.DEFINE_string("config", "mag_1_1024nfft",
                           "Which model to use in configs/")
tf.app.flags.DEFINE_string("dataset", "NSYNTH_RC4_TEST",
                           "Which dataset to use. The data provider will "
                           "automatically retrieve the associated spectrogram "
                           "dataset.")
tf.app.flags.DEFINE_string("dataset_split", "train",
                           "Which subset of data to use (train, test)")
tf.app.flags.DEFINE_string("config_hparams",
                           "num_latent=128,batch_size=8,mag_only=true,"
                           "n_fft=1024,fw_loss_coeff=10.0,fw_loss_cutoff=4000",
                           "Comma-delineated string of hyperparameters.")
tf.app.flags.DEFINE_string("user_name", "", "Name of user provided by BORG")


def save_arrays(save_dir, hparams, z_val, key_val, pitch_val, instrument_val,
                instrument_family_val, instrument_source_val, qualities_val):
  """Save arrays as npy files.

  Args:
    save_dir: Directory where arrays are saved.
    hparams: Hyperparameters.
    z_val: Array to save.
    key_val: Array to save.
    pitch_val: Array to save.
    instrument_val: Array to save.
    instrument_family_val: Array to save.
    instrument_source_val: Array to save.
    qualities_val: Array to save.
  """
  z_save_val = np.array(z_val).reshape(-1, hparams.num_latent)
  key_save_val = np.array(key_val).ravel()
  pitch_save_val = np.array(pitch_val).ravel()
  instrument_save_val = np.array(instrument_val).ravel()
  instrument_family_save_val = np.array(instrument_family_val).ravel()
  instrument_source_save_val = np.array(instrument_source_val).ravel()
  qualities_save_val = np.array(qualities_val).reshape(-1, 10)

  save_name = os.path.join(save_dir, "{}_%s.npy".format(FLAGS.dataset_split))
  with tf.gfile.Open(save_name % "z", "w") as f:
    np.save(f, z_save_val)
  with tf.gfile.Open(save_name % "key", "w") as f:
    np.save(f, key_save_val)
  with tf.gfile.Open(save_name % "pitch", "w") as f:
    np.save(f, pitch_save_val)
  with tf.gfile.Open(save_name % "instrument", "w") as f:
    np.save(f, instrument_save_val)
  with tf.gfile.Open(save_name % "instrument_family", "w") as f:
    np.save(f, instrument_family_save_val)
  with tf.gfile.Open(save_name % "instrument_source", "w") as f:
    np.save(f, instrument_source_save_val)
  with tf.gfile.Open(save_name % "qualities", "w") as f:
    np.save(f, qualities_save_val)

  tf.logging.info("Successfully saved to {}".format(save_name % ""))
  tf.logging.info("Z_Save:{}".format(z_save_val.shape))
  tf.logging.info("Key_Save:{}".format(key_save_val.shape))
  tf.logging.info("Pitch_Save:{}".format(pitch_save_val.shape))


def main(unused_argv):
  # Make some directories
  if FLAGS.job_name:
    uname = FLAGS.user_name if FLAGS.user_name else getpass.getuser()
    base = ("nsynth/%s/baseline/" %
            uname)
    job_dir = os.path.join(base, FLAGS.job_name)
    tf.logging.info(uname, base, job_dir)
    assert tf.gfile.Exists(job_dir), "Can't find directory %s" % job_dir
    save_dir = os.path.join(job_dir, "save_z")
    checkpoint_dir = os.path.join(job_dir, "train")

  tf.logging.info("SAVE_DIR: {}".format(save_dir))
  tf.logging.info("CHECKPOINT_DIR: {}".format(checkpoint_dir))

  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  tf.logging.info("CHECKPOINT_PATH: {}".format(checkpoint_path))
  if not checkpoint_path:
    tf.logging.info("There was a problem determining the latest checkpoint.")
    sys.exit(1)

  if not tf.gfile.Exists(save_dir):
    tf.gfile.MakeDirs(save_dir)

  # Make the graph
  with tf.Graph().as_default():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      dataset = datasets.get_dataset(FLAGS.dataset, None)
      model = utils.get_module("models.%s" % FLAGS.model)

      hparams = model.get_hparams()
      hparams.parse(FLAGS.config_hparams)
      hparams.parse("samples_per_second=%d" % dataset.samples_per_second)
      hparams.parse("num_samples=%d" % dataset.num_samples)

      is_training = (FLAGS.dataset_split == "train")

      # Load the trained model with is_training=False
      with tf.name_scope("Reader"):
        batch = reader.NSynthReader(
            dataset, hparams, num_epochs=1,
            is_training=is_training).get_batch()

      if is_training:
        _ = model.train_op(batch, hparams, FLAGS.config)
      else:
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
          res_val = sess.run([
              z, batch.key, batch.pitch, batch.instrument,
              batch.instrument_family, batch.instrument_source, batch.qualities
          ])
          z_val.append(res_val[0])
          key_val.append(res_val[1])
          pitch_val.append(res_val[2])
          instrument_val.append(res_val[3])
          instrument_family_val.append(res_val[4])
          instrument_source_val.append(res_val[5])
          qualities_val.append(res_val[6])
          tf.logging.info("Iter: %d" % i)
          tf.logging.info("Z:{}".format(res_val[0].shape))
          tf.logging.info("Key:{}".format(res_val[1]))
          tf.logging.info("Pitch:{}".format(res_val[2]))
          tf.logging.info("Instrument:{}".format(res_val[3]))
          tf.logging.info("InstrumentFamily:{}".format(res_val[4]))
          tf.logging.info("InstrumentSource:{}".format(res_val[5]))
          tf.logging.info("Qalities:{}".format(res_val[6]))
          i += 1
          if i + 1 % 1000 == 0:
            save_arrays(save_dir, hparams, z_val, key_val, pitch_val,
                        instrument_val, instrument_family_val,
                        instrument_source_val, qualities_val)
      # Report all exceptions to the coordinator, pylint: disable=broad-except
      except Exception, e:
        coord.request_stop(e)
      # pylint: enable=broad-except
      finally:
        save_arrays(save_dir, hparams, z_val, key_val, pitch_val,
                    instrument_val, instrument_family_val,
                    instrument_source_val, qualities_val)
        # Terminate as usual.  It is innocuous to request stop twice.
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
  tf.app.run()
