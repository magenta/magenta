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
"""Create samples from latent space.

This file performs reconstructions, latent interpolations, pitch interpolations,
and audio analogies.
"""

import getpass
import itertools
import os
import sys

# internal imports
import numpy as np
import scipy.io.wavfile
import tensorflow as tf

from magenta.models.nsynth.baseline import datasets
from magenta.models.nsynth.baseline import reader
from magenta.models.nsynth.baseline import utils

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

tf.app.flags.DEFINE_string("master", "local",
                           "BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_string("job_name", "43_mag0_hans_fm10_4000",
                           "Name of the run, determines checkpoint_dir, "
                           "save_dir")
tf.app.flags.DEFINE_string(
    "data_dir", ""
    "interpolation/gold_samples/", "Directory where the sounds are loaded from."
    "nsynth_interp_ex_().npy (fname, pitch, audio)")
tf.app.flags.DEFINE_string("model", "ae", "Which model to use in models/")
tf.app.flags.DEFINE_string("config", "mag_1_1024nfft",
                           "Which model to use in configs/")
tf.app.flags.DEFINE_string("dataset", "HANS",
                           "Which dataset to use. The data provider will "
                           "automatically retrieve the associated spectrogram "
                           "dataset.")
tf.app.flags.DEFINE_string("config_hparams", "num_latent=128,batch_size=8,"
                           "mag_only=true,n_fft=1024,fw_loss_coeff=10.0,"
                           "fw_loss_cutoff=4000",
                           "Comma-delineated string of hyperparameters.")
tf.app.flags.DEFINE_string("interp_name", "0",
                           "Name of folder to save all the files in")
tf.app.flags.DEFINE_string("user_name", "", "Name of user provided by BORG")


def get_batch_values(dataset, hparams, is_training=False):
  g = tf.Graph()
  with g.as_default():
    # Test Set Reader
    nsynth_reader = reader.NSynthReader(
        dataset, hparams, is_training=is_training)
    batch = nsynth_reader.get_batch()
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    with tf.Session(graph=g) as sess:
      sess.run(init_op)
      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      audio, pitch = sess.run([batch.spectrogram, batch.pitch])
      # Wait for threads to finish.
      coord.request_stop()
      coord.join(threads)
      return audio, pitch


def ispecgram(x, hparams, use_all=False, idx=0, num_iters=1000):
  if hparams.pad:
    # Add back in Nyquist Frequency
    mb, _, w, c = x.shape
    x = np.concatenate([x, np.zeros([mb, 1, w, c])], axis=1)
  spec = x if use_all else x[idx:idx + 1, :, :, :]
  return utils.batch_ispecgram(
      spec,
      n_fft=hparams.n_fft,
      hop_length=hparams.hop_length,
      mask=hparams.mask,
      log_mag=hparams.log_mag,
      use_cqt=hparams.use_cqt,
      re_im=hparams.re_im,
      dphase=hparams.dphase,
      mag_only=hparams.mag_only,
      num_iters=num_iters)


def specgram(audio, hparams):
  """Get spectrogram of a batch of audio.

  Args:
    audio: Batch of audio.
    hparams: Hyperparmenters.

  Returns:
    spec: Batch of spectrograms.
  """
  spec = utils.batch_specgram(
      audio,
      n_fft=hparams.n_fft,
      hop_length=hparams.hop_length,
      mask=hparams.mask,
      log_mag=hparams.log_mag,
      use_cqt=hparams.use_cqt,
      re_im=hparams.re_im,
      dphase=hparams.dphase,
      mag_only=hparams.mag_only)
  if hparams.pad:
    tf.logging.info("SPEC BEFORE PAD: {}".format(spec.shape))
    if len(spec.shape) == 5:
      spec = spec[:, :, :, :, 0]
    tf.logging.info("SPEC BEFORE PAD: {}".format(spec.shape))
    mb, h, w, c = spec.shape
    num_padding = 2**int(np.ceil(np.log(w) / np.log(2))) - w
    spec = np.concatenate([spec, np.zeros([mb, h, num_padding, c])], axis=2)
    spec = spec[:, :-1, :]
    tf.logging.info("SPEC AFTER PAD: {}".format(spec.shape))
  return spec


def save_wav(path, audio, sr=16000):
  with tf.gfile.Open(path, "w") as f:
    scipy.io.wavfile.write(f, sr, audio.ravel())


def get_minibatches(data, mb):
  """Minibatch iterator.

  Args:
    data: Array of data with minibatch as first dimension.
    mb: Minibatch size.

  Yields:
    batch: A row-slice of the data array of size == mb. Zero-padded for last
               batch.
  """
  n = data.shape[0]
  batches = n // mb
  # Extra batch if data doesn"t split evenly
  residual = n % mb
  batches = batches + 1 if residual > 0 else batches
  for i in range(batches):
    # Last Batch
    if i == batches - 1:
      if residual:
        # Pad with zeros
        dshape = list(data.shape)
        padding = np.zeros([mb - residual] + dshape[1:])
        batch = np.vstack([data[i * mb:], padding])
      else:
        batch = data[-mb:]
    # Other Batches
    else:
      batch = data[i * mb:(i + 1) * mb]
    yield batch


def main(unused_argv):
  # Set save directory and checkpoint paths
  if FLAGS.job_name:
    uname = FLAGS.user_name if FLAGS.user_name else getpass.getuser()
    base = ("/nsynth/%s/baseline/" %
            uname)
    job_dir = os.path.join(base, FLAGS.job_name)
    tf.logging.info(uname, base, job_dir)
    assert tf.gfile.Exists(job_dir), "Can't find directory %s" % job_dir
    save_dir = os.path.join(job_dir, "interp", FLAGS.interp_name)
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
      # If ps_tasks is 0, the local device is used. When using multiple
      # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
      # across the different devices.
      dataset = datasets.get_dataset(FLAGS.dataset, None)
      model = utils.get_module("models.%s" % FLAGS.model)
      hparams = model.get_hparams()

      # Set hparams from flags
      hparams.parse(FLAGS.config_hparams)
      hparams.parse("samples_per_second=%d" % dataset.samples_per_second)
      hparams.parse("num_samples=%d" % dataset.num_samples)

      with tf.name_scope("Reader"):
        batch = reader.NSynthReader(
            dataset, hparams, is_training=False).get_batch()

      # Define the graph
      _ = model.eval_op(batch, hparams, FLAGS.config)

      # Add ops to save and restore all the variables.
      # Restore variables from disk.
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Model restored.")

      x = tf.get_collection("x")[0]
      pitch = tf.get_collection("pitch")[0]
      z = tf.get_collection("z")[0]
      xhat = tf.get_collection("xhat")[0]

      # Get a batch of data if no data dir is set
      if FLAGS.data_dir:
        tf.logging.info("DATA_DIR: {}".format(FLAGS.data_dir))
        with tf.gfile.Open(
            os.path.join(FLAGS.data_dir, "nsynth_interp_ex_audio.npy"),
            "r") as f:
          x_all_val = np.load(f)
        with tf.gfile.Open(
            os.path.join(FLAGS.data_dir, "nsynth_interp_ex_pitch.npy"),
            "r") as f:
          pitch_all_val = np.load(f)
        x_all_val = specgram(x_all_val, hparams)
        pitch_all_val = pitch_all_val[:, np.newaxis]
        tf.logging.info("Original Pitches: {}".format(pitch_all_val))
      else:
        tf.logging.info("USING EVAL DATA")
        x_all_val, pitch_all_val = get_batch_values(
            dataset, hparams, is_training=False)
      tf.logging.info(
          "BATCH: {}, {}".format(x_all_val.shape, pitch_all_val.shape))

      # Run all the data through the network
      mb = hparams.batch_size
      xhat_all_val = []
      z_all_val = []
      for x_val, pitch_val in zip(
          get_minibatches(x_all_val, mb), get_minibatches(pitch_all_val, mb)):
        xhat_val, z_val = sess.run([xhat, z], {x: x_val, pitch: pitch_val})
        xhat_all_val.append(xhat_val)
        z_all_val.append(z_val)
      xhat_all_val = np.vstack(xhat_all_val)
      z_all_val = np.vstack(z_all_val)
      n_all = z_all_val.shape[0]
      tf.logging.info("Total Number of Samples:{}".format(n_all))

      # ----------------------
      # Reconstructions
      # ----------------------
      for i in xrange(n_all):
        orig_val = ispecgram(x_all_val, hparams, idx=i, num_iters=1000)
        pathorig_val = os.path.join(save_dir, "orig_{}.wav".format(i))
        save_wav(pathorig_val, orig_val)

        recon_val = ispecgram(xhat_all_val, hparams, idx=i, num_iters=1000)
        pathrecon_val = os.path.join(save_dir, "recon_{}.wav".format(i))
        save_wav(pathrecon_val, recon_val)

      # ----------------------
      # Pitch interpolation
      # ----------------------
      for j, (codes, pitch_val) in enumerate(
          zip(
              get_minibatches(z_all_val, mb), get_minibatches(
                  pitch_all_val, mb))):
        for p in [-12, -8, -5, 0, 4, 7, 12]:
          tf.logging.info("PITCH: {}".format(p))
          new_pitch_val = np.clip(pitch_val + p * np.ones(mb)[:, np.newaxis], 0,
                                  127)
          xhat_pitched_val = sess.run(xhat, {z: codes, pitch: new_pitch_val})
          audio_pitched_val = ispecgram(
              xhat_pitched_val, hparams, use_all=True, num_iters=1000)

          for i in xrange(mb):
            sample_idx = j * mb + i
            path = os.path.join(save_dir, "ex_{}_pitch_{}.wav".format(
                sample_idx, int(new_pitch_val[i][0])))
            tf.logging.info(
                "Saving {}, {}, {}, {}".format(path, sample_idx, mb, p))
            save_wav(path, audio_pitched_val[i])

      # ----------------------
      # Linear interpolation
      # ----------------------
      for idx_a, idx_b in itertools.combinations(range(n_all), 2):
        code_a = z_all_val[idx_a]
        code_b = z_all_val[idx_b]
        for linear in [False, True]:
          for i, interp_value in enumerate(np.linspace(0.25, 0.75, 3)):
            # Make a new code
            tf.logging.info(
                "Interp: {}, {}, {}".format(idx_a, idx_b, interp_value))
            if linear:
              # Linear Interpolation
              newcode = interp_value * code_b + (1.0 - interp_value) * code_a
            else:
              # Spherical Interpolation
              theta = np.arccos(code_a.ravel().dot(code_b.ravel()) / (
                  np.sum(code_a**2)**0.5 * np.sum(code_b**2)**0.5))
              newcode = (code_b * np.sin(interp_value * theta) / np.sin(theta) +
                         code_a * np.sin(
                             (1 - interp_value) * theta) / np.sin(theta))

            # Run it through the network
            newcodes = np.tile(newcode[np.newaxis, :, :, :], [mb, 1, 1, 1])
            xhat_interp = sess.run(xhat, {z: newcodes, pitch: pitch_val})

            # Save it
            audio_interp_val = ispecgram(
                xhat_interp, hparams, idx=0, num_iters=1000)
            path = os.path.join(save_dir, "{}_interp_{}-{}_{}.wav".format(
                ["spherical", "linear"][linear], idx_a, idx_b, interp_value))
            save_wav(path, audio_interp_val)

      # ----------------------
      # Analogies
      # ----------------------
      for idx_a, idx_b, idx_c in list(itertools.combinations(range(n_all), 3)):
        code_a = z_all_val[idx_a]
        code_b = z_all_val[idx_b]
        code_c = z_all_val[idx_c]

        # Make a new code
        tf.logging.info("Analogy: {}:{} as {}:?".format(idx_a, idx_b, idx_c))
        newcode = (code_b + code_c) - code_a

        # Run it through the network
        newcodes = np.tile(newcode[np.newaxis, :, :, :], [mb, 1, 1, 1])
        xhat_interp = sess.run(xhat, {z: newcodes, pitch: pitch_val})

        # Save it
        audio_interp_val = ispecgram(
            xhat_interp, hparams, idx=0, num_iters=1000)
        path = os.path.join(save_dir, "analogy_{}_{}_as_{}_.wav".format(
            idx_a, idx_b, idx_c))
        save_wav(path, audio_interp_val)


if __name__ == "__main__":
  tf.app.run()
