# Copyright 2017 Google Inc. All Rights Reserved.
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
"""With a trained model, compute the embeddings on a directory of WAV files."""

import os
import sys

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.nsynth import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("expdir", "",
                           "The log directory for this experiment. Required if "
                           "`checkpoint_path` is not given.")
tf.app.flags.DEFINE_string("checkpoint_path", "",
                           "A path to the checkpoint. If not given, the latest "
                           "checkpoint in `expdir` will be used.")
tf.app.flags.DEFINE_string("wavdir", "",
                           "The directory of WAVs to yield embeddings from.")
tf.app.flags.DEFINE_string("savedir", "", "Where to save the embeddings.")
tf.app.flags.DEFINE_string("config", "h512_bo16", "Model configuration name")
tf.app.flags.DEFINE_integer("sample_length", 64000, "Sample length.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Sample length.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")


def main(unused_argv=None):
  tf.logging.set_verbosity(FLAGS.log)

  if FLAGS.config is None:
    raise RuntimeError("No config name specified.")

  config = utils.get_module("wavenet." + FLAGS.config).Config()

  if FLAGS.checkpoint_path:
    checkpoint_path = FLAGS.checkpoint_path
  else:
    expdir = FLAGS.expdir
    tf.logging.info("Will load latest checkpoint from %s.", expdir)
    while not tf.gfile.Exists(expdir):
      tf.logging.fatal("\tExperiment save dir '%s' does not exist!", expdir)
      sys.exit(1)

    try:
      checkpoint_path = tf.train.latest_checkpoint(expdir)
    except tf.errors.NotFoundError:
      tf.logging.fatal("There was a problem determining the latest checkpoint.")
      sys.exit(1)

  if not tf.train.checkpoint_exists(checkpoint_path):
    tf.logging.fatal("Invalid checkpoint path: %s", checkpoint_path)
    sys.exit(1)

  tf.logging.info("Will restore from checkpoint: %s", checkpoint_path)

  wavdir = FLAGS.wavdir
  tf.logging.info("Will load Wavs from %s." % wavdir)

  savedir = FLAGS.savedir
  tf.logging.info("Will save embeddings to %s." % savedir)
  if not tf.gfile.Exists(savedir):
    tf.logging.info("Creating save directory...")
    tf.gfile.MakeDirs(savedir)

  tf.logging.info("Building graph")
  with tf.Graph().as_default(), tf.device("/gpu:0"):
    sample_length = FLAGS.sample_length
    batch_size = FLAGS.batch_size
    wav_placeholder = tf.placeholder(
        tf.float32, shape=[batch_size, sample_length])
    graph = config.build({"wav": wav_placeholder}, is_training=False)
    graph_encoding = graph["encoding"]

    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    variables_to_restore = ema.variables_to_restore()

    # Create a saver, which is used to restore the parameters from checkpoints
    saver = tf.train.Saver(variables_to_restore)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    # Set the opt_level to prevent py_funcs from being executed multiple times.
    session_config.graph_options.optimizer_options.opt_level = 2
    sess = tf.Session("", config=session_config)

    tf.logging.info("\tRestoring from checkpoint.")
    saver.restore(sess, checkpoint_path)

    def is_wav(f):
      return f.lower().endswith(".wav")

    wavfiles = sorted([
        os.path.join(wavdir, fname) for fname in tf.gfile.ListDirectory(wavdir)
        if is_wav(fname)
    ])

    for start_file in xrange(0, len(wavfiles), batch_size):
      batch_number = (start_file / batch_size) + 1
      tf.logging.info("On file number %s (batch %d).", start_file, batch_number)
      end_file = start_file + batch_size
      files = wavfiles[start_file:end_file]

      # Ensure that files has batch_size elements.
      batch_filler = batch_size - len(files)
      files.extend(batch_filler * [files[-1]])

      wavdata = np.array([utils.load_wav(f)[:sample_length] for f in files])

      try:
        encoding = sess.run(
            graph_encoding, feed_dict={wav_placeholder: wavdata})
        for num, (wavfile, enc) in enumerate(zip(wavfiles, encoding)):
          filename = "%s_embeddings.npy" % wavfile.split("/")[-1].strip(".wav")
          with tf.gfile.Open(os.path.join(savedir, filename), "w") as f:
            np.save(f, enc)

          if num + batch_filler + 1 == batch_size:
            break
      except Exception, e:
        tf.logging.info("Unexpected error happened: %s.", e)
        raise


if __name__ == "__main__":
  tf.app.run()
