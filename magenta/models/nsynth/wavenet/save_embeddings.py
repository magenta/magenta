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
tf.app.flags.DEFINE_integer("sample_rate", 16000, "Sample rate.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Sample length.")
tf.app.flags.DEFINE_integer("file_pattern", (".wav", ".mp3"), "File types to load")
tf.app.flags.DEFINE_integer("normalize", True, "If loaded files should be normalized")
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
  file_pattern = FLAGS.file_pattern
  tf.logging.info("Will load Wavs from %s." % wavdir)
  sample_rate = FLAGS.sample_rate
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

    def is_audio_file(f):
      return f.lower().endswith(file_pattern)

    wavfiles = sorted([
        os.path.join(wavdir, fname) for fname in tf.gfile.ListDirectory(wavdir)
        if is_audio_file(fname)
    ])

    for start_file in xrange(0, len(wavfiles), batch_size):
        batch_number = (start_file / batch_size) + 1
        end_file = start_file + batch_size
        files = wavfiles[start_file:end_file]

        # Ensure that files has batch_size elements.
        batch_filler = batch_size - len(files)
        files.extend(batch_filler * [files[-1]])

        for f in files:
            tf.logging.info("On file %s of %d (batch %d).", start_file, len(wavfiles), batch_number)
            encoding = []
            wavdata = utils.load_wav(f, sample_rate)
            total_length = len(wavdata)
            a = 0
            while len(wavdata) > 0:
                a = a + sample_length
                if len(wavdata) > sample_length:
                    piece = wavdata[:sample_length]
                    wavdata = wavdata[sample_length:]
                else:
                    piece = wavdata
                    wavdata = []
                if len(piece) < sample_length:
                    piece_factor = len(piece) / sample_length
                    piece = np.pad(piece, (0,(sample_length-len(piece))), mode='constant', constant_values=0)
                else:
                    piece_factor = None
                try:
                    piece = np.reshape(piece, (-1, sample_length))
                    pred = sess.run(
                            graph_encoding, feed_dict={wav_placeholder: piece})
                    if piece_factor is None:
                        encoding.append(pred.reshape(-1, config.ae_bottleneck_width))
                    else:
                        final_chunk_length = min(int(len(pred)*piece_factor)+1, len(pred))
                        encoding.append(pred[:final_chunk_length].reshape(-1, config.ae_bottleneck_width))
                    tf.logging.info("Classified: %i of %i samples", a, total_length)
                except Exception as e:
                    tf.logging.info("Unexpected error happened: %s.", e)
            filename = "%s_embeddings.npy" % f.split("/")[-1].strip(".wav")
            with tf.gfile.Open(os.path.join(savedir, filename), "w") as outfile:
                np.save(outfile, np.asarray(encoding).reshape(-1, config.ae_bottleneck_width))

if __name__ == "__main__":
  tf.app.run()
