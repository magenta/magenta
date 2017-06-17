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
from magenta.models.nsynth.wavenet.fastgen import encode

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("source_path", "",
                           "The directory of WAVs to yield embeddings from.")
tf.app.flags.DEFINE_string("save_path", "", "The directory to save "
                           "the embeddings.")
tf.app.flags.DEFINE_string("checkpoint_path", "",
                           "A path to the checkpoint. If not given, the latest "
                           "checkpoint in `expdir` will be used.")
tf.app.flags.DEFINE_string("expdir", "",
                           "The log directory for this experiment. Required if "
                           "`checkpoint_path` is not given.")
tf.app.flags.DEFINE_integer("sample_length", 64000, "Sample length.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Sample length.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")


def main(unused_argv=None):
  tf.logging.set_verbosity(FLAGS.log)

  if FLAGS.checkpoint_path:
    checkpoint_path = utils.shell_path(FLAGS.checkpoint_path)
  else:
    expdir = utils.shell_path(FLAGS.expdir)
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

  source_path = utils.shell_path(FLAGS.source_path)
  tf.logging.info("Will load Wavs from %s." % source_path)

  save_path = utils.shell_path(FLAGS.save_path)
  tf.logging.info("Will save embeddings to %s." % save_path)
  if not tf.gfile.Exists(save_path):
    tf.logging.info("Creating save directory...")
    tf.gfile.MakeDirs(save_path)

  sample_length = FLAGS.sample_length
  batch_size = FLAGS.batch_size

  def is_wav(f):
    return f.lower().endswith(".wav")

  wavfiles = sorted([
      os.path.join(source_path, fname)
      for fname in tf.gfile.ListDirectory(source_path) if is_wav(fname)
  ])

  for start_file in xrange(0, len(wavfiles), batch_size):
    batch_number = (start_file / batch_size) + 1
    tf.logging.info("On file number %s (batch %d).", start_file, batch_number)
    end_file = start_file + batch_size
    wavefiles_batch = wavfiles[start_file:end_file]

    # Ensure that files has batch_size elements.
    batch_filler = batch_size - len(wavefiles_batch)
    wavefiles_batch.extend(batch_filler * [wavefiles_batch[-1]])
    wav_data = np.array(
        [utils.load_audio(f, sample_length) for f in wavefiles_batch])
    try:
      tf.reset_default_graph()
      # Load up the model for encoding and find the encoding
      encoding = encode(wav_data, checkpoint_path, sample_length=sample_length)
      if encoding.ndim == 2:
        encoding = np.expand_dims(encoding, 0)

      tf.logging.info("Encoding:")
      tf.logging.info(encoding.shape)
      tf.logging.info("Sample length: %d" % sample_length)

      for num, (wavfile, enc) in enumerate(zip(wavefiles_batch, encoding)):
        filename = "%s_embeddings.npy" % wavfile.split("/")[-1].strip(".wav")
        with tf.gfile.Open(os.path.join(save_path, filename), "w") as f:
          np.save(f, enc)

        if num + batch_filler + 1 == batch_size:
          break
    except Exception, e:
      tf.logging.info("Unexpected error happened: %s.", e)
      raise


def console_entry_point():
  tf.app.run(main)


if __name__ == "__main__":
  console_entry_point()
