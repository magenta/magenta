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
"""A binary for generating samples given a folder of .wav files or encodings."""

import os
import tensorflow as tf

from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.fastgen import synthesize

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("source_path", "", "Path to directory with either "
                           ".wav files or precomputed encodings in .npy files."
                           "If .wav files are present, use wav files. If no "
                           ".wav files are present, use .npy files")
tf.app.flags.DEFINE_string("encodings", False, "If True, use only .npy files.")
tf.app.flags.DEFINE_string("save_path", "", "Path to output file dir.")
tf.app.flags.DEFINE_string("checkpoint_path", "model.ckpt-200000",
                           "Path to checkpoint.")
tf.app.flags.DEFINE_integer("sample_length", 64000,
                            "Output file size in samples.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")


def main(unused_argv=None):
  source_path = utils.shell_path(FLAGS.source_path)
  checkpoint_path = utils.shell_path(FLAGS.checkpoint_path)
  save_path = utils.shell_path(FLAGS.save_path)
  if not save_path:
    raise RuntimeError("Must specify a save_path.")
  tf.logging.set_verbosity(FLAGS.log)

  # Generate from wav files
  if tf.gfile.IsDirectory(source_path):
    files = tf.gfile.ListDirectory(source_path)
    exts = [os.path.splitext(f)[1] for f in files]
    if ".wav" in exts:
      postfix = ".wav"
    elif ".npy" in exts:
      postfix = ".npy"
    else:
      raise RuntimeError("Folder must contain .wav or .npy files.")
    postfix = ".npy" if FLAGS.encodings else postfix
    files = sorted([os.path.join(source_path, fname)
                    for fname in files
                    if fname.lower().endswith(postfix)])

  elif source_path.lower().endswith(postfix):
    files = [source_path]
  else:
    files = []
  for f in files:
    out_file = os.path.join(save_path,
                            "gen_" +
                            os.path.splitext(os.path.basename(f))[0] + ".wav")
    tf.logging.info("OUTFILE %s" % out_file)
    synthesize(source_file=f,
               checkpoint_path=checkpoint_path,
               out_file=out_file,
               sample_length=FLAGS.sample_length)


def console_entry_point():
  tf.app.run(main)


if __name__ == "__main__":
  console_entry_point()
