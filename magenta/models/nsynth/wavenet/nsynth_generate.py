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

# Lint as: python3
"""A binary for generating samples given a folder of .wav files or encodings."""

import os

from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("source_path", "", "Path to directory with either "
                           ".wav files or precomputed encodings in .npy files."
                           "If .wav files are present, use wav files. If no "
                           ".wav files are present, use .npy files")
tf.app.flags.DEFINE_boolean("npy_only", False, "If True, use only .npy files.")
tf.app.flags.DEFINE_string("save_path", "", "Path to output file dir.")
tf.app.flags.DEFINE_string("checkpoint_path", "model.ckpt-200000",
                           "Path to checkpoint.")
tf.app.flags.DEFINE_integer("sample_length", 64000,
                            "Max output file size in samples.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Number of samples per a batch.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")
tf.app.flags.DEFINE_integer("gpu_number", 0,
                            "Number of the gpu to use for multigpu generation.")


def main(unused_argv=None):
  os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_number)
  source_path = utils.shell_path(FLAGS.source_path)
  checkpoint_path = utils.shell_path(FLAGS.checkpoint_path)
  save_path = utils.shell_path(FLAGS.save_path)
  if not save_path:
    raise ValueError("Must specify a save_path.")
  tf.logging.set_verbosity(FLAGS.log)

  # Use directory of files
  if tf.gfile.IsDirectory(source_path):
    files = tf.gfile.ListDirectory(source_path)
    file_extensions = [os.path.splitext(f)[1] for f in files]
    if ".wav" in file_extensions:
      file_extension = ".wav"
    elif ".npy" in file_extensions:
      file_extension = ".npy"
    else:
      raise RuntimeError("Folder must contain .wav or .npy files.")
    file_extension = ".npy" if FLAGS.npy_only else file_extension
    files = sorted([
        os.path.join(source_path, fname)
        for fname in files
        if fname.lower().endswith(file_extension)
    ])
  # Use a single file
  elif source_path.lower().endswith((".wav", ".npy")):
    file_extension = os.path.splitext(source_path.lower())[1]
    files = [source_path]
  else:
    raise ValueError(
        "source_path {} must be a folder or file.".format(source_path))

  # Now synthesize from files one batch at a time
  batch_size = FLAGS.batch_size
  sample_length = FLAGS.sample_length
  n = len(files)
  for start in range(0, n, batch_size):
    end = start + batch_size
    batch_files = files[start:end]
    save_names = [
        os.path.join(save_path,
                     "gen_" + os.path.splitext(os.path.basename(f))[0] + ".wav")
        for f in batch_files
    ]
    # Encode waveforms
    if file_extension == ".wav":
      batch_data = fastgen.load_batch_audio(
          batch_files, sample_length=sample_length)
      encodings = fastgen.encode(
          batch_data, checkpoint_path, sample_length=sample_length)
    # Or load encodings
    else:
      encodings = fastgen.load_batch_encodings(
          batch_files, sample_length=sample_length)
    # Synthesize multi-gpu
    if FLAGS.gpu_number != 0:
      with tf.device("/device:GPU:%d" % FLAGS.gpu_number):
        fastgen.synthesize(
            encodings, save_names, checkpoint_path=checkpoint_path)
    # Single gpu
    else:
      fastgen.synthesize(
          encodings, save_names, checkpoint_path=checkpoint_path)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == "__main__":
  console_entry_point()
