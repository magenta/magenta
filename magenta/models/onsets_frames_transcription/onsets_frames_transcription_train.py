# Copyright 2019 The Magenta Authors.
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

r"""Train Onsets and Frames piano transcription model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import model
from magenta.models.onsets_frames_transcription import train_util

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('master', '',
                           'Name of the TensorFlow runtime to use.')
tf.app.flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of train/eval examples.')
tf.app.flags.DEFINE_string(
    'model_dir', '~/tmp/onsets_frames',
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
tf.app.flags.DEFINE_integer('num_steps', 1000000,
                            'Number of training steps or `None` for infinite.')
tf.app.flags.DEFINE_integer(
    'keep_checkpoint_max', 100,
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def run(hparams, model_dir):
  """Run train/eval/test."""
  train_util.train(
      master=FLAGS.master,
      model_dir=model_dir,
      examples_path=FLAGS.examples_path,
      hparams=hparams,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      num_steps=FLAGS.num_steps)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)
  tf.app.flags.mark_flags_as_required(['examples_path'])

  model_dir = os.path.expanduser(FLAGS.model_dir)

  hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS,
                                   model.get_default_hparams())

  # Command line flags override any of the preceding hyperparameter values.
  hparams.parse(FLAGS.hparams)

  run(hparams, model_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
