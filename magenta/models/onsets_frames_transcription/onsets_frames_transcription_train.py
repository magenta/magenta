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

r"""Train Onsets and Frames piano transcription model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# internal imports

import tensorflow as tf

from magenta.models.onsets_frames_transcription import model
from magenta.models.onsets_frames_transcription import train_util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of train/eval examples.')
tf.app.flags.DEFINE_string(
    'run_dir', '~/tmp/onsets_frames',
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
tf.app.flags.DEFINE_string(
    'eval_dir', None,
    'Path where eval summaries will be written. If not specified, will be a '
    'subdirectory of run_dir.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Path to the checkpoint to use in `test` mode. If not provided, latest '
    'in `run_dir` will be used.')
tf.app.flags.DEFINE_integer(
    'num_steps', 50000,
    'Number of training steps or `None` for infinite.')
tf.app.flags.DEFINE_integer(
    'eval_num_batches', None,
    'Number of batches to use during evaluation or `None` for all batches '
    'in the data source.')
tf.app.flags.DEFINE_integer(
    'checkpoints_to_keep', 100,
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
tf.app.flags.DEFINE_string(
    'mode', 'train', 'Which mode to use (train, eval, or test).')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def run(hparams, run_dir):
  """Run train/eval/test."""
  train_dir = os.path.join(run_dir, 'train')

  if FLAGS.mode == 'eval':
    eval_dir = os.path.join(run_dir, 'eval')
    if FLAGS.eval_dir:
      eval_dir = os.path.join(eval_dir, FLAGS.eval_dir)
    train_util.evaluate(
        train_dir=train_dir,
        eval_dir=eval_dir,
        examples_path=FLAGS.examples_path,
        num_batches=FLAGS.eval_num_batches,
        hparams=hparams)
  elif FLAGS.mode == 'test':
    checkpoint_path = (os.path.expanduser(FLAGS.checkpoint_path)
                       if FLAGS.checkpoint_path else
                       tf.train.latest_checkpoint(train_dir))
    tf.logging.info('Testing with checkpoint: %s', checkpoint_path)
    test_dir = os.path.join(run_dir, 'test')
    train_util.test(
        checkpoint_path=checkpoint_path,
        test_dir=test_dir,
        examples_path=FLAGS.examples_path,
        num_batches=FLAGS.eval_num_batches,
        hparams=hparams)
  elif FLAGS.mode == 'train':
    train_util.train(
        train_dir=train_dir,
        examples_path=FLAGS.examples_path,
        hparams=hparams,
        checkpoints_to_keep=FLAGS.checkpoints_to_keep,
        num_steps=FLAGS.num_steps)
  else:
    raise ValueError('Invalid mode: {}'.format(FLAGS.mode))


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  run_dir = os.path.expanduser(FLAGS.run_dir)

  hparams = model.get_default_hparams()

  # Command line flags override any of the preceding hyperparameter values.
  hparams.parse(FLAGS.hparams)

  run(hparams, run_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
