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
"""MusicVAE training script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# internal imports
import tensorflow as tf

from magenta.models.music_vae import configs
from magenta.models.music_vae import data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'master', 'local',
    'The TensorFlow master to use.')
flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of NoteSequence examples. Overrides the config.')
flags.DEFINE_string(
    'run_dir', None,
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
flags.DEFINE_integer(
    'num_steps', 200000,
    'Number of training steps or `None` for infinite.')
flags.DEFINE_integer(
    'eval_num_batches', None,
    'Number of batches to use during evaluation or `None` for all batches '
    'in the data source.')
flags.DEFINE_integer(
    'checkpoints_to_keep', 100,
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
flags.DEFINE_string(
    'mode', 'train',
    'Which mode to use (`train` or `eval`).')
flags.DEFINE_string(
    'config', '',
    'The name of the config to use.')
flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values to merge '
    'with those in the config.')
flags.DEFINE_integer(
    'task', 0,
    'The task number for this worker.')
flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter server tasks.')
flags.DEFINE_integer(
    'num_sync_workers', 0,
    'The number of synchronized workers.')
flags.DEFINE_integer(
    'num_data_threads', 4,
    'The number of data preprocessing threads.')
flags.DEFINE_integer(
    'shuffle_buffer_size', 256,
    'Size of shuffle buffer.')
flags.DEFINE_string(
    'eval_dir_suffix', '', 'Suffix to add to eval output directory.')


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, examples_path, output_dir):
  """Writes a tensorboard text summary of the trial."""

  examples_path_summary = tf.summary.text(
      'examples_path', tf.constant(examples_path, name='examples_path'),
      collections=[])

  hparams_dict = hparams.values()

  # Create a markdown table from hparams.
  header = '| Key | Value |\n| :--- | :--- |\n'
  keys = sorted(hparams_dict.keys())
  lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
  hparams_table = header + '\n'.join(lines) + '\n'

  hparam_summary = tf.summary.text(
      'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
    writer.add_summary(examples_path_summary.eval())
    writer.add_summary(hparam_summary.eval())
    writer.close()


def train(train_dir,
          config,
          dataset,
          checkpoints_to_keep=5,
          num_steps=None,
          master='',
          num_sync_workers=0,
          num_ps_tasks=0,
          task=0):
  """Train loop."""
  tf.gfile.MakeDirs(train_dir)
  is_chief = (task == 0)
  _trial_summary(config.hparams, config.train_examples_path, train_dir)
  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(
        num_ps_tasks, merge_devices=True)):
      config.note_sequence_converter.is_training = True
      train_dataset = (
          dataset
          .repeat()
          .shuffle(buffer_size=FLAGS.shuffle_buffer_size))
      train_dataset = train_dataset.padded_batch(
          config.hparams.batch_size, train_dataset.output_shapes)

      iterator = train_dataset.make_one_shot_iterator()
      input_sequence, output_sequence, sequence_length = iterator.get_next()
      input_sequence.set_shape(
          [config.hparams.batch_size, None,
           config.note_sequence_converter.input_depth])
      output_sequence.set_shape(
          [config.hparams.batch_size, None,
           config.note_sequence_converter.output_depth])

      model = config.model
      model.build(config.hparams,
                  config.note_sequence_converter.output_depth,
                  is_training=True)

      optimizer = model.train(input_sequence, output_sequence, sequence_length)

      hooks = []
      if num_sync_workers:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            num_sync_workers)
        hooks.append(optimizer.make_session_run_hook(is_chief))

      gvs = optimizer.compute_gradients(model.loss)
      g = config.hparams.grad_clip
      capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
      train_op = optimizer.apply_gradients(
          capped_gvs, global_step=model.global_step, name='train_step')

      logging_dict = {'global_step': model.global_step,
                      'loss': model.loss}

      hooks.append(tf.train.LoggingTensorHook(logging_dict, every_n_iter=100))
      if num_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

      scaffold = tf.train.Scaffold(
          saver=tf.train.Saver(max_to_keep=checkpoints_to_keep))
      tf.contrib.training.train(
          train_op=train_op,
          logdir=train_dir,
          scaffold=scaffold,
          hooks=hooks,
          save_checkpoint_secs=60,
          master=master,
          is_chief=is_chief)


def evaluate(train_dir,
             eval_dir,
             config,
             dataset,
             num_batches=None,
             master=''):
  """Evaluate the model repeatedly."""
  tf.gfile.MakeDirs(eval_dir)

  _trial_summary(config.hparams, config.eval_examples_path, eval_dir)
  with tf.Graph().as_default():
    if not num_batches:
      num_batches = data.count_examples(
          config.eval_examples_path,
          config.note_sequence_converter) // config.hparams.batch_size
    eval_dataset = (
        dataset
        .padded_batch(config.hparams.batch_size, dataset.output_shapes)
        .take(num_batches))
    iterator = eval_dataset.make_one_shot_iterator()
    input_sequence, output_sequence, sequence_length = iterator.get_next()
    input_sequence.set_shape(
        [config.hparams.batch_size, None,
         config.note_sequence_converter.input_depth])
    output_sequence.set_shape(
        [config.hparams.batch_size, None,
         config.note_sequence_converter.output_depth])

    model = config.model
    model.build(config.hparams,
                config.note_sequence_converter.output_depth,
                is_training=False)

    eval_op = model.eval(input_sequence, output_sequence, sequence_length)

    hooks = [
        tf.contrib.training.StopAfterNEvalsHook(num_batches),
        tf.contrib.training.SummaryAtEndHook(eval_dir)]
    tf.contrib.training.evaluate_repeatedly(
        train_dir,
        eval_ops=eval_op,
        hooks=hooks,
        eval_interval_secs=60,
        master=master)


def run(config_map, file_reader_class=tf.data.TFRecordDataset):
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.
    file_reader_class: The tf.data.Dataset class to use for reading files.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  if not FLAGS.run_dir:
    raise ValueError('Invalid run directory: %s' % FLAGS.run_dir)
  run_dir = os.path.expanduser(FLAGS.run_dir)
  train_dir = os.path.join(run_dir, 'train')

  if FLAGS.mode not in ['train', 'eval']:
    raise ValueError('Invalid mode: %s' % FLAGS.mode)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  if FLAGS.hparams:
    config.hparams.parse(FLAGS.hparams)
  config_update_map = {}
  if FLAGS.examples_path:
    config_update_map['%s_examples_path' % FLAGS.mode] = FLAGS.examples_path
  config = configs.update_config(config, config_update_map)
  if FLAGS.num_sync_workers:
    config.hparams.batch_size //= FLAGS.num_sync_workers

  dataset = data.get_dataset(
      config,
      file_reader_class=file_reader_class,
      num_threads=FLAGS.num_data_threads,
      is_training=True)

  if FLAGS.mode == 'eval':
    eval_dir = os.path.join(run_dir, 'eval' + FLAGS.eval_dir_suffix)
    evaluate(
        train_dir,
        eval_dir,
        config=config,
        dataset=dataset,
        num_batches=FLAGS.eval_num_batches,
        master=FLAGS.master)
  elif FLAGS.mode == 'train':
    train(
        train_dir,
        config=config,
        dataset=dataset,
        checkpoints_to_keep=FLAGS.checkpoints_to_keep,
        num_steps=FLAGS.num_steps,
        master=FLAGS.master,
        num_sync_workers=FLAGS.num_sync_workers,
        num_ps_tasks=FLAGS.num_ps_tasks,
        task=FLAGS.task)


def main(unused_argv):
  run(configs.config_map)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
