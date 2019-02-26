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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import model

import tensorflow as tf


def _get_data(examples_path, hparams, is_training):
  """Gets transcription data."""
  hparams_dict = hparams.values()
  return data.provide_batch(
      hparams.batch_size,
      examples=examples_path,
      hparams=hparams,
      truncated_length=hparams_dict.get('truncated_length', None),
      is_training=is_training)


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, examples_path, output_dir):
  """Writes a tensorboard text summary of the trial."""

  examples_path_summary = tf.summary.text(
      'examples_path', tf.constant(examples_path, name='examples_path'),
      collections=[])

  tf.logging.info('Writing hparams summary: %s', hparams)

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


def create_estimator(model_dir,
                     hparams,
                     master='',
                     keep_checkpoint_max=None,
                     warm_start_from=None):
  """Creates an estimator."""

  class MasterRunConfig(tf.estimator.RunConfig):
    """Hack to allow setting master in RunConfig via a flag."""

    def __init__(self, master, *unused_args, **unused_kwargs):
      super(MasterRunConfig, self).__init__(*unused_args, **unused_kwargs)
      self._master = master

  config = MasterRunConfig(
      master=master,
      train_distribute=tf.distribute.MirroredStrategy(),
      save_summary_steps=100,
      save_checkpoints_steps=300,
      keep_checkpoint_max=keep_checkpoint_max,
      keep_checkpoint_every_n_hours=1)

  return tf.estimator.Estimator(
      model_fn=model.model_fn,
      model_dir=model_dir,
      params=hparams,
      config=config,
      warm_start_from=warm_start_from)


def train(master,
          model_dir,
          examples_path,
          hparams,
          keep_checkpoint_max,
          num_steps=None):
  """Train loop."""
  estimator = create_estimator(
      model_dir=model_dir, master=master, hparams=hparams,
      keep_checkpoint_max=keep_checkpoint_max)

  if estimator.config.is_chief:
    _trial_summary(hparams, examples_path, model_dir)

  transcription_data = functools.partial(
      _get_data, examples_path, hparams, is_training=True)

  estimator.train(input_fn=transcription_data, max_steps=num_steps)
