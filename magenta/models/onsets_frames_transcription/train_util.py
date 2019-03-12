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

import copy
import functools

from magenta.models.onsets_frames_transcription import data

import tensorflow as tf


def _get_data(examples, preprocess_examples, params, is_training):
  """Gets transcription data."""
  return data.provide_batch(
      examples=examples,
      preprocess_examples=preprocess_examples,
      hparams=params,
      is_training=is_training)


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, model_dir, examples_path, output_dir):
  """Writes a tensorboard text summary of the trial."""
  model_dir_summary = tf.summary.text(
      'model_dir', tf.constant(model_dir, name='model_dir'), collections=[])

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
    writer.add_summary(model_dir_summary.eval())
    writer.add_summary(examples_path_summary.eval())
    writer.add_summary(hparam_summary.eval())
    writer.close()


def create_estimator(model_fn,
                     model_dir,
                     hparams,
                     use_tpu=False,
                     master='',
                     save_checkpoint_steps=300,
                     save_summary_steps=300,
                     keep_checkpoint_max=None,
                     warm_start_from=None):
  """Creates an estimator."""
  config = tf.contrib.tpu.RunConfig(
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=save_checkpoint_steps),
      master=master,
      save_summary_steps=save_summary_steps,
      save_checkpoints_steps=save_checkpoint_steps,
      keep_checkpoint_max=keep_checkpoint_max,
      keep_checkpoint_every_n_hours=1)

  params = copy.deepcopy(hparams)
  params.del_hparam('batch_size')
  return tf.contrib.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      model_dir=model_dir,
      params=params,
      train_batch_size=hparams.batch_size,
      eval_batch_size=hparams.batch_size,
      predict_batch_size=hparams.batch_size,
      config=config,
      warm_start_from=warm_start_from,
      eval_on_tpu=False)


def train(master,
          model_fn,
          model_dir,
          examples_path,
          preprocess_examples,
          hparams,
          keep_checkpoint_max,
          use_tpu,
          num_steps=None):
  """Train loop."""
  estimator = create_estimator(
      model_fn=model_fn,
      model_dir=model_dir,
      master=master,
      hparams=hparams,
      keep_checkpoint_max=keep_checkpoint_max,
      use_tpu=use_tpu)

  if estimator.config.is_chief:
    _trial_summary(
        hparams=hparams,
        examples_path=examples_path,
        model_dir=model_dir,
        output_dir=model_dir)

  transcription_data = functools.partial(
      _get_data,
      examples=examples_path,
      preprocess_examples=preprocess_examples,
      is_training=True)

  estimator.train(input_fn=transcription_data, max_steps=num_steps)


def evaluate(master,
             model_fn,
             model_dir,
             examples_path,
             preprocess_examples,
             hparams,
             name,
             num_steps=None):
  """Train loop."""
  hparams.batch_size = 1

  estimator = create_estimator(
      model_fn=model_fn, model_dir=model_dir, master=master, hparams=hparams)

  transcription_data = functools.partial(
      _get_data,
      examples=examples_path,
      preprocess_examples=preprocess_examples,
      is_training=False)

  _trial_summary(
      hparams=hparams,
      examples_path=examples_path,
      model_dir=model_dir,
      output_dir=estimator.eval_dir(name))

  checkpoint_path = None
  while True:
    checkpoint_path = tf.contrib.training.wait_for_new_checkpoint(
        model_dir, last_checkpoint=checkpoint_path)
    estimator.evaluate(input_fn=transcription_data, steps=num_steps, name=name)
