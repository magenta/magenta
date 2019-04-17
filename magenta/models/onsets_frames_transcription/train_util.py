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
import random

from magenta.models.onsets_frames_transcription import data

import tensorflow as tf


def _get_data(examples, preprocess_examples, params,
              is_training, shuffle_examples=None, skip_n_initial_records=0,
              semisupervised_configs=None):
  """Gets transcription data."""
  return data.provide_batch(
      examples=examples,
      preprocess_examples=preprocess_examples,
      hparams=params,
      is_training=is_training,
      semisupervised_configs=semisupervised_configs,
      shuffle_examples=shuffle_examples,
      skip_n_initial_records=skip_n_initial_records)


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, model_dir, examples_path,
                   output_dir, semisupervised_configs=None):
  """Writes a tensorboard text summary of the trial."""
  model_dir_summary = tf.summary.text(
      'model_dir', tf.constant(model_dir, name='model_dir'), collections=[])

  data_summaries = []
  if semisupervised_configs:
    for i, ex in enumerate(semisupervised_configs):
      header = '| Path | Batch Ratio | Label Ratio |\n| :--- | :--- | :--- |\n'
      line = '| %s | %s | %s |' % (ex.examples_path,
                                   ex.batch_ratio,
                                   ex.label_ratio)
      table = header + line + '\n'
      name = 'semisupervised_data_%d' % i
      data_summaries.append(
          tf.summary.text(name, tf.constant(table, name=name), collections=[]))
  else:
    data_summaries.append(tf.summary.text(
        'examples_path', tf.constant(examples_path, name='examples_path'),
        collections=[]))

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
    for data_summary in data_summaries:
      writer.add_summary(data_summary.eval())
    writer.add_summary(model_dir_summary.eval())
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
      eval_batch_size=hparams.eval_batch_size,
      predict_batch_size=hparams.predict_batch_size,
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
          num_steps=None,
          semisupervised_configs=None):
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
        output_dir=model_dir,
        semisupervised_configs=semisupervised_configs)

  transcription_data = functools.partial(
      _get_data,
      examples=examples_path,
      preprocess_examples=preprocess_examples,
      is_training=True,
      semisupervised_configs=semisupervised_configs)

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
  estimator = create_estimator(
      model_fn=model_fn, model_dir=model_dir, master=master, hparams=hparams)

  transcription_data_base = functools.partial(
      _get_data,
      examples=examples_path,
      preprocess_examples=preprocess_examples,
      is_training=False)

  if num_steps is None:
    transcription_data = transcription_data_base
  else:
    # If num_steps is specified, we will evaluate only a subset of the data.
    #
    # The following is a hack that works around the problems of not being able
    # to determine the number of records in a given TFRecord shard without
    # reading the whole thing and not being able to persist a tf.data.Dataset
    # session across multiple estimator evaluate calls.
    #
    # This code tries to select a different subset for every evaluation by doing
    # the following:
    # - Setting shuffle_examples=True. This shuffles not only individual
    #   examples, but also shuffles the order in which shards are read.
    # - Skipping N examples before starting evaluation, where N is selected
    #   randomly for each evaluation run. This provides a different starting
    #   offset.

    # In order to skip a random number of records, we need to provide an upper
    # bound that will still let us run num_steps evaluation steps before running
    # out of data. The following code does a one-time check on startup to see
    # if there are up to num_steps * 5 records available, which would allow
    # a maximum skip range of [0, num_steps*4].
    records_to_check = num_steps * 5
    tf.logging.info('Checking for at least %d records...', records_to_check)
    records_available = 0
    with tf.Graph().as_default():
      record_check_params = copy.deepcopy(hparams)
      record_check_params.batch_size = 1
      iterator = transcription_data_base(
          params=record_check_params).make_initializable_iterator()
      next_record = iterator.get_next()
      with tf.Session() as sess:
        sess.run(iterator.initializer)
        try:
          for i in range(records_to_check):
            del i
            sess.run(next_record)
            records_available += 1
            if records_available % 10 == 0:
              tf.logging.info('Found %d records...', records_available)
        except tf.errors.OutOfRangeError:
          pass
    # Determine max number of records we could skip and still have num_steps
    # records remaining.
    max_records_to_skip = max(0, records_available - num_steps)
    tf.logging.info('Found at least %d records. '
                    'Will skip a maximum of %d records during eval runs '
                    'in order to support %d evaluation steps.',
                    records_available, max_records_to_skip, num_steps)

    # Since we're doing a limited number of steps, we should shuffle the
    # examples we're evaluating so each evaluation is over a different portion
    # of the dataset.
    def transcription_data(params, *args, **kwargs):
      assert not args
      skip_n_initial_records = random.randint(0, max_records_to_skip)
      tf.logging.info('Skipping %d initial record(s)', skip_n_initial_records)
      return transcription_data_base(
          params=params,
          shuffle_examples=True,
          skip_n_initial_records=skip_n_initial_records,
          **kwargs)

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
