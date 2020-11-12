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
"""Utilities for training."""

import collections
import copy
import functools
import random
import sys
import tensorflow.compat.v1 as tf
import tf_slim


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, model_dir, output_dir, additional_trial_info):
  """Writes a tensorboard text summary of the trial."""

  summaries_to_write = collections.OrderedDict()
  summaries_to_write['model_dir'] = model_dir
  summaries_to_write['command_line_args'] = ' \\'.join(sys.argv)

  tf.logging.info('Writing hparams summary: %s', hparams)

  hparams_dict = hparams.values()

  # Create a markdown table from hparams.
  header = '| Key | Value |\n| :--- | :--- |\n'
  keys = sorted(hparams_dict.keys())
  lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
  hparams_table = header + '\n'.join(lines) + '\n'

  summaries_to_write['hparams'] = hparams_table

  summaries_to_write.update(additional_trial_info)

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
    for name, summary in summaries_to_write.items():
      tf.logging.info('Writing summary for %s: %s', name, summary)
      writer.add_summary(
          tf.summary.text(name, tf.constant(summary, name=name),
                          collections=[]).eval())
    writer.close()


def create_estimator(model_fn,
                     model_dir,
                     hparams,
                     use_tpu=False,
                     master='',
                     tpu_cluster=None,
                     save_checkpoint_steps=300,
                     save_summary_steps=300,
                     keep_checkpoint_max=None,
                     warm_start_from=None):
  """Creates an estimator."""
  def wrapped_model_fn(features, labels, mode, params, config):
    """Wrap model_fn to restore labels value if present in features."""
    # Workaround for Estimator API that forces 'labels' to be None when in
    # predict mode.
    # https://github.com/tensorflow/tensorflow/issues/17824
    # See also infer_util.labels_to_features_wrapper
    if labels is None and hasattr(features, 'labels'):
      labels = features.labels
    return model_fn(features, labels, mode, params, config)

  if tpu_cluster:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu_cluster)
    master = None
  else:
    tpu_cluster_resolver = None

  config = tf.estimator.tpu.RunConfig(
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=save_checkpoint_steps),
      master=master,
      cluster=tpu_cluster_resolver,
      save_summary_steps=save_summary_steps,
      save_checkpoints_steps=save_checkpoint_steps,
      keep_checkpoint_max=keep_checkpoint_max,
      keep_checkpoint_every_n_hours=1)

  params = copy.deepcopy(hparams)
  params.del_hparam('batch_size')
  return tf.estimator.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=wrapped_model_fn,
      model_dir=model_dir,
      params=params,
      train_batch_size=hparams.batch_size,
      eval_batch_size=hparams.eval_batch_size,
      predict_batch_size=hparams.predict_batch_size,
      config=config,
      warm_start_from=warm_start_from,
      eval_on_tpu=False)


def train(master,
          tpu_cluster,
          model_fn,
          data_fn,
          additional_trial_info,
          model_dir,
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
      tpu_cluster=tpu_cluster,
      hparams=hparams,
      keep_checkpoint_max=keep_checkpoint_max,
      use_tpu=use_tpu)

  if estimator.config.is_chief:
    _trial_summary(
        hparams=hparams,
        model_dir=model_dir,
        output_dir=model_dir,
        additional_trial_info=additional_trial_info)

  transcription_data = functools.partial(
      data_fn,
      preprocess_examples=preprocess_examples,
      is_training=True,
      shuffle_examples=True,
      skip_n_initial_records=0)

  estimator.train(input_fn=transcription_data, max_steps=num_steps)


def evaluate(master,
             model_fn,
             data_fn,
             additional_trial_info,
             model_dir,
             preprocess_examples,
             hparams,
             name,
             num_steps=None):
  """Evaluation loop."""
  estimator = create_estimator(
      model_fn=model_fn, model_dir=model_dir, master=master, hparams=hparams)

  transcription_data_base = functools.partial(
      data_fn,
      preprocess_examples=preprocess_examples,
      is_training=False)

  if num_steps is None:
    transcription_data = functools.partial(
        transcription_data_base,
        shuffle_examples=False, skip_n_initial_records=0)
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
      dataset = transcription_data_base(
          params=record_check_params,
          shuffle_examples=False,
          skip_n_initial_records=0)
      iterator = tf.data.make_initializable_iterator(dataset)
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
      model_dir=model_dir,
      output_dir=estimator.eval_dir(name),
      additional_trial_info=additional_trial_info)

  checkpoint_path = None
  while True:
    checkpoint_path = tf_slim.evaluation.wait_for_new_checkpoint(
        model_dir, last_checkpoint=checkpoint_path)
    estimator.evaluate(input_fn=transcription_data, steps=num_steps,
                       checkpoint_path=checkpoint_path, name=name)
