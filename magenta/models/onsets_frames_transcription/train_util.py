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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import random
import sys
import tensorflow as tf

# Should not be called from within the graph to avoid redundant summaries.
from magenta.models.onsets_frames_transcription.model_util import ModelWrapper, ModelType


def _trial_summary(hparams, model_dir, output_dir, additional_trial_info):
    """Writes a tensorboard text summary of the trial."""

    summaries_to_write = collections.OrderedDict()
    summaries_to_write['model_dir'] = model_dir
    summaries_to_write['command_line_args'] = ' \\'.join(sys.argv)

    tf.compat.v1.logging.info('Writing hparams summary: %s', hparams)

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
            tf.compat.v1.logging.info('Writing summary for %s: %s', name, summary)
            writer.add_summary(
                tf.summary.text(name, tf.constant(summary, name=name),
                                collections=[]).eval())
        writer.close()


def train(master,
          data_fn,
          additional_trial_info,
          model_dir,
          preprocess_examples,
          hparams,
          keep_checkpoint_max,
          use_tpu,
          num_steps=50):
    """Train loop."""

    transcription_data = functools.partial(
        data_fn,
        preprocess_examples=preprocess_examples,
        is_training=True,
        shuffle_examples=True,
        skip_n_initial_records=0)

    midi_model = ModelWrapper('./models', ModelType.MIDI, id=hparams.model_id,
                              dataset=transcription_data(params=hparams),
                              batch_size=hparams.batch_size, steps_per_epoch=10, hparams=hparams)
    midi_model.load_model(72.20, 65.00)

    for i in range(num_steps):
        midi_model.train_and_save(epochs=hparams.epochs_per_save)

    # estimator.train(input_fn=transcription_data, max_steps=num_steps)


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
        tf.compat.v1.logging.info('Checking for at least %d records...', records_to_check)
        records_available = 0
        with tf.Graph().as_default():
            record_check_params = copy.deepcopy(hparams)
            record_check_params.batch_size = 1
            iterator = transcription_data_base(
                params=record_check_params,
                shuffle_examples=False,
                skip_n_initial_records=0,
            ).make_initializable_iterator()
            next_record = iterator.get_next()
            with tf.Session() as sess:
                sess.run(iterator.initializer)
                try:
                    for i in range(records_to_check):
                        del i
                        sess.run(next_record)
                        records_available += 1
                        if records_available % 10 == 0:
                            tf.compat.v1.logging.info('Found %d records...', records_available)
                except tf.errors.OutOfRangeError:
                    pass
        # Determine max number of records we could skip and still have num_steps
        # records remaining.
        max_records_to_skip = max(0, records_available - num_steps)
        tf.compat.v1.logging.info('Found at least %d records. '
                                  'Will skip a maximum of %d records during eval runs '
                                  'in order to support %d evaluation steps.',
                                  records_available, max_records_to_skip, num_steps)

        # Since we're doing a limited number of steps, we should shuffle the
        # examples we're evaluating so each evaluation is over a different portion
        # of the dataset.
        def transcription_data(params, *args, **kwargs):
            assert not args
            skip_n_initial_records = random.randint(0, max_records_to_skip)
            tf.compat.v1.logging.info('Skipping %d initial record(s)', skip_n_initial_records)
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
        checkpoint_path = contrib_training.wait_for_new_checkpoint(
            model_dir, last_checkpoint=checkpoint_path)
        estimator.evaluate(input_fn=transcription_data, steps=num_steps,
                           checkpoint_path=checkpoint_path, name=name)
