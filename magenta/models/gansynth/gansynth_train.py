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

r"""Train a progressive GANSynth model.

Example usage: (From base directory)
>>> python magenta/models/gansynth/train.py

To use a config of hyperparameters:
>>> python magenta/models/gansynth/train.py --config=mel_prog_hires

To use a config of hyperparameters and manual hparams:
>>> python magenta/models/gansynth/train.py --config=mel_prog_hires \
>>> --hparams='{"train_data_path":"/path/to/nsynth-train.tfrecord"}'

List of hyperparameters can be found in model.py.
Trains in a couple days on a single V100 GPU.

Adapted from the original Progressive GAN paper for images.
See https://arxiv.org/abs/1710.10196 for details about the model.
See https://github.com/tkarras/progressive_growing_of_gans for the original
theano implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import json
import os
import time

from absl import logging
import absl.flags
from magenta.models.gansynth.lib import data_helpers
from magenta.models.gansynth.lib import data_normalizer
from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib import train_util
from magenta.models.gansynth.lib import util
import tensorflow.compat.v1 as tf


absl.flags.DEFINE_string('hparams', '{}', 'Flags dict as JSON string.')
absl.flags.DEFINE_string('config', '', 'Name of config module.')
FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def init_data_normalizer(config):
  """Initializes data normalizer."""
  normalizer = data_normalizer.registry[config['data_normalizer']](config)
  if normalizer.exists():
    return

  if config['task'] == 0:
    tf.reset_default_graph()
    data_helper = data_helpers.registry[config['data_type']](config)
    real_images, _ = data_helper.provide_data(batch_size=10)

    # Save normalizer.
    # Note if normalizer has been saved, save() is no-op. To regenerate the
    # normalizer, delete the normalizer file in train_root_dir/assets
    normalizer.save(real_images)
  else:
    while not normalizer.exists():
      time.sleep(5)


def run(config):
  """Entry point to run training."""
  init_data_normalizer(config)

  stage_ids = train_util.get_stage_ids(**config)
  if not config['train_progressive']:
    stage_ids = list(stage_ids)[-1:]

  # Train one stage at a time
  for stage_id in stage_ids:
    batch_size = train_util.get_batch_size(stage_id, **config)
    tf.reset_default_graph()
    with tf.device(tf.train.replica_device_setter(config['ps_tasks'])):
      model = lib_model.Model(stage_id, batch_size, config)
      model.add_summaries()
      print('Variables:')
      for v in tf.global_variables():
        print('\t', v.name, v.get_shape().as_list())
      logging.info('Calling train.train')
      train_util.train(model, **config)


def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True
  # Set hyperparams from json args and defaults
  flags = lib_flags.Flags()
  # Config hparams
  if FLAGS.config:
    config_module = importlib.import_module(
        'magenta.models.gansynth.configs.{}'.format(FLAGS.config))
    flags.load(config_module.hparams)
  # Command line hparams
  flags.load_json(FLAGS.hparams)
  # Set default flags
  lib_model.set_flags(flags)

  print('Flags:')
  flags.print_values()

  # Create training directory
  flags['train_root_dir'] = util.expand_path(flags['train_root_dir'])
  if not tf.gfile.Exists(flags['train_root_dir']):
    tf.gfile.MakeDirs(flags['train_root_dir'])

  # Save the flags to help with loading the model latter
  fname = os.path.join(flags['train_root_dir'], 'experiment.json')
  with tf.gfile.Open(fname, 'w') as f:
    json.dump(flags, f)  # pytype: disable=wrong-arg-types

  # Run training
  run(flags)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
