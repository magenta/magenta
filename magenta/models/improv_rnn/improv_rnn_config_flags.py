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

"""Provides a class, defaults, and utils for improv RNN model configuration."""

from magenta.models.improv_rnn import improv_rnn_model
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'config',
    None,
    "Which config to use. Must be one of 'basic_improv', 'attention_improv', "
    "or 'chord_pitches_improv'.")
tf.app.flags.DEFINE_string(
    'generator_id',
    None,
    'A unique ID for the generator, overriding the default.')
tf.app.flags.DEFINE_string(
    'generator_description',
    None,
    'A description of the generator, overriding the default.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Comma-separated list of `name=value` pairs. For each pair, the value of '
    'the hyperparameter named `name` is set to `value`. This mapping is merged '
    'with the default hyperparameters.')


class ImprovRnnConfigError(Exception):
  pass


def config_from_flags():
  """Parses flags and returns the appropriate ImprovRnnConfig.

  Returns:
    The appropriate ImprovRnnConfig based on the supplied flags.

  Raises:
     ImprovRnnConfigError: When an invalid config is supplied.
  """
  if FLAGS.config not in improv_rnn_model.default_configs:
    raise ImprovRnnConfigError(
        '`--config` must be one of %s. Got %s.' % (
            improv_rnn_model.default_configs.keys(), FLAGS.config))
  config = improv_rnn_model.default_configs[FLAGS.config]
  config.hparams.parse(FLAGS.hparams)
  if FLAGS.generator_id is not None:
    config.details.id = FLAGS.generator_id
  if FLAGS.generator_description is not None:
    config.details.description = FLAGS.generator_description
  return config
