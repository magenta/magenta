# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Provides a class, defaults, and utils for Melody RNN model configuration."""

# internal imports
import tensorflow as tf

import magenta
from magenta.models.melody_rnn import melody_rnn_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'config',
    None,
    "Which config to use. Must be one of 'basic', 'lookback', or 'attention'. "
    "Mutually exclusive with `--melody_encoder_decoder`.")
tf.app.flags.DEFINE_string(
    'melody_encoder_decoder',
    None,
    "Which encoder/decoder to use. Must be one of 'onehot', 'lookback', or "
    "'key'. Mutually exclusive with `--config`.")
tf.app.flags.DEFINE_string(
    'generator_id',
    None,
    'A unique ID for the generator. Overrides the default if `--config` is '
    'also supplied.')
tf.app.flags.DEFINE_string(
    'generator_description',
    None,
    'A description of the generator. Overrides the default if `--config` is '
    'also supplied.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Comma-separated list of `name=value` pairs. For each pair, the value of '
    'the hyperparameter named `name` is set to `value`. This mapping is merged '
    'with the default hyperparameters.')


class MelodyRnnConfigFlagsException(Exception):
  pass


def one_hot_melody_encoder_decoder(min_note, max_note):
  """Return a OneHotEventSequenceEncoderDecoder for melodies.

  Args:
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.

  Returns:
    A melody OneHotEventSequenceEncoderDecoder.
  """
  return magenta.music.OneHotEventSequenceEncoderDecoder(
      magenta.music.MelodyOneHotEncoding(min_note, max_note))


def lookback_melody_encoder_decoder(min_note, max_note):
  """Return a LookbackEventSequenceEncoderDecoder for melodies.

  Args:
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.

  Returns:
    A melody LookbackEventSequenceEncoderDecoder.
  """
  return magenta.music.LookbackEventSequenceEncoderDecoder(
      magenta.music.MelodyOneHotEncoding(min_note, max_note))


# Dictionary of functions that take `min_note` and `max_note` and return the
# appropriate EventSequenceEncoderDecoder object.
melody_encoder_decoders = {
    'onehot': one_hot_melody_encoder_decoder,
    'lookback': lookback_melody_encoder_decoder,
    'key': magenta.music.KeyMelodyEncoderDecoder
}


def config_from_flags():
  """Parses flags and returns the appropriate MelodyRnnConfig.

  If `--config` is supplied, returns the matching default MelodyRnnConfig after
  updating the hyperparameters based on `--hparams`.

  If `--melody_encoder_decoder` is supplied, returns a new MelodyRnnConfig using
  the matching EventSequenceEncoderDecoder, generator details supplied by
  `--generator_id` and `--generator_description`, and hyperparameters based on
  `--hparams`.

  Returns:
    The appropriate MelodyRnnConfig based on the supplied flags.

  Raises:
     MelodyRnnConfigFlagsException: When not exactly one of `--config` or
         `melody_encoder_decoder` is supplied.
  """
  if (FLAGS.melody_encoder_decoder, FLAGS.config).count(None) != 1:
    raise MelodyRnnConfigFlagsException(
        'Exactly one of `--config` or `--melody_encoder_decoder` must be '
        'supplied.')

  if FLAGS.melody_encoder_decoder is not None:
    if FLAGS.melody_encoder_decoder not in melody_encoder_decoders:
      raise MelodyRnnConfigFlagsException(
          '`--melody_encoder_decoder` must be one of %s. Got %s.' % (
              melody_encoder_decoders.keys(), FLAGS.melody_encoder_decoder))
    if FLAGS.generator_id is not None:
      generator_details = magenta.protobuf.generator_pb2.GeneratorDetails(
          id=FLAGS.generator_id)
      if FLAGS.generator_description is not None:
        generator_details.description = FLAGS.generator_description
    else:
      generator_details = None
    encoder_decoder = melody_encoder_decoders[FLAGS.melody_encoder_decoder](
        melody_rnn_model.DEFAULT_MIN_NOTE, melody_rnn_model.DEFAULT_MAX_NOTE)
    hparams = tf.contrib.training.HParams()
    hparams.parse(FLAGS.hparams)
    return melody_rnn_model.MelodyRnnConfig(
        generator_details, encoder_decoder, hparams)
  else:
    if FLAGS.config not in melody_rnn_model.default_configs:
      raise MelodyRnnConfigFlagsException(
          '`--config` must be one of %s. Got %s.' % (
              melody_rnn_model.default_configs.keys(), FLAGS.config))
    config = melody_rnn_model.default_configs[FLAGS.config]
    config.hparams.parse(FLAGS.hparams)
    if FLAGS.generator_id is not None:
      config.details.id = FLAGS.generator_id
    if FLAGS.generator_description is not None:
      config.details.description = FLAGS.generator_description
    return config
