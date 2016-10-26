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

MIN_MIDI_PITCH = magenta.music.MIN_MIDI_PITCH
MAX_MIDI_PITCH = magenta.music.MAX_MIDI_PITCH
NOTES_PER_OCTAVE = magenta.music.NOTES_PER_OCTAVE

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = 0

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'config',
    None,
    "Which config to use. Must be one of 'basic', 'lookback', or 'attention'. "
    "Mutually exclusive with `--melody_encoding`.")
tf.app.flags.DEFINE_string(
    'melody_encoding',
    None,
    "Which encoding to use. Must be one of 'onehot', 'lookback', or 'key'."
    "Mutually exclusive with `--config`.")
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
    'hparams', '{}',
    'String representation of a Python dictionary containing hyperparameter '
    'to value mapping. This mapping is merged with the default '
    'hyperparameters if `--config` is also supplied.')


class MelodyRnnConfigException(Exception):
  pass


class MelodyRnnConfig(object):
  """Stores a configuration for a MelodyRnn.

  You can change `min_note` and `max_note` to increase/decrease the melody
  range. Since melodies are transposed into this range to be run through
  the model and then transposed back into their original range after the
  melodies have been extended, the location of the range is somewhat
  arbitrary, but the size of the range determines the possible size of the
  generated melodies range. `transpose_to_key` should be set to the key
  that if melodies were transposed into that key, they would best sit
  between `min_note` and `max_note` with having as few notes outside that
  range.

  Attributes:
    details: The GeneratorDetails message describing the config.
    encoding: The MelodyRnnEncoding object to use.
    hparams: The HParams containing hyperparameters to use.
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.
    transpose_to_key: The key that encoded melodies will be transposed into, or
        None if it should not be transposed.
  """

  def __init__(self, details, encoding, hparams, min_note=DEFAULT_MIN_NOTE,
               max_note=DEFAULT_MAX_NOTE,
               transpose_to_key=DEFAULT_TRANSPOSE_TO_KEY):
    if min_note < MIN_MIDI_PITCH:
      raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > MAX_MIDI_PITCH + 1:
      raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note - min_note < NOTES_PER_OCTAVE:
      raise ValueError('max_note - min_note must be >= 12. min_note is %d. '
                       'max_note is %d. max_note - min_note is %d.' %
                       (min_note, max_note, max_note - min_note))
    if (transpose_to_key is not None and
        (transpose_to_key < 0 or transpose_to_key > NOTES_PER_OCTAVE - 1)):
      raise ValueError('transpose_to_key must be >= 0 and <= 11. '
                       'transpose_to_key is %d.' % transpose_to_key)

    self.details = details
    self.encoding = encoding
    self.hparams = hparams
    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key


# Default configurations.
default_configs = {
    'basic_rnn': MelodyRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='basic_rnn',
            description='Melody RNN with one-hot encoding.'),
        magenta.music.OneHotEventSequenceEncoding(
            magenta.music.MelodyOneHotEncoding(
                min_note=DEFAULT_MIN_NOTE,
                max_note=DEFAULT_MAX_NOTE)),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            clip_norm=5,
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.85)),
    'lookback_rnn': MelodyRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='lookback_rnn',
            description='Melody RNN with lookback encoding.'),
        magenta.music.LookbackEventSequenceEncoding(
            magenta.music.MelodyOneHotEncoding(
                min_note=DEFAULT_MIN_NOTE,
                max_note=DEFAULT_MAX_NOTE)),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            clip_norm=5,
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.95)),
    'attention_rnn': MelodyRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='attention_rnn',
            description='Melody RNN with lookback encoding and attention.'),
        magenta.music.KeyMelodyEncoding(
            min_note=DEFAULT_MIN_NOTE,
            max_note=DEFAULT_MAX_NOTE),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            attn_length=40,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.97))
}


def one_hot_melody_encoding(min_note, max_note):
  """Return a OneHotEventSequenceEncoding for melodies.

  Args:
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.

  Returns:
    A melody OneHotEventSequenceEncoding.
  """
  return magenta.music.OneHotEventSequenceEncoding(
      magenta.music.MelodyOneHotEncoding(min_note, max_note))


def lookback_melody_encoding(min_note, max_note):
  """Return a LookbackEventSequenceEncoding for melodies.

  Args:
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.

  Returns:
    A melody LookbackEventSequenceEncoding.
  """
  return magenta.music.LookbackEventSequenceEncoding(
      magenta.music.MelodyOneHotEncoding(min_note, max_note))


# Dictionary of functions that take `min_note` and `max_note` and return the
# appropriate EventSequenceEncoding object.
melody_encodings = {
    'onehot': one_hot_melody_encoding,
    'lookback': lookback_melody_encoding,
    'key': magenta.music.KeyMelodyEncoding
}


def config_from_flags():
  """Parses flags and returns the appropriate MelodyRnnConfig.

  If `--config` is supplied, returns the matching default MelodyRnnConfig after
  updating the hyperparameters based on `--hparams`.
  If `--melody_encoding` is supplied, returns a new MelodyRnnConfig using the
  matching EventSequenceEncoding, generator details supplied by
  `--generator_id` and `--generator_description`, and hyperparameters based on
  `--hparams`.

  Returns:
     The appropriate MelodyRnnConfig based on the supplied flags.
  Raises:
     MelodyRnnConfigException: When not exactly one of `--config` or
         `melody_encoding` is supplied.
  """
  if (FLAGS.melody_encoding, FLAGS.config).count(None) != 1:
    raise MelodyRnnConfigException(
        'Exactly one of `--config` or `--melody_encoding` must be '
        'supplied.')

  if FLAGS.melody_encoding is not None:
    if FLAGS.melody_encoding not in melody_encodings:
      raise MelodyRnnConfigException(
          '`--melody_encoding` must be one of %s. Got %s.' % (
              melody_encodings.keys(), FLAGS.melody_encoding))
    if FLAGS.generator_id is not None:
      generator_details = magenta.protobuf.generator_pb2.GeneratorDetails(
          id=FLAGS.generator_id)
      if FLAGS.generator_description is not None:
        generator_details.description = FLAGS.generator_description
    else:
      generator_details = None
    encoding = melody_encodings[FLAGS.melody_encoding](
        DEFAULT_MIN_NOTE, DEFAULT_MAX_NOTE)
    hparams = magenta.common.HParams()
    hparams.parse(FLAGS.hparams)
    return MelodyRnnConfig(generator_details, encoding, hparams)
  else:
    if FLAGS.config not in default_configs:
      raise MelodyRnnConfigException(
          '`--config` must be one of %s. Got %s.' % (
              default_configs.keys(), FLAGS.config))
    config = default_configs[FLAGS.config]
    config.hparams.parse(FLAGS.hparams)
    if FLAGS.generator_id is not None:
      config.details.id = FLAGS.generator_id
    if FLAGS.generator_description is not None:
      config.details.description = FLAGS.generator_description
    return config
