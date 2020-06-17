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

"""MusicVAE generation script."""

# TODO(adarob): Add support for models with conditioning.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
    'output_dir', '/tmp/music_vae/generated',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'config', None,
    'The name of the config to use.')
flags.DEFINE_string(
    'mode', 'sample',
    'Generate mode (either `sample` or `interpolate`).')
flags.DEFINE_string(
    'input_midi_1', None,
    'Path of start MIDI file for interpolation.')
flags.DEFINE_string(
    'input_midi_2', None,
    'Path of end MIDI file for interpolation.')
flags.DEFINE_integer(
    'num_outputs', 5,
    'In `sample` mode, the number of samples to produce. In `interpolate` '
    'mode, the number of steps (including the endpoints).')
flags.DEFINE_integer(
    'max_batch_size', 8,
    'The maximum batch size to use. Decrease if you are seeing an OOM.')
flags.DEFINE_float(
    'temperature', 0.5,
    'The randomness of the decoding process.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(
      np.dot(np.squeeze(p0/np.linalg.norm(p0)),
             np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def run(config_map):
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

  if FLAGS.run_dir is None == FLAGS.checkpoint_file is None:
    raise ValueError(
        'Exactly one of `--run_dir` or `--checkpoint_file` must be specified.')
  if FLAGS.output_dir is None:
    raise ValueError('`--output_dir` is required.')
  tf.gfile.MakeDirs(FLAGS.output_dir)
  if FLAGS.mode != 'sample' and FLAGS.mode != 'interpolate':
    raise ValueError('Invalid value for `--mode`: %s' % FLAGS.mode)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config name: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  config.data_converter.max_tensors_per_item = None

  if FLAGS.mode == 'interpolate':
    if FLAGS.input_midi_1 is None or FLAGS.input_midi_2 is None:
      raise ValueError(
          '`--input_midi_1` and `--input_midi_2` must be specified in '
          '`interpolate` mode.')
    input_midi_1 = os.path.expanduser(FLAGS.input_midi_1)
    input_midi_2 = os.path.expanduser(FLAGS.input_midi_2)
    if not os.path.exists(input_midi_1):
      raise ValueError('Input MIDI 1 not found: %s' % FLAGS.input_midi_1)
    if not os.path.exists(input_midi_2):
      raise ValueError('Input MIDI 2 not found: %s' % FLAGS.input_midi_2)
    input_1 = note_seq.midi_file_to_note_sequence(input_midi_1)
    input_2 = note_seq.midi_file_to_note_sequence(input_midi_2)

    def _check_extract_examples(input_ns, path, input_number):
      """Make sure each input returns exactly one example from the converter."""
      tensors = config.data_converter.to_tensors(input_ns).outputs
      if not tensors:
        print(
            'MusicVAE configs have very specific input requirements. Could not '
            'extract any valid inputs from `%s`. Try another MIDI file.' % path)
        sys.exit()
      elif len(tensors) > 1:
        basename = os.path.join(
            FLAGS.output_dir,
            '%s_input%d-extractions_%s-*-of-%03d.mid' %
            (FLAGS.config, input_number, date_and_time, len(tensors)))
        for i, ns in enumerate(config.data_converter.from_tensors(tensors)):
          note_seq.sequence_proto_to_midi_file(
              ns, basename.replace('*', '%03d' % i))
        print(
            '%d valid inputs extracted from `%s`. Outputting these potential '
            'inputs as `%s`. Call script again with one of these instead.' %
            (len(tensors), path, basename))
        sys.exit()
    logging.info(
        'Attempting to extract examples from input MIDIs using config `%s`...',
        FLAGS.config)
    _check_extract_examples(input_1, FLAGS.input_midi_1, 1)
    _check_extract_examples(input_2, FLAGS.input_midi_2, 2)

  logging.info('Loading model...')
  if FLAGS.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
        os.path.join(FLAGS.run_dir, 'train'))
  else:
    checkpoint_dir_or_path = os.path.expanduser(FLAGS.checkpoint_file)
  model = TrainedModel(
      config, batch_size=min(FLAGS.max_batch_size, FLAGS.num_outputs),
      checkpoint_dir_or_path=checkpoint_dir_or_path)

  if FLAGS.mode == 'interpolate':
    logging.info('Interpolating...')
    _, mu, _ = model.encode([input_1, input_2])
    z = np.array([
        _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, FLAGS.num_outputs)])
    results = model.decode(
        length=config.hparams.max_seq_len,
        z=z,
        temperature=FLAGS.temperature)
  elif FLAGS.mode == 'sample':
    logging.info('Sampling...')
    results = model.sample(
        n=FLAGS.num_outputs,
        length=config.hparams.max_seq_len,
        temperature=FLAGS.temperature)

  basename = os.path.join(
      FLAGS.output_dir,
      '%s_%s_%s-*-of-%03d.mid' %
      (FLAGS.config, FLAGS.mode, date_and_time, FLAGS.num_outputs))
  logging.info('Outputting %d files as `%s`...', FLAGS.num_outputs, basename)
  for i, ns in enumerate(results):
    note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

  logging.info('Done.')


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  run(configs.CONFIG_MAP)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
