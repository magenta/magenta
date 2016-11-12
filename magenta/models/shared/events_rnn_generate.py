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
"""Generate event sequences from a trained checkpoint of an RNN model."""

import os
import time

# internal imports
from six.moves import range  # pylint: disable=redefined-builtin

import tensorflow as tf
import magenta

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
tf.app.flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
tf.app.flags.DEFINE_string(
    'bundle_file', None,
    'Path to the bundle file. If specified, this will take priority over '
    'run_dir and checkpoint_file, unless save_generator_bundle is True, in '
    'which case both this flag and either run_dir or checkpoint_file are '
    'required')
tf.app.flags.DEFINE_boolean(
    'save_generator_bundle', False,
    'If true, instead of generating a sequence, will save this generator as a '
    'bundle file in the location specified by the bundle_file flag')
tf.app.flags.DEFINE_string(
    'output_dir', '/tmp/events_rnn/generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 10,
    'The number of sequences to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_float(
    'temperature', 1.0,
    'The randomness of the generated sequences. 1.0 uses the unaltered softmax '
    'probabilities, greater than 1.0 makes events more random, less than 1.0 '
    'makes events less random.')
tf.app.flags.DEFINE_integer(
    'beam_size', 1,
    'The beam size to use for beam search when generating event sequences.')
tf.app.flags.DEFINE_integer(
    'branch_factor', 1,
    'The branch factor to use for beam search when generating event sequences.')
tf.app.flags.DEFINE_integer(
    'steps_per_iteration', 1,
    'The number of steps to take per beam search iteration.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')


def setup_logs():
  """Sets log level to the one specified in the flags."""
  tf.logging.set_verbosity(FLAGS.log)


def get_checkpoint():
  """Get the training dir or checkpoint path to be used by the model."""
  if ((FLAGS.run_dir or FLAGS.checkpoint_file) and
      FLAGS.bundle_file and not should_save_generator_bundle()):
    raise magenta.music.SequenceGeneratorException(
        'Cannot specify both bundle_file and run_dir or checkpoint_file')
  if FLAGS.run_dir:
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')
    return train_dir
  elif FLAGS.checkpoint_file:
    return os.path.expanduser(FLAGS.checkpoint_file)
  else:
    return None


def get_bundle():
  """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.

  Returns:
    Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
    not set or the save_generator_bundle flag is set.
  """
  if should_save_generator_bundle():
    return None
  if FLAGS.bundle_file is None:
    return None
  bundle_file = os.path.expanduser(FLAGS.bundle_file)
  return magenta.music.read_bundle_file(bundle_file)


def should_save_generator_bundle():
  """Returns whether the generator should save a bundle.

  If true, the generator should save its checkpoint and metagraph into a bundle
  file, specified by get_bundle_file, instead of generating a sequence.

  Returns:
    Whether the generator should save a bundle.
  """
  return FLAGS.save_generator_bundle


def run_with_flags(generator, input_sequence, generator_options):
  """Generates event sequences and saves them as MIDI files.

  Uses the options specified by the flags defined in this module.

  Args:
    generator: The SequenceGenerator to use for generation.
    input_sequence: The input NoteSequence to pass to the generator.
    generator_options: A GeneratorOptions proto with options to use for
        generation. This proto should have the sections to generate already
        specified, and will be augmented with options relating to beam search.
  """
  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return
  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  generator_options.args['temperature'].float_value = FLAGS.temperature
  generator_options.args['beam_size'].int_value = FLAGS.beam_size
  generator_options.args['branch_factor'].int_value = FLAGS.branch_factor
  generator_options.args[
      'steps_per_iteration'].int_value = FLAGS.steps_per_iteration
  tf.logging.debug('input_sequence: %s', input_sequence)
  tf.logging.debug('generator_options: %s', generator_options)

  # Make the generate request num_outputs times and save the output as midi
  # files.
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  digits = len(str(FLAGS.num_outputs))
  for i in range(FLAGS.num_outputs):
    generated_sequence = generator.generate(input_sequence, generator_options)

    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(FLAGS.output_dir, midi_filename)
    magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

  tf.logging.info('Wrote %d MIDI files to %s',
                  FLAGS.num_outputs, FLAGS.output_dir)


def save_generator_bundle(generator):
  bundle_filename = os.path.expanduser(FLAGS.bundle_file)
  tf.logging.info('Saving generator bundle to %s', bundle_filename)
  generator.create_bundle_file(bundle_filename)
