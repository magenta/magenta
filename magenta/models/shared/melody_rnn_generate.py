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
"""Generate melodies from a trained checkpoint of a melody RNN model.

Uses flags to define operation.
"""

import ast
import os
import time

# internal imports
from six.moves import range  # pylint: disable=redefined-builtin

import tensorflow as tf
import magenta

from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2


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
    'hparams', '{}',
    'String representation of a Python dictionary containing hyperparameter '
    'to value mapping. This mapping is merged with the default '
    'hyperparameters.')
tf.app.flags.DEFINE_string(
    'output_dir', '/tmp/melody_rnn/generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 10,
    'The number of melodies to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_integer(
    'num_steps', 128,
    'The total number of steps the generated melodies should be, priming '
    'melody length + generated steps. Each step is a 16th of a bar.')
tf.app.flags.DEFINE_string(
    'primer_melody', '',
    'A string representation of a Python list of '
    'magenta.music.Melody event values. For example: '
    '"[60, -2, 60, -2, 67, -2, 67, -2]". If specified, this melody will be '
    'used as the priming melody. If a priming melody is not specified, '
    'melodies will be generated from scratch.')
tf.app.flags.DEFINE_string(
    'primer_midi', '',
    'The path to a MIDI file containing a melody that will be used as a '
    'priming melody. If a primer melody is not specified, melodies will be '
    'generated from scratch.')
tf.app.flags.DEFINE_float(
    'qpm', None,
    'The quarters per minute to play generated output at. If a primer MIDI is '
    'given, the qpm from that will override this flag. If qpm is None, qpm '
    'will default to 120.')
tf.app.flags.DEFINE_float(
    'temperature', 1.0,
    'The randomness of the generated melodies. 1.0 uses the unaltered softmax '
    'probabilities, greater than 1.0 makes melodies more random, less than '
    '1.0 makes melodies less random.')
tf.app.flags.DEFINE_integer(
    'steps_per_quarter', 4, 'What precision to use when quantizing the melody.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')


def get_hparams():
  """Get the hparams dictionary to be used by the model."""
  hparams = ast.literal_eval(FLAGS.hparams if FLAGS.hparams else '{}')
  hparams['temperature'] = FLAGS.temperature
  return hparams


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


def get_bundle_file():
  """Get the path to the bundle file with both a checkpoint and metagraph."""
  if FLAGS.bundle_file is None:
    return None
  else:
    return os.path.expanduser(FLAGS.bundle_file)


def get_bundle():
  """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.

  Returns:
    Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
    not set or the save_generator_bundle flag is set.
  """
  if should_save_generator_bundle():
    return None
  bundle_file = get_bundle_file()
  if bundle_file is None:
    return None
  return magenta.music.read_bundle_file(bundle_file)


def should_save_generator_bundle():
  """Returns whether the generator should save a bundle.

  If true, the generator should save its checkpoint and metagraph into a bundle
  file, specified by get_bundle_file, instead of generating a sequence.

  Returns:
    Whether the generator should save a bundle.
  """
  return FLAGS.save_generator_bundle


def get_steps_per_quarter():
  """Get the number of steps per quarter note."""
  return FLAGS.steps_per_quarter


def _steps_to_seconds(steps, qpm):
  """Converts steps to seconds.

  Uses the current flag value for steps_per_quarter.

  Args:
    steps: number of steps.
    qpm: current qpm.

  Returns:
    Number of seconds the steps represent.
  """
  return steps * 60.0 / qpm / get_steps_per_quarter()


def setup_logs():
  """Sets log level to the one specified in the flags."""
  tf.logging.set_verbosity(FLAGS.log)


def run_with_flags(melody_rnn_sequence_generator):
  """Generates melodies and saves them as MIDI files.

  Uses the options specified by the flags defined in this module. Intended to be
  called from the main function of one of the melody generator modules.

  Args:
    melody_rnn_sequence_generator: A MelodyRnnSequenceGenerator object specific
        to your model.
  """
  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return

  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)
  if FLAGS.primer_midi:
    FLAGS.primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  primer_sequence = None
  qpm = FLAGS.qpm if FLAGS.qpm else magenta.music.DEFAULT_QUARTERS_PER_MINUTE
  if FLAGS.primer_melody:
    primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence(qpm=qpm)
  elif FLAGS.primer_midi:
    primer_sequence = magenta.music.midi_file_to_sequence_proto(
        FLAGS.primer_midi)
    if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
      qpm = primer_sequence.tempos[0].qpm

  # Derive the total number of seconds to generate based on the QPM of the
  # priming sequence and the num_steps flag.
  total_seconds = _steps_to_seconds(FLAGS.num_steps, qpm)

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  if primer_sequence:
    input_sequence = primer_sequence
    generate_section = generator_options.generate_sections.add()
    # Set the start time to begin on the next step after the last note ends.
    notes_by_end_time = sorted(primer_sequence.notes, key=lambda n: n.end_time)
    last_end_time = notes_by_end_time[-1].end_time if notes_by_end_time else 0
    generate_section.start_time_seconds = last_end_time + _steps_to_seconds(
        1, qpm)
    generate_section.end_time_seconds = total_seconds

    if generate_section.start_time_seconds >= generate_section.end_time_seconds:
      tf.logging.fatal(
          'Priming sequence is longer than the total number of steps '
          'requested: Priming sequence length: %s, Generation length '
          'requested: %s',
          generate_section.start_time_seconds, total_seconds)
      return
  else:
    input_sequence = music_pb2.NoteSequence()
    input_sequence.tempos.add().qpm = qpm
    generate_section = generator_options.generate_sections.add()
    generate_section.start_time_seconds = 0
    generate_section.end_time_seconds = total_seconds
  tf.logging.debug('input_sequence: %s', input_sequence)
  tf.logging.debug('generator_options: %s', generator_options)

  # Make the generate request num_outputs times and save the output as midi
  # files.
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  digits = len(str(FLAGS.num_outputs))
  for i in range(FLAGS.num_outputs):
    generated_sequence = melody_rnn_sequence_generator.generate(
        input_sequence, generator_options)

    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(FLAGS.output_dir, midi_filename)
    magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

  tf.logging.info('Wrote %d MIDI files to %s',
                  FLAGS.num_outputs, FLAGS.output_dir)
