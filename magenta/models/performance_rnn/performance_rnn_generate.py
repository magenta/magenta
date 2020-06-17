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
"""Generate polyphonic performances from a trained checkpoint.

Uses flags to define operation.
"""
import ast
import os
import time

from magenta.models.performance_rnn import performance_model
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.models.shared import sequence_generator
from magenta.models.shared import sequence_generator_bundle
import note_seq
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
tf.app.flags.DEFINE_string(
    'bundle_file', None,
    'Path to the bundle file. If specified, this will take priority over '
    'run_dir, unless save_generator_bundle is True, in which case both this '
    'flag and run_dir are required')
tf.app.flags.DEFINE_boolean(
    'save_generator_bundle', False,
    'If true, instead of generating a sequence, will save this generator as a '
    'bundle file in the location specified by the bundle_file flag')
tf.app.flags.DEFINE_string(
    'bundle_description', None,
    'A short, human-readable text description of the bundle (e.g., training '
    'data, hyper parameters, etc.).')
tf.app.flags.DEFINE_string(
    'config', 'performance', 'Config to use.')
tf.app.flags.DEFINE_string(
    'output_dir', '/tmp/performance_rnn/generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 10,
    'The number of tracks to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_integer(
    'num_steps', 3000,
    'The total number of steps the generated track should be, priming '
    'track length + generated steps. Each step is 10 milliseconds.')
tf.app.flags.DEFINE_string(
    'primer_pitches', '',
    'A string representation of a Python list of pitches that will be used as '
    'a starting chord with a quarter note duration. For example: '
    '"[60, 64, 67]"')
tf.app.flags.DEFINE_string(
    'primer_melody', '', 'A string representation of a Python list of '
    'note_seq.Melody event values. For example: '
    '"[60, -2, 60, -2, 67, -2, 67, -2]". The primer melody will be played at '
    'a fixed tempo of 120 QPM with 4 steps per quarter note.')
tf.app.flags.DEFINE_string(
    'primer_midi', '',
    'The path to a MIDI file containing a polyphonic track that will be used '
    'as a priming track.')
tf.app.flags.DEFINE_string(
    'disable_conditioning', None,
    'When optional conditioning is available, a string representation of a '
    'Boolean indicating whether or not to disable conditioning. Similar to '
    'control signals, this can also be a list of Booleans; when it is a list, '
    'the other conditioning variables will be ignored for segments where '
    'conditioning is disabled.')
tf.app.flags.DEFINE_float(
    'temperature', 1.0,
    'The randomness of the generated tracks. 1.0 uses the unaltered '
    'softmax probabilities, greater than 1.0 makes tracks more random, less '
    'than 1.0 makes tracks less random.')
tf.app.flags.DEFINE_integer(
    'beam_size', 1,
    'The beam size to use for beam search when generating tracks.')
tf.app.flags.DEFINE_integer(
    'branch_factor', 1,
    'The branch factor to use for beam search when generating tracks.')
tf.app.flags.DEFINE_integer(
    'steps_per_iteration', 1,
    'The number of steps to take per beam search iteration.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Comma-separated list of `name=value` pairs. For each pair, the value of '
    'the hyperparameter named `name` is set to `value`. This mapping is merged '
    'with the default hyperparameters.')

# Add flags for all performance control signals.
for control_signal_cls in note_seq.all_performance_control_signals:
  tf.app.flags.DEFINE_string(
      control_signal_cls.name, None, control_signal_cls.description)


def get_checkpoint():
  """Get the training dir or checkpoint path to be used by the model."""
  if FLAGS.run_dir and FLAGS.bundle_file and not FLAGS.save_generator_bundle:
    raise sequence_generator.SequenceGeneratorError(
        'Cannot specify both bundle_file and run_dir')
  if FLAGS.run_dir:
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')
    return train_dir
  else:
    return None


def get_bundle():
  """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.

  Returns:
    Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
    not set or the save_generator_bundle flag is set.
  """
  if FLAGS.save_generator_bundle:
    return None
  if FLAGS.bundle_file is None:
    return None
  bundle_file = os.path.expanduser(FLAGS.bundle_file)
  return sequence_generator_bundle.read_bundle_file(bundle_file)


def run_with_flags(generator):
  """Generates performance tracks and saves them as MIDI files.

  Uses the options specified by the flags defined in this module.

  Args:
    generator: The PerformanceRnnSequenceGenerator to use for generation.
  """
  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return
  output_dir = os.path.expanduser(FLAGS.output_dir)

  primer_midi = None
  if FLAGS.primer_midi:
    primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  primer_sequence = None
  if FLAGS.primer_pitches:
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.ticks_per_quarter = note_seq.STANDARD_PPQ
    for pitch in ast.literal_eval(FLAGS.primer_pitches):
      note = primer_sequence.notes.add()
      note.start_time = 0
      note.end_time = 60.0 / note_seq.DEFAULT_QUARTERS_PER_MINUTE
      note.pitch = pitch
      note.velocity = 100
      primer_sequence.total_time = note.end_time
  elif FLAGS.primer_melody:
    primer_melody = note_seq.Melody(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence()
  elif primer_midi:
    primer_sequence = note_seq.midi_file_to_sequence_proto(primer_midi)
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to empty sequence.')
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.ticks_per_quarter = note_seq.STANDARD_PPQ

  # Derive the total number of seconds to generate.
  seconds_per_step = 1.0 / generator.steps_per_second
  generate_end_time = FLAGS.num_steps * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  # Set the start time to begin when the last note ends.
  generate_section = generator_options.generate_sections.add(
      start_time=primer_sequence.total_time,
      end_time=generate_end_time)

  if generate_section.start_time >= generate_section.end_time:
    tf.logging.fatal(
        'Priming sequence is longer than the total number of steps '
        'requested: Priming sequence length: %s, Total length '
        'requested: %s',
        generate_section.start_time, generate_end_time)
    return

  for control_cls in note_seq.all_performance_control_signals:
    if FLAGS[control_cls.name].value is not None and (
        generator.control_signals is None or not any(
            control.name == control_cls.name
            for control in generator.control_signals)):
      tf.logging.warning(
          'Control signal requested via flag, but generator is not set up to '
          'condition on this control signal. Request will be ignored: %s = %s',
          control_cls.name, FLAGS[control_cls.name].value)

  if (FLAGS.disable_conditioning is not None and
      not generator.optional_conditioning):
    tf.logging.warning(
        'Disable conditioning flag set, but generator is not set up for '
        'optional conditioning. Requested disable conditioning flag will be '
        'ignored: %s', FLAGS.disable_conditioning)

  if generator.control_signals:
    for control in generator.control_signals:
      if FLAGS[control.name].value is not None:
        generator_options.args[control.name].string_value = (
            FLAGS[control.name].value)
  if FLAGS.disable_conditioning is not None:
    generator_options.args['disable_conditioning'].string_value = (
        FLAGS.disable_conditioning)

  generator_options.args['temperature'].float_value = FLAGS.temperature
  generator_options.args['beam_size'].int_value = FLAGS.beam_size
  generator_options.args['branch_factor'].int_value = FLAGS.branch_factor
  generator_options.args[
      'steps_per_iteration'].int_value = FLAGS.steps_per_iteration

  tf.logging.debug('primer_sequence: %s', primer_sequence)
  tf.logging.debug('generator_options: %s', generator_options)

  # Make the generate request num_outputs times and save the output as midi
  # files.
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  digits = len(str(FLAGS.num_outputs))
  for i in range(FLAGS.num_outputs):
    generated_sequence = generator.generate(primer_sequence, generator_options)

    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(output_dir, midi_filename)
    note_seq.sequence_proto_to_midi_file(generated_sequence, midi_path)

  tf.logging.info('Wrote %d MIDI files to %s',
                  FLAGS.num_outputs, output_dir)


def main(unused_argv):
  """Saves bundle or runs generator based on flags."""
  tf.logging.set_verbosity(FLAGS.log)

  bundle = get_bundle()

  config_id = bundle.generator_details.id if bundle else FLAGS.config
  config = performance_model.default_configs[config_id]
  config.hparams.parse(FLAGS.hparams)
  # Having too large of a batch size will slow generation down unnecessarily.
  config.hparams.batch_size = min(
      config.hparams.batch_size, FLAGS.beam_size * FLAGS.branch_factor)

  generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
      model=performance_model.PerformanceRnnModel(config),
      details=config.details,
      steps_per_second=config.steps_per_second,
      num_velocity_bins=config.num_velocity_bins,
      control_signals=config.control_signals,
      optional_conditioning=config.optional_conditioning,
      checkpoint=get_checkpoint(),
      bundle=bundle,
      note_performance=config.note_performance)

  if FLAGS.save_generator_bundle:
    bundle_filename = os.path.expanduser(FLAGS.bundle_file)
    if FLAGS.bundle_description is None:
      tf.logging.warning('No bundle description provided.')
    tf.logging.info('Saving generator bundle to %s', bundle_filename)
    generator.create_bundle_file(bundle_filename, FLAGS.bundle_description)
  else:
    run_with_flags(generator)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
