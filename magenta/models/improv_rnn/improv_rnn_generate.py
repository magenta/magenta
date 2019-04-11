# Copyright 2019 The Magenta Authors.
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

"""Generate melodies from a trained checkpoint of an improv RNN model."""

import ast
import os
import time

import magenta
from magenta.models.improv_rnn import improv_rnn_config_flags
from magenta.models.improv_rnn import improv_rnn_model
from magenta.models.improv_rnn import improv_rnn_sequence_generator
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2
import tensorflow as tf

CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL

# Velocity at which to play chord notes when rendering chords.
CHORD_VELOCITY = 50

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
    'output_dir', '/tmp/improv_rnn/generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 10,
    'The number of lead sheets to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_integer(
    'steps_per_chord', 16,
    'The number of melody steps to take per backing chord. Each step is a 16th '
    'of a bar, so if backing_chords = "C G Am F" and steps_per_chord = 16, '
    'four bars will be generated.')
tf.app.flags.DEFINE_string(
    'primer_melody', '',
    'A string representation of a Python list of '
    'magenta.music.Melody event values. For example: '
    '"[60, -2, 60, -2, 67, -2, 67, -2]". If specified, this melody will be '
    'used as the priming melody. If a priming melody is not specified, '
    'melodies will be generated from scratch.')
tf.app.flags.DEFINE_string(
    'backing_chords', 'C G Am F C G F C',
    'A string representation of a chord progression, with chord symbols '
    'separated by spaces. For example: "C Dm7 G13 Cmaj7". The duration of each '
    'chord, in steps, is specified by the steps_per_chord flag.')
tf.app.flags.DEFINE_string(
    'primer_midi', '',
    'The path to a MIDI file containing a melody that will be used as a '
    'priming melody. If a primer melody is not specified, melodies will be '
    'generated from scratch.')
tf.app.flags.DEFINE_boolean(
    'render_chords', False,
    'If true, the backing chords will also be rendered as notes in the output '
    'MIDI files.')
tf.app.flags.DEFINE_float(
    'qpm', None,
    'The quarters per minute to play generated output at. If a primer MIDI is '
    'given, the qpm from that will override this flag. If qpm is None, qpm '
    'will default to 120.')
tf.app.flags.DEFINE_float(
    'temperature', 1.0,
    'The randomness of the generated melodies. 1.0 uses the unaltered softmax '
    'probabilities, greater than 1.0 makes melodies more random, less than 1.0 '
    'makes melodies less random.')
tf.app.flags.DEFINE_integer(
    'beam_size', 1,
    'The beam size to use for beam search when generating melodies.')
tf.app.flags.DEFINE_integer(
    'branch_factor', 1,
    'The branch factor to use for beam search when generating melodies.')
tf.app.flags.DEFINE_integer(
    'steps_per_iteration', 1,
    'The number of melody steps to take per beam search iteration.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')


def get_checkpoint():
  """Get the training dir to be used by the model."""
  if FLAGS.run_dir and FLAGS.bundle_file and not FLAGS.save_generator_bundle:
    raise magenta.music.SequenceGeneratorError(
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
  return magenta.music.read_bundle_file(bundle_file)


def run_with_flags(generator):
  """Generates melodies and saves them as MIDI files.

  Uses the options specified by the flags defined in this module.

  Args:
    generator: The ImprovRnnSequenceGenerator to use for generation.
  """
  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return
  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)

  primer_midi = None
  if FLAGS.primer_midi:
    primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  primer_sequence = None
  qpm = FLAGS.qpm if FLAGS.qpm else magenta.music.DEFAULT_QUARTERS_PER_MINUTE
  if FLAGS.primer_melody:
    primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence(qpm=qpm)
  elif primer_midi:
    primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
    if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
      qpm = primer_sequence.tempos[0].qpm
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to a single middle C.')
    primer_melody = magenta.music.Melody([60])
    primer_sequence = primer_melody.to_sequence(qpm=qpm)

  # Create backing chord progression from flags.
  raw_chords = FLAGS.backing_chords.split()
  repeated_chords = [chord for chord in raw_chords
                     for _ in range(FLAGS.steps_per_chord)]
  backing_chords = magenta.music.ChordProgression(repeated_chords)

  # Derive the total number of seconds to generate based on the QPM of the
  # priming sequence and the length of the backing chord progression.
  seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
  total_seconds = len(backing_chords) * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  if primer_sequence:
    input_sequence = primer_sequence
    # Set the start time to begin on the next step after the last note ends.
    if primer_sequence.notes:
      last_end_time = max(n.end_time for n in primer_sequence.notes)
    else:
      last_end_time = 0
    generate_section = generator_options.generate_sections.add(
        start_time=last_end_time + seconds_per_step,
        end_time=total_seconds)

    if generate_section.start_time >= generate_section.end_time:
      tf.logging.fatal(
          'Priming sequence is longer than the total number of steps '
          'requested: Priming sequence length: %s, Generation length '
          'requested: %s',
          generate_section.start_time, total_seconds)
      return
  else:
    input_sequence = music_pb2.NoteSequence()
    input_sequence.tempos.add().qpm = qpm
    generate_section = generator_options.generate_sections.add(
        start_time=0,
        end_time=total_seconds)

  # Add the backing chords to the input sequence.
  chord_sequence = backing_chords.to_sequence(sequence_start_time=0.0, qpm=qpm)
  for text_annotation in chord_sequence.text_annotations:
    if text_annotation.annotation_type == CHORD_SYMBOL:
      chord = input_sequence.text_annotations.add()
      chord.CopyFrom(text_annotation)
  input_sequence.total_time = len(backing_chords) * seconds_per_step

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

    if FLAGS.render_chords:
      renderer = magenta.music.BasicChordRenderer(velocity=CHORD_VELOCITY)
      renderer.render(generated_sequence)

    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(FLAGS.output_dir, midi_filename)
    magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

  tf.logging.info('Wrote %d MIDI files to %s',
                  FLAGS.num_outputs, FLAGS.output_dir)


def main(unused_argv):
  """Saves bundle or runs generator based on flags."""
  tf.logging.set_verbosity(FLAGS.log)

  bundle = get_bundle()

  if bundle:
    config_id = bundle.generator_details.id
    config = improv_rnn_model.default_configs[config_id]
    config.hparams.parse(FLAGS.hparams)
  else:
    config = improv_rnn_config_flags.config_from_flags()
  # Having too large of a batch size will slow generation down unnecessarily.
  config.hparams.batch_size = min(
      config.hparams.batch_size, FLAGS.beam_size * FLAGS.branch_factor)

  generator = improv_rnn_sequence_generator.ImprovRnnSequenceGenerator(
      model=improv_rnn_model.ImprovRnnModel(config),
      details=config.details,
      steps_per_quarter=config.steps_per_quarter,
      checkpoint=get_checkpoint(),
      bundle=bundle)

  if FLAGS.save_generator_bundle:
    bundle_filename = os.path.expanduser(FLAGS.bundle_file)
    if FLAGS.bundle_description is None:
      tf.logging.warning('No bundle description provided.')
    tf.logging.info('Saving generator bundle to %s', bundle_filename)
    generator.create_bundle_file(bundle_filename, FLAGS.bundle_description)
  else:
    run_with_flags(generator)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
