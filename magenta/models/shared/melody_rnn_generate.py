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

from magenta.lib import melodies_lib
from magenta.lib import midi_io
from magenta.protobuf import generator_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run_dir', '/tmp/melody_rnn/logdir/run1',
                           'Path to the directory where the latest checkpoint '
                           'will be loaded from.')
tf.app.flags.DEFINE_string('hparams', '{}',
                           'String representation of a Python dictionary '
                           'containing hyperparameter to value mapping. This '
                           'mapping is merged with the default '
                           'hyperparameters.')
tf.app.flags.DEFINE_string('output_dir', '/tmp/melody_rnn/generated',
                           'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer('num_outputs', 10,
                            'The number of melodies to generate. One MIDI '
                            'file will be created for each.')
tf.app.flags.DEFINE_integer('num_steps', 128,
                            'The total number of steps the generated melodies '
                            'should be, priming melody length + generated '
                            'steps. Each step is a 16th of a bar.')
tf.app.flags.DEFINE_string('primer_melody', '',
                           'A string representation of a Python list of '
                           'melodies_lib.MonophonicMelody event values. For '
                           'example: "[60, -2, 60, -2, 67, -2, 67, -2]". If '
                           'specified, this melody will be used as the priming '
                           'melody. If a priming melody is not specified, '
                           'melodies will be generated from scratch.')
tf.app.flags.DEFINE_string('primer_midi', '',
                           'The path to a MIDI file containing a melody that '
                           'will be used as a priming melody. If a primer '
                           'melody is not specified, melodies will be '
                           'generated from scratch.')
tf.app.flags.DEFINE_float('bpm', None,
                          'The beats per minute to play generated output at. '
                          'If a primer MIDI is given, the bpm from that will '
                          'override this flag. If bpm is None, bpm will '
                          'default to 120.')
tf.app.flags.DEFINE_float('temperature', 1.0,
                          'The randomness of the generated melodies. 1.0 uses '
                          'the unaltered softmax probabilities, greater than '
                          '1.0 makes melodies more random, less than 1.0 makes '
                          'melodies less random.')
tf.app.flags.DEFINE_integer('steps_per_beat', 4,
                            'What precision to use when quantizing the melody.')


def get_hparams():
  """Get the hparams dictionary to be used by the model."""
  hparams = ast.literal_eval(FLAGS.hparams if FLAGS.hparams else '{}')
  hparams['temperature'] = FLAGS.temperature
  return hparams


def get_train_dir():
  """Get the training dir to be used by the model."""
  if not FLAGS.run_dir:
    tf.logging.fatal('--run_dir required')
  return os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')


def get_steps_per_beat():
  """Get the number of steps per beat."""
  return FLAGS.steps_per_beat


def _steps_to_seconds(steps, bpm):
  """Converts steps to seconds.

  Uses the current flag value for steps_per_beat.

  Args:
    steps: number of steps.
    bpm: current bpm.

  Returns:
    Number of seconds the steps represent.
  """
  return steps * 60.0 / bpm / get_steps_per_beat()


def run_with_flags(melody_rnn_sequence_generator):
  """Generates melodies and saves them as MIDI files.

  Uses the options specified by the flags defined in this module. Intended to be
  called from the main function of one of the melody generator modules.

  Args:
    melody_rnn_sequence_generator: A MelodyRnnSequenceGenerator object specific
        to your model.
  """
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return

  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)
  if FLAGS.primer_midi:
    FLAGS.primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  primer_sequence = None
  bpm = FLAGS.bpm if FLAGS.bpm else melodies_lib.DEFAULT_BEATS_PER_MINUTE
  if FLAGS.primer_melody:
    primer_melody = melodies_lib.MonophonicMelody()
    primer_melody.from_event_list(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence(bpm=bpm)
  elif FLAGS.primer_midi:
    primer_sequence = midi_io.midi_file_to_sequence_proto(FLAGS.primer_midi)
    if primer_sequence.tempos and primer_sequence.tempos[0].bpm:
      bpm = primer_sequence.tempos[0].bpm

  # Derive the total number of seconds to generate based on the BPM of the
  # priming sequence and the num_steps flag.
  total_seconds = _steps_to_seconds(FLAGS.num_steps, bpm)

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generate_request = generator_pb2.GenerateSequenceRequest()
  if primer_sequence:
    generate_request.input_sequence.CopyFrom(primer_sequence)
    generate_section = (
        generate_request.generator_options.generate_sections.add())
    # Set the start time to begin on the next step after the last note ends.
    notes_by_end_time = sorted(primer_sequence.notes, key=lambda n: n.end_time)
    last_end_time = notes_by_end_time[-1].end_time if notes_by_end_time else 0
    generate_section.start_time_seconds = last_end_time + _steps_to_seconds(
        1, bpm)
    generate_section.end_time_seconds = total_seconds

    if generate_section.start_time_seconds >= generate_section.end_time_seconds:
      tf.logging.fatal(
          'Priming sequence is longer than the total number of steps '
          'requested: Priming sequence length: %s, Generation length '
          'requested: %s',
          generate_section.start_time_seconds, total_seconds)
      return
  else:
    generate_section = (
        generate_request.generator_options.generate_sections.add())
    generate_section.start_time_seconds = 0
    generate_section.end_time_seconds = total_seconds
    generate_request.input_sequence.tempos.add().bpm = bpm
  tf.logging.info('generate_request: %s', generate_request)

  # Make the generate request num_outputs times and save the output as midi
  # files.
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  digits = len(str(FLAGS.num_outputs))
  for i in range(FLAGS.num_outputs):
    generate_response = melody_rnn_sequence_generator.generate(
        generate_request)

    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(FLAGS.output_dir, midi_filename)
    midi_io.sequence_proto_to_midi_file(
        generate_response.generated_sequence, midi_path)

  tf.logging.info('Wrote %d MIDI files to %s',
                  FLAGS.num_outputs, FLAGS.output_dir)
