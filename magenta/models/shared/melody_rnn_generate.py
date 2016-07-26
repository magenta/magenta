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
"""Generate melodies from a trained checkpoint of a melody RNN model."""

import ast
import os
import random
import time

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import midi_io
from six.moves import range  # pylint: disable=redefined-builtin
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
tf.app.flags.DEFINE_float('temperature', 1.0,
                          'The randomness of the generated melodies. 1.0 uses '
                          'the unaltered softmax probabilities, greater than '
                          '1.0 makes melodies more random, less than 1.0 makes '
                          'melodies less random.')
tf.app.flags.DEFINE_int('steps_per_beat', 4,
                        'What precision to use when quantizing the melody.')

def get_hparams():
  hparams = ast.literal_eval(FLAGS.hparams if FLAGS.hparams else '{}')
  hparams['temperature'] = FLAGS.temperature
  return hparams

def get_train_dir():
  if not FLAGS.run_dir:
    tf.logging.fatal('--run_dir required')
  return os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')

def get_steps_per_beat():
  return FLAGS.steps_per_beat

def run_with_flags(melody_rnn_sequence_generator):
  """Generates melodies and saves them as MIDI files.

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
  if FLAGS.primer_melody:
    primer_melody = melodies_lib.MonophonicMelody()
    primer_melody.from_event_list(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence()
  elif FLAGS.primer_midi:
    primer_sequence = midi_io.midi_file_to_sequence_proto(FLAGS.primer_midi)

  # Derive start/stop time in seconds from num_steps
  bpm = (primer_sequence.tempos[0].bpm if priming_sequence.tempos
         else melodies_lib.DEFAULT_BEATS_PER_MINUTE)
  generate_seconds = FLAGS.num_steps * get_steps_per_beat() * (bpm / 60.0)
  # Specify start/stop time for generation based on adding generate_seconds to
  # the end of the priming sequence.
  notes_by_end_time = sorted(primer_sequence.notes, key=lambda n: n.end_time)

  generate_request = generator_pb2.GenerateSequenceRequest()
  generate_request.input_sequence.CopyFrom(primer_sequence)
  generate_request.start_time = notes_by_end_time[-1].end_time + (
      bpm / 60.0 / get_steps_per_beat())
  generate_response.end_time = generate_request.start_time + generate_seconds

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



if __name__ == '__main__':
  tf.app.run()
