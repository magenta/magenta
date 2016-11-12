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

# internal imports

import tensorflow as tf
import magenta

from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.melody_rnn import melody_rnn_model
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import events_rnn_generate
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

FLAGS = tf.app.flags.FLAGS
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
tf.app.flags.DEFINE_integer(
    'steps_per_quarter', 4, 'What precision to use when quantizing the melody.')


def _steps_to_seconds(steps, qpm):
  """Converts steps to seconds.

  Uses the current flag value for steps_per_quarter.

  Args:
    steps: number of steps.
    qpm: current qpm.

  Returns:
    Number of seconds the steps represent.
  """
  return steps * 60.0 / qpm / FLAGS.steps_per_quarter


def run_with_flags(generator):
  """Generates melodies and saves them as MIDI files.

  Uses the options specified by the flags defined in this module.

  Args:
    generator: The MelodyRnnSequenceGenerator to use for generation.
  """
  if FLAGS.primer_midi:
    FLAGS.primer_midi = os.path.expanduser(FLAGS.primer_midi)

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
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to a single middle C.')
    primer_melody = magenta.music.Melody([60])
    primer_sequence = primer_melody.to_sequence(qpm=qpm)

  # Derive the total number of seconds to generate based on the QPM of the
  # priming sequence and the num_steps flag.
  total_seconds = _steps_to_seconds(FLAGS.num_steps, qpm)

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  if primer_sequence:
    input_sequence = primer_sequence
    # Set the start time to begin on the next step after the last note ends.
    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    generate_section = generator_options.generate_sections.add(
        start_time=last_end_time + _steps_to_seconds(1, qpm),
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

  events_rnn_generate.run_with_flags(generator, input_sequence,
                                     generator_options)


def main(unused_argv):
  """Saves bundle or runs generator based on flags."""
  events_rnn_generate.setup_logs()

  config = melody_rnn_config_flags.config_from_flags()
  generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
      model=melody_rnn_model.MelodyRnnModel(config),
      details=config.details,
      steps_per_quarter=FLAGS.steps_per_quarter,
      checkpoint=events_rnn_generate.get_checkpoint(),
      bundle=events_rnn_generate.get_bundle())

  if events_rnn_generate.should_save_generator_bundle():
    events_rnn_generate.save_generator_bundle(generator)
  else:
    run_with_flags(generator)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
