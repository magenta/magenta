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

"""Command line utility for fine alignment of wav/midi pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import align_fine_lib
from magenta.music import audio_io
from magenta.music import midi_io

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'Directory for input files')
flags.DEFINE_string('output_dir', None, 'Directory for output files')
flags.DEFINE_string('sf2_path', '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                    'SF2 file for synthesis')
flags.DEFINE_float('penalty_mul', 1.0,
                   'Penalty multiplier for non-diagnoal moves')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  for input_file in sorted(os.listdir(FLAGS.input_dir)):
    if not input_file.endswith('.wav'):
      continue
    wav_filename = input_file
    midi_filename = input_file.replace('.wav', '.mid')
    logging.info('Aligning %s to %s', midi_filename, wav_filename)

    samples = audio_io.load_audio(
        os.path.join(FLAGS.input_dir, wav_filename), align_fine_lib.SAMPLE_RATE)
    ns = midi_io.midi_file_to_sequence_proto(
        os.path.join(FLAGS.input_dir, midi_filename))

    aligned_ns, unused_stats = align_fine_lib.align_cpp(
        samples,
        align_fine_lib.SAMPLE_RATE,
        ns,
        align_fine_lib.CQT_HOP_LENGTH_FINE,
        sf2_path=FLAGS.sf2_path,
        penalty_mul=FLAGS.penalty_mul)

    midi_io.sequence_proto_to_midi_file(
        aligned_ns, os.path.join(FLAGS.output_dir, midi_filename))

  logging.info('Done')


if __name__ == '__main__':
  flags.mark_flags_as_required(['input_dir', 'output_dir'])
  app.run(main)
