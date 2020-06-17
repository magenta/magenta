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
"""Compare a directory of abc and midi files.

Assumes a directory of abc files converted with something like:
# First, remove 'hornpipe' rhythm marker because abc2midi changes note durations
# when that is present.
ls *.abc | xargs -l1 sed -i '/R: hornpipe/d'
ls *.abc | xargs -l1 abc2midi
"""

import os
import pdb
import re

from note_seq import abc_parser
from note_seq import midi_io
from note_seq import sequences_lib
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', None,
                           'Directory containing files to convert.')


class CompareDirectory(tf.test.TestCase):
  """Fake test used to compare directories of abc and midi files."""

  def runTest(self):
    pass

  def compare_directory(self, directory):
    self.maxDiff = None  # pylint: disable=invalid-name

    files_in_dir = tf.gfile.ListDirectory(directory)
    files_parsed = 0
    for file_in_dir in files_in_dir:
      if not file_in_dir.endswith('.abc'):
        continue
      abc = os.path.join(directory, file_in_dir)
      midis = {}
      ref_num = 1
      while True:
        midi = re.sub(r'\.abc$', str(ref_num) + '.mid',
                      os.path.join(directory, file_in_dir))
        if not tf.gfile.Exists(midi):
          break
        midis[ref_num] = midi
        ref_num += 1

      print('parsing {}: {}'.format(files_parsed, abc))
      tunes, exceptions = abc_parser.parse_abc_tunebook_file(abc)
      files_parsed += 1
      self.assertEqual(len(tunes), len(midis) - len(exceptions))

      for tune in tunes.values():
        expanded_tune = sequences_lib.expand_section_groups(tune)
        midi_ns = midi_io.midi_file_to_sequence_proto(
            midis[tune.reference_number])
        # abc2midi adds a 1-tick delay to the start of every note, but we don't.
        tick_length = ((1 / (midi_ns.tempos[0].qpm / 60)) /
                       midi_ns.ticks_per_quarter)
        for note in midi_ns.notes:
          note.start_time -= tick_length
          # For now, don't compare velocities.
          note.velocity = 90
        if len(midi_ns.notes) != len(expanded_tune.notes):
          pdb.set_trace()
          self.assertProtoEquals(midi_ns, expanded_tune)
        for midi_note, test_note in zip(midi_ns.notes, expanded_tune.notes):
          try:
            self.assertProtoEquals(midi_note, test_note)
          except Exception as e:  # pylint: disable=broad-except
            print(e)
            pdb.set_trace()
        self.assertEqual(midi_ns.total_time, expanded_tune.total_time)


def main(unused_argv):
  if not FLAGS.input_dir:
    tf.logging.fatal('--input_dir required')
    return

  input_dir = os.path.expanduser(FLAGS.input_dir)

  CompareDirectory().compare_directory(input_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
