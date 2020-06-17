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
"""Tests for converting a directory of MIDIs to a NoteSequence TFRecord file."""

import os
import tempfile

from magenta.scripts import convert_dir_to_note_sequences
from note_seq import music_pb2
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ConvertMidiDirToSequencesTest(tf.test.TestCase):

  def setUp(self):
    midi_filename = os.path.join(tf.resource_loader.get_data_files_path(),
                                 '../testdata/example.mid')

    root_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    sub_1_dir = os.path.join(root_dir, 'sub_1')
    sub_2_dir = os.path.join(root_dir, 'sub_2')
    sub_1_sub_dir = os.path.join(sub_1_dir, 'sub')

    tf.gfile.MkDir(sub_1_dir)
    tf.gfile.MkDir(sub_2_dir)
    tf.gfile.MkDir(sub_1_sub_dir)

    tf.gfile.Copy(midi_filename, os.path.join(root_dir, 'midi_1.mid'))
    tf.gfile.Copy(midi_filename, os.path.join(root_dir, 'midi_2.mid'))
    tf.gfile.Copy(midi_filename, os.path.join(sub_1_dir, 'midi_3.mid'))
    tf.gfile.Copy(midi_filename, os.path.join(sub_2_dir, 'midi_3.mid'))
    tf.gfile.Copy(midi_filename, os.path.join(sub_2_dir, 'midi_4.mid'))
    tf.gfile.Copy(midi_filename, os.path.join(sub_1_sub_dir, 'midi_5.mid'))

    tf.gfile.GFile(
        os.path.join(root_dir, 'non_midi_file'),
        mode='w').write('non-midi data')

    self.expected_sub_dirs = {
        '': {'sub_1', 'sub_2', 'sub_1/sub'},
        'sub_1': {'sub'},
        'sub_1/sub': set(),
        'sub_2': set()
    }
    self.expected_dir_midi_contents = {
        '': {'midi_1.mid', 'midi_2.mid'},
        'sub_1': {'midi_3.mid'},
        'sub_2': {'midi_3.mid', 'midi_4.mid'},
        'sub_1/sub': {'midi_5.mid'}
    }
    self.root_dir = root_dir

  def runTest(self, relative_root, recursive):
    """Tests the output for the given parameters."""
    root_dir = os.path.join(self.root_dir, relative_root)
    expected_filenames = self.expected_dir_midi_contents[relative_root]
    if recursive:
      for sub_dir in self.expected_sub_dirs[relative_root]:
        for filename in self.expected_dir_midi_contents[
            os.path.join(relative_root, sub_dir)]:
          expected_filenames.add(os.path.join(sub_dir, filename))

    with tempfile.NamedTemporaryFile(
        prefix='ConvertMidiDirToSequencesTest') as output_file:
      convert_dir_to_note_sequences.convert_directory(
          root_dir, output_file.name, recursive)
      actual_filenames = set()
      reader = tf.python_io.tf_record_iterator(output_file.name)
      for serialized_sequence in reader:
        sequence = music_pb2.NoteSequence.FromString(serialized_sequence)
        self.assertEqual(
            convert_dir_to_note_sequences.generate_note_sequence_id(
                sequence.filename, os.path.basename(relative_root), 'midi'),
            sequence.id)
        self.assertEqual(os.path.basename(root_dir), sequence.collection_name)
        self.assertNotEqual(0, len(sequence.notes))
        actual_filenames.add(sequence.filename)

    self.assertEqual(expected_filenames, actual_filenames)

  def testConvertMidiDirToSequences_NoRecurse(self):
    self.runTest('', recursive=False)
    self.runTest('sub_1', recursive=False)
    self.runTest('sub_1/sub', recursive=False)
    self.runTest('sub_2', recursive=False)

  def testConvertMidiDirToSequences_Recurse(self):
    self.runTest('', recursive=True)
    self.runTest('sub_1', recursive=True)
    self.runTest('sub_1/sub', recursive=True)
    self.runTest('sub_2', recursive=True)


if __name__ == '__main__':
  tf.test.main()
