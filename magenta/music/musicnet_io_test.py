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
"""Tests for MusicNet data parsing."""

import os

import numpy as np
import tensorflow as tf

from magenta.music import musicnet_io


class MusicNetIoTest(tf.test.TestCase):

  def setUp(self):
    # This example archive contains a single file consisting of just a major
    # chord.
    self.musicnet_example_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/musicnet_example.npz')

  def testNoteIntervalTreeToSequenceProto(self):
    example = np.load(self.musicnet_example_filename, encoding='latin1')
    note_interval_tree = example['test'][1]
    sequence = musicnet_io.note_interval_tree_to_sequence_proto(
        note_interval_tree, 44100)
    self.assertEqual(3, len(sequence.notes))
    self.assertEqual(72, min(note.pitch for note in sequence.notes))
    self.assertEqual(79, max(note.pitch for note in sequence.notes))
    self.assertTrue(all(note.instrument == 0 for note in sequence.notes))
    self.assertTrue(all(note.program == 41 for note in sequence.notes))
    self.assertEqual(0.5, sequence.total_time)

  def testMusicNetIterator(self):
    iterator = musicnet_io.musicnet_iterator(self.musicnet_example_filename)
    pairs = list(iterator)
    audio, sequence = pairs[0]
    self.assertEqual(1, len(pairs))
    self.assertEqual('test', sequence.filename)
    self.assertEqual('MusicNet', sequence.collection_name)
    self.assertEqual('/id/musicnet/test', sequence.id)
    self.assertEqual(3, len(sequence.notes))
    self.assertEqual(66150, len(audio))


if __name__ == '__main__':
  tf.test.main()
