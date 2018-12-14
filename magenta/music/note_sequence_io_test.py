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
"""Tests to ensure correct reading and writing of NoteSequence record files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

from magenta.music import note_sequence_io
from magenta.protobuf import music_pb2


class NoteSequenceIoTest(tf.test.TestCase):

  def testGenerateId(self):
    sequence_id_1 = note_sequence_io.generate_note_sequence_id(
        '/my/file/name', 'my_collection', 'midi')
    self.assertEqual('/id/midi/my_collection/', sequence_id_1[0:23])
    sequence_id_2 = note_sequence_io.generate_note_sequence_id(
        '/my/file/name', 'your_collection', 'abc')
    self.assertEqual('/id/abc/your_collection/', sequence_id_2[0:24])
    self.assertEqual(sequence_id_1[23:], sequence_id_2[24:])

    sequence_id_3 = note_sequence_io.generate_note_sequence_id(
        '/your/file/name', 'my_collection', 'abc')
    self.assertNotEqual(sequence_id_3[22:], sequence_id_1[23:])
    self.assertNotEqual(sequence_id_3[22:], sequence_id_2[24:])

  def testNoteSequenceRecordWriterAndIterator(self):
    sequences = []
    for i in range(4):
      sequence = music_pb2.NoteSequence()
      sequence.id = str(i)
      sequence.notes.add().pitch = i
      sequences.append(sequence)

    with tempfile.NamedTemporaryFile(prefix='NoteSequenceIoTest') as temp_file:
      with note_sequence_io.NoteSequenceRecordWriter(temp_file.name) as writer:
        for sequence in sequences:
          writer.write(sequence)

      for i, sequence in enumerate(
          note_sequence_io.note_sequence_record_iterator(temp_file.name)):
        self.assertEqual(sequence, sequences[i])

if __name__ == '__main__':
  tf.test.main()
