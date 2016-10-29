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
"""For reading/writing serialized NoteSequence protos to/from TFRecord files."""

import hashlib

# internal imports
import tensorflow as tf

from magenta.protobuf import music_pb2


def generate_note_sequence_id(filename, collection_name, source_type):
  """Generates a unique ID for a sequence.

  The format is:'/id/<type>/<collection name>/<hash>'.

  Args:
    filename: The string path to the source file relative to the root of the
        collection.
    collection_name: The collection from which the file comes.
    source_type: The source type as a string (e.g. "midi" or "abc").

  Returns:
    The generated sequence ID as a string.
  """
  # TODO(adarob): Replace with FarmHash when it becomes a part of TensorFlow.
  filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
  return '/id/%s/%s/%s' % (
      source_type.lower(), collection_name, filename_fingerprint.hexdigest())


def note_sequence_record_iterator(path):
  """An iterator that reads and parses NoteSequence protos from a TFRecord file.

  Args:
    path: The path to the TFRecord file containing serialized NoteSequences.

  Yields:
    NoteSequence protos.

  Raises:
    IOError: If `path` cannot be opened for reading.
  """
  reader = tf.python_io.tf_record_iterator(path)
  for serialized_sequence in reader:
    yield music_pb2.NoteSequence.FromString(serialized_sequence)


class NoteSequenceRecordWriter(tf.python_io.TFRecordWriter):
  """A class to write serialized NoteSequence protos to a TFRecord file.

  This class implements `__enter__` and `__exit__`, and can be used in `with`
  blocks like a normal file.

  @@__init__
  @@write
  @@close
  """

  def write(self, note_sequence):
    """Serializes a NoteSequence proto and writes it to the file.

    Args:
      note_sequence: A NoteSequence proto to write.
    """
    tf.python_io.TFRecordWriter.write(self, note_sequence.SerializeToString())
