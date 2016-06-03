"""For reading/writing serialized NoteSequence protos to/from TFRecords files."""

import hashlib
import tensorflow as tf

from magenta.protobuf import music_pb2


def generate_id(filename, collection_name, source_type):
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
  """An iterator that reads and parses NoteSequence protos from a TFRecords file.

  Args:
    path: The path to the TFRecords file constaining serialized NoteSequences.

  Yields:
    NoteSequence protos.

  Raises:
    IOError: If `path` cannot be opened for reading.
  """
  reader = tf.python_io.tf_record_iterator(path)
  for serialized_sequence in reader:
    yield music_pb2.NoteSequence.FromString(serialized_sequence)


class NoteSequenceRecordWriter(tf.python_io.TFRecordWriter):
  """A class to write serialized NoteSequence protos to a TFRecords file.

  This class implements `__enter__` and `__exit__`, and can be used in `with`
  blocks like a normal file.

  @@__init__
  @@write
  @@close
  """

  def write(self, note_sequence):
    """Serizes a NoteSequence proto and writes it to the file.

    Args:
      sequence: A NoteSequence proto to write.
    """
    tf.python_io.TFRecordWriter.write(self, note_sequence.SerializeToString())
