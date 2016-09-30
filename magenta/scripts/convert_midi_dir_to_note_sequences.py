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
r""""Converts MIDIs to NoteSequence protos and writes TFRecord file.

Example usage:
  $ bazel build magenta/scripts:convert_midi_dir_to_note_sequences

  $ ./bazel-bin/magenta/scripts/convert_midi_dir_to_note_sequences \
    --midi_dir=/path/to/midi/dir \
    --output_file=/path/to/tfrecord/file \
    --recursive
"""

import os

# internal imports
import tensorflow as tf

from magenta.music import midi_io
from magenta.music import note_sequence_io

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('midi_dir', None,
                           'Directory containing MIDI files to convert.')
tf.app.flags.DEFINE_string('output_file', None,
                           'Path to output TFRecord file. Will be overwritten '
                           'if it already exists.')
tf.app.flags.DEFINE_bool('recursive', False,
                         'Whether or not to recurse into subdirectories.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


def convert_directory(root_dir, sub_dir, sequence_writer, recursive=False):
  """Converts MIDIs to NoteSequences and writes to `sequence_writer`.

  MIDI files found in the specified directory specified by the combination of
  `root_dir` and `sub_dir` and converted to NoteSequence protos with the
  basename of `root_dir` as the collection_name, and the relative path to the
  MIDI file from `root_dir` as the filename. If `recursive` is true, recursively
  converts any subdirectories of the specified directory.

  Args:
    root_dir: A string specifying a root directory.
    sub_dir: A string specifying a path to a directory under `root_dir` in which
        to convert MIDI contents.
    sequence_writer: A NoteSequenceRecordWriter to write the resulting
        NoteSequence protos to.
    recursive: A boolean specifying whether or not recursively convert MIDIs
        contained in subdirectories of the specified directory.

  Returns:
    The number of NoteSequence protos written as an integer.
  """
  dir_to_convert = os.path.join(root_dir, sub_dir)
  tf.logging.info("Converting MIDI files in '%s'.", dir_to_convert)
  files_in_dir = tf.gfile.ListDirectory(os.path.join(dir_to_convert))
  recurse_sub_dirs = []
  sequences_written = 0
  sequences_skipped = 0
  for file_in_dir in files_in_dir:
    full_file_path = os.path.join(dir_to_convert, file_in_dir)
    if tf.gfile.IsDirectory(full_file_path):
      if recursive:
        recurse_sub_dirs.append(os.path.join(sub_dir, file_in_dir))
      continue
    try:
      sequence = midi_io.midi_to_sequence_proto(
          tf.gfile.FastGFile(full_file_path).read())
    except midi_io.MIDIConversionError as e:
      tf.logging.warning(
          'Could not parse MIDI file %s. It will be skipped. Error was: %s',
          full_file_path, e)
      sequences_skipped += 1
      continue
    sequence.collection_name = os.path.basename(root_dir)
    sequence.filename = os.path.join(sub_dir, file_in_dir)
    sequence.id = note_sequence_io.generate_note_sequence_id(
        sequence.filename, sequence.collection_name, 'midi')
    sequence_writer.write(sequence)
    sequences_written += 1
  tf.logging.info("Converted %d MIDI files in '%s'.", sequences_written,
                  dir_to_convert)
  tf.logging.info('Could not parse %d MIDI files.', sequences_skipped)
  for recurse_sub_dir in recurse_sub_dirs:
    sequences_written += convert_directory(
        root_dir, recurse_sub_dir, sequence_writer, recursive)
  return sequences_written


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  if not FLAGS.midi_dir:
    tf.logging.fatal('--midi_dir required')
    return
  if not FLAGS.output_file:
    tf.logging.fatal('--output_file required')
    return

  FLAGS.midi_dir = os.path.expanduser(FLAGS.midi_dir)
  FLAGS.output_file = os.path.expanduser(FLAGS.output_file)

  if not os.path.exists(os.path.dirname(FLAGS.output_file)):
    os.makedirs(os.path.dirname(FLAGS.output_file))

  with note_sequence_io.NoteSequenceRecordWriter(
      FLAGS.output_file) as sequence_writer:
    sequences_written = convert_directory(FLAGS.midi_dir, '', sequence_writer,
                                          FLAGS.recursive)
    tf.logging.info("Wrote %d NoteSequence protos to '%s'", sequences_written,
                    FLAGS.output_file)


def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
