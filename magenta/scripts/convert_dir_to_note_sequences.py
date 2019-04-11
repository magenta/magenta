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

r""""Converts music files to NoteSequence protos and writes TFRecord file.

Currently supports MIDI (.mid, .midi) and MusicXML (.xml, .mxl) files.

Example usage:
  $ python magenta/scripts/convert_dir_to_note_sequences.py \
    --input_dir=/path/to/input/dir \
    --output_file=/path/to/tfrecord/file \
    --log=INFO
"""

import os

from magenta.music import abc_parser
from magenta.music import midi_io
from magenta.music import musicxml_reader
from magenta.music import note_sequence_io
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', None,
                           'Directory containing files to convert.')
tf.app.flags.DEFINE_string('output_file', None,
                           'Path to output TFRecord file. Will be overwritten '
                           'if it already exists.')
tf.app.flags.DEFINE_bool('recursive', False,
                         'Whether or not to recurse into subdirectories.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


def convert_files(root_dir, sub_dir, writer, recursive=False):
  """Converts files.

  Args:
    root_dir: A string specifying a root directory.
    sub_dir: A string specifying a path to a directory under `root_dir` in which
        to convert contents.
    writer: A TFRecord writer
    recursive: A boolean specifying whether or not recursively convert files
        contained in subdirectories of the specified directory.

  Returns:
    A map from the resulting Futures to the file paths being converted.
  """
  dir_to_convert = os.path.join(root_dir, sub_dir)
  tf.logging.info("Converting files in '%s'.", dir_to_convert)
  files_in_dir = tf.gfile.ListDirectory(os.path.join(dir_to_convert))
  recurse_sub_dirs = []
  written_count = 0
  for file_in_dir in files_in_dir:
    tf.logging.log_every_n(tf.logging.INFO, '%d files converted.',
                           1000, written_count)
    full_file_path = os.path.join(dir_to_convert, file_in_dir)
    if (full_file_path.lower().endswith('.mid') or
        full_file_path.lower().endswith('.midi')):
      try:
        sequence = convert_midi(root_dir, sub_dir, full_file_path)
      except Exception as exc:  # pylint: disable=broad-except
        tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
        continue
      if sequence:
        writer.write(sequence)
    elif (full_file_path.lower().endswith('.xml') or
          full_file_path.lower().endswith('.mxl')):
      try:
        sequence = convert_musicxml(root_dir, sub_dir, full_file_path)
      except Exception as exc:  # pylint: disable=broad-except
        tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
        continue
      if sequence:
        writer.write(sequence)
    elif full_file_path.lower().endswith('.abc'):
      try:
        sequences = convert_abc(root_dir, sub_dir, full_file_path)
      except Exception as exc:  # pylint: disable=broad-except
        tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
        continue
      if sequences:
        for sequence in sequences:
          writer.write(sequence)
    else:
      if recursive and tf.gfile.IsDirectory(full_file_path):
        recurse_sub_dirs.append(os.path.join(sub_dir, file_in_dir))
      else:
        tf.logging.warning(
            'Unable to find a converter for file %s', full_file_path)

  for recurse_sub_dir in recurse_sub_dirs:
    convert_files(root_dir, recurse_sub_dir, writer, recursive)


def convert_midi(root_dir, sub_dir, full_file_path):
  """Converts a midi file to a sequence proto.

  Args:
    root_dir: A string specifying the root directory for the files being
        converted.
    sub_dir: The directory being converted currently.
    full_file_path: the full path to the file to convert.

  Returns:
    Either a NoteSequence proto or None if the file could not be converted.
  """
  try:
    sequence = midi_io.midi_to_sequence_proto(
        tf.gfile.GFile(full_file_path, 'rb').read())
  except midi_io.MIDIConversionError as e:
    tf.logging.warning(
        'Could not parse MIDI file %s. It will be skipped. Error was: %s',
        full_file_path, e)
    return None
  sequence.collection_name = os.path.basename(root_dir)
  sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
  sequence.id = note_sequence_io.generate_note_sequence_id(
      sequence.filename, sequence.collection_name, 'midi')
  tf.logging.info('Converted MIDI file %s.', full_file_path)
  return sequence


def convert_musicxml(root_dir, sub_dir, full_file_path):
  """Converts a musicxml file to a sequence proto.

  Args:
    root_dir: A string specifying the root directory for the files being
        converted.
    sub_dir: The directory being converted currently.
    full_file_path: the full path to the file to convert.

  Returns:
    Either a NoteSequence proto or None if the file could not be converted.
  """
  try:
    sequence = musicxml_reader.musicxml_file_to_sequence_proto(full_file_path)
  except musicxml_reader.MusicXMLConversionError as e:
    tf.logging.warning(
        'Could not parse MusicXML file %s. It will be skipped. Error was: %s',
        full_file_path, e)
    return None
  sequence.collection_name = os.path.basename(root_dir)
  sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
  sequence.id = note_sequence_io.generate_note_sequence_id(
      sequence.filename, sequence.collection_name, 'musicxml')
  tf.logging.info('Converted MusicXML file %s.', full_file_path)
  return sequence


def convert_abc(root_dir, sub_dir, full_file_path):
  """Converts an abc file to a sequence proto.

  Args:
    root_dir: A string specifying the root directory for the files being
        converted.
    sub_dir: The directory being converted currently.
    full_file_path: the full path to the file to convert.

  Returns:
    Either a NoteSequence proto or None if the file could not be converted.
  """
  try:
    tunes, exceptions = abc_parser.parse_abc_tunebook(
        tf.gfile.GFile(full_file_path, 'rb').read())
  except abc_parser.ABCParseError as e:
    tf.logging.warning(
        'Could not parse ABC file %s. It will be skipped. Error was: %s',
        full_file_path, e)
    return None

  for exception in exceptions:
    tf.logging.warning(
        'Could not parse tune in ABC file %s. It will be skipped. Error was: '
        '%s', full_file_path, exception)

  sequences = []
  for idx, tune in tunes.iteritems():
    tune.collection_name = os.path.basename(root_dir)
    tune.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
    tune.id = note_sequence_io.generate_note_sequence_id(
        '{}_{}'.format(tune.filename, idx), tune.collection_name, 'abc')
    sequences.append(tune)
    tf.logging.info('Converted ABC file %s.', full_file_path)
  return sequences


def convert_directory(root_dir, output_file, recursive=False):
  """Converts files to NoteSequences and writes to `output_file`.

  Input files found in `root_dir` are converted to NoteSequence protos with the
  basename of `root_dir` as the collection_name, and the relative path to the
  file from `root_dir` as the filename. If `recursive` is true, recursively
  converts any subdirectories of the specified directory.

  Args:
    root_dir: A string specifying a root directory.
    output_file: Path to TFRecord file to write results to.
    recursive: A boolean specifying whether or not recursively convert files
        contained in subdirectories of the specified directory.
  """
  with note_sequence_io.NoteSequenceRecordWriter(output_file) as writer:
    convert_files(root_dir, '', writer, recursive)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  if not FLAGS.input_dir:
    tf.logging.fatal('--input_dir required')
    return
  if not FLAGS.output_file:
    tf.logging.fatal('--output_file required')
    return

  input_dir = os.path.expanduser(FLAGS.input_dir)
  output_file = os.path.expanduser(FLAGS.output_file)
  output_dir = os.path.dirname(output_file)

  if output_dir:
    tf.gfile.MakeDirs(output_dir)

  convert_directory(input_dir, output_file, FLAGS.recursive)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
