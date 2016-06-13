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
"""Converts Sequence protos to SequenceExample protos.

Protos (or protocol buffers) are stored in TFRecord files.
Run convert_midi_dir_to_note_sequences.py to generate a TFRecord
of Sequence protos from MIDI files. Run this script to extract
melodies from those sequences for training models.
"""

import logging
import random
import sys
import tensorflow as tf

from magenta.lib import encoders
from magenta.lib import melodies_lib
from magenta.protobuf import music_pb2


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('train_output', None,
                           'TFRecord to write SequenceExample protos to. '
                           'Contains training set.')
tf.app.flags.DEFINE_string('eval_output', None,
                           'TFRecord to write SequenceExample protos to. '
                           'Contains eval set. No eval set is produced if '
                           'this flag is not set.')
tf.app.flags.DEFINE_float('eval_ratio', 0.0,
                          'Fraction of input to set asside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_string('encoder', 'basic_one_hot_encoder',
                           'Name of the function in the encoders library to use '
                           'for converting melodies into SequenceExample protos.')


def run_conversion(encoder, sequences_file, train_output, eval_output='', eval_ratio=0.0):
  """Loop that converts NoteSequence protos to SequenceExample protos.

  Args:
    encoder: String name of encoder function from encoders.py to use.
    sequences_file: String path pointing to TFRecord file of NoteSequence
        protos.
    train_output: String path to TFRecord file that training samples will be
        saved to.
    eval_output: If set, string path to TFRecord file that evaluation samples
        will be saved to. Omit this argument to not produce an eval set.
    eval_ratio: Fraction of input that will be saved to eval set. A random
        partition is chosen, so the actual train/eval ratio will vary.
  """
  encoder_func = getattr(encoders, encoder)

  reader = tf.python_io.tf_record_iterator(sequences_file)
  train_writer = tf.python_io.TFRecordWriter(train_output)
  eval_writer = (tf.python_io.TFRecordWriter(eval_output)
                 if eval_output else None)

  input_count = 0
  train_output_count = 0
  eval_output_count = 0
  for buf in reader:
    sequence_data = music_pb2.NoteSequence()
    sequence_data.ParseFromString(buf)
    extracted_melodies = melodies_lib.extract_melodies(sequence_data)
    for melody in extracted_melodies:
      sequence_example, _ = encoder_func(melody)
      serialized = sequence_example.SerializeToString()
      if eval_writer and random.random() < eval_ratio:
        eval_writer.write(serialized)
        eval_output_count += 1
      else:
        train_writer.write(serialized)
        train_output_count += 1
    input_count += 1
    tf.logging.log_every_n(logging.INFO, 
                           'Extracted %d melodies from %d sequences.',
                           500,
                           eval_output_count + train_output_count,
                           input_count)

  logging.info('Found %d sequences', input_count)
  logging.info('Extracted %d melodies for training.', train_output_count)
  if eval_writer:
    logging.info('Extracted %d melodies for evaluation.', eval_output_count)


def main(unused_argv):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  run_conversion(encoder=FLAGS.encoder, sequences_file=FLAGS.input,
                 train_output=FLAGS.train_output,
                 eval_output=FLAGS.eval_output, eval_ratio=FLAGS.eval_ratio)


if __name__ == '__main__':
  tf.app.run()
