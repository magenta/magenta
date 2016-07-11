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
"""Converts NoteSequence protos to SequenceExample protos.

Protos (or protocol buffers) are stored in TFRecord files.
Run convert_midi_dir_to_note_sequences.py to generate a TFRecord
of Sequence protos from MIDI files. Run this script to extract
melodies from those sequences for training models.
"""

import logging
import random

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import note_sequence_io
from magenta.protobuf import music_pb2


def run_conversion(melody_encoder_decoder, note_sequences_file, train_output,
                   eval_output=None, eval_ratio=0.0):
  """Loop that converts NoteSequence protos to SequenceExample protos.

  Args:
    melody_encoder_decoder: A melodies_lib.MelodyEncoderDecoder object.
    note_sequences_file: String path pointing to TFRecord file of NoteSequence
        protos.
    train_output: String path to TFRecord file that training samples will be
        saved to.
    eval_output: If set, string path to TFRecord file that evaluation samples
        will be saved to. Omit this argument to not produce an eval set.
    eval_ratio: Fraction of input that will be saved to eval set. A random
        partition is chosen, so the actual train/eval ratio will vary.
  """
  reader = note_sequence_io.note_sequence_record_iterator(note_sequences_file)
  train_writer = tf.python_io.TFRecordWriter(train_output)
  eval_writer = (tf.python_io.TFRecordWriter(eval_output)
                 if eval_output else None)

  input_count = 0
  train_output_count = 0
  eval_output_count = 0
  logging.info('Extracting melodies...')
  for sequence_data in reader:
    # Only extract melodies from 4/4 time music.
    if not (sequence_data.time_signatures[0].numerator == 4 and
            sequence_data.time_signatures[0].denominator == 4):
      continue
    extracted_melodies = melodies_lib.extract_melodies(sequence_data)
    for melody in extracted_melodies:
      sequence_example = melody_encoder_decoder.encode(melody)
      serialized = sequence_example.SerializeToString()
      if eval_writer and random.random() < eval_ratio:
        eval_writer.write(serialized)
        eval_output_count += 1
      else:
        train_writer.write(serialized)
        train_output_count += 1
    input_count += 1
    if input_count % 10 == 0:
      logging.info('Extracted %d melodies from %d sequences.',
                   eval_output_count + train_output_count,
                   input_count)

  logging.info('Done.\nExtracted %d melodies from %d sequences.',
               eval_output_count + train_output_count,
               input_count)
  logging.info('Extracted %d melodies for training.', train_output_count)
  if eval_writer:
    logging.info('Extracted %d melodies for evaluation.', eval_output_count)
