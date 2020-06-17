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
"""Create the tfrecord files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

import glob
import os
import re

from magenta.models.onsets_frames_transcription import audio_label_data_utils

from note_seq import audio_io
from note_seq import midi_io

import six
import tensorflow.compat.v1 as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', None,
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', './',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')

test_dirs = ['ENSTDkCl/MUS', 'ENSTDkAm/MUS']
train_dirs = [
    'AkPnBcht/MUS', 'AkPnBsdf/MUS', 'AkPnCGdD/MUS', 'AkPnStgb/MUS',
    'SptkBGAm/MUS', 'SptkBGCl/MUS', 'StbgTGd2/MUS'
]


def filename_to_id(filename):
  """Translate a .wav or .mid path to a MAPS sequence id."""
  return re.match(r'.*MUS-(.*)_[^_]+\.\w{3}',
                  six.ensure_str(os.path.basename(filename))).group(1)


def generate_train_set(exclude_ids):
  """Generate the train TFRecord."""
  train_file_pairs = []
  for directory in train_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      if filename_to_id(wav_file) not in exclude_ids:
        train_file_pairs.append((wav_file, mid_file))

  train_output_name = os.path.join(FLAGS.output_dir,
                                   'maps_config2_train.tfrecord')

  with tf.python_io.TFRecordWriter(train_output_name) as writer:
    for idx, pair in enumerate(train_file_pairs):
      print('{} of {}: {}'.format(idx, len(train_file_pairs), pair[0]))
      # load the wav data
      wav_data = tf.gfile.Open(pair[0], 'rb').read()
      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])
      for example in audio_label_data_utils.process_record(
          wav_data, ns, pair[0], FLAGS.min_length, FLAGS.max_length,
          FLAGS.sample_rate):
        writer.write(example.SerializeToString())


def generate_test_set():
  """Generate the test TFRecord."""
  test_file_pairs = []
  for directory in test_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      test_file_pairs.append((wav_file, mid_file))

  test_output_name = os.path.join(FLAGS.output_dir,
                                  'maps_config2_test.tfrecord')

  with tf.python_io.TFRecordWriter(test_output_name) as writer:
    for idx, pair in enumerate(test_file_pairs):
      print('{} of {}: {}'.format(idx, len(test_file_pairs), pair[0]))
      # load the wav data and resample it.
      samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
      wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      example = audio_label_data_utils.create_example(pair[0], ns, wav_data)
      writer.write(example.SerializeToString())

  return [filename_to_id(wav) for wav, _ in test_file_pairs]


def main(unused_argv):
  test_ids = generate_test_set()
  generate_train_set(test_ids)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
