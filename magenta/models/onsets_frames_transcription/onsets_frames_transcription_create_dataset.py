# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Create the recordio files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import re

import librosa
import numpy as np
import tensorflow as tf

from magenta.models.onsets_frames_transcription import create_dataset_util
from magenta.music import audio_io
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', None,
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', './',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')

TEST_DIRS = ['ENSTDkCl/MUS', 'ENSTDkAm/MUS']
TRAIN_DIRS = ['AkPnBcht/MUS', 'AkPnBsdf/MUS', 'AkPnCGdD/MUS', 'AkPnStgb/MUS',
              'SptkBGAm/MUS', 'SptkBGCl/MUS', 'StbgTGd2/MUS']


def filename_to_id(filename):
  """Translate a .wav or .mid path to a MAPS sequence id."""
  return re.match(r'.*MUS-(.*)_[^_]+\.\w{3}',
                  os.path.basename(filename)).group(1)


def generate_train_set(exclude_ids):
  """Generate the train TFRecord."""
  train_file_pairs = []
  for directory in TRAIN_DIRS:
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
    for pair in train_file_pairs:
      print(pair)
      # load the wav data
      wav_data = tf.gfile.Open(pair[0], 'rb').read()
      samples = audio_io.wav_data_to_samples(wav_data, FLAGS.sample_rate)
      samples = librosa.util.normalize(samples, norm=np.inf)

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      splits = create_dataset_util.find_split_points(
          ns, samples, FLAGS.sample_rate, FLAGS.min_length, FLAGS.max_length)

      velocities = [note.velocity for note in ns.notes]
      velocity_max = np.max(velocities)
      velocity_min = np.min(velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)

      for start, end in zip(splits[:-1], splits[1:]):
        if end - start < FLAGS.min_length:
          continue

        new_ns = sequences_lib.extract_subsequence(ns, start, end)
        new_wav_data = audio_io.crop_wav_data(wav_data, FLAGS.sample_rate,
                                              start, end - start)
        example = tf.train.Example(features=tf.train.Features(feature={
            'id':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[pair[0]]
                )),
            'sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_ns.SerializeToString()]
                )),
            'audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_wav_data]
                )),
            'velocity_range':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_velocity_tuple.SerializeToString()]
                )),
            }))
        writer.write(example.SerializeToString())


def generate_test_set():
  """Generate the test TFRecord."""
  test_file_pairs = []
  for directory in TEST_DIRS:
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
    for pair in test_file_pairs:
      print(pair)
      # load the wav data and resample it.
      samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
      wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      velocities = [note.velocity for note in ns.notes]
      velocity_max = np.max(velocities)
      velocity_min = np.min(velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)

      example = tf.train.Example(features=tf.train.Features(feature={
          'id':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[pair[0]]
              )),
          'sequence':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[ns.SerializeToString()]
              )),
          'audio':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[wav_data]
              )),
          'velocity_range':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[new_velocity_tuple.SerializeToString()]
              )),
          }))
      writer.write(example.SerializeToString())

  return [filename_to_id(wav) for wav, _ in test_file_pairs]


def main(unused_argv):
  test_ids = generate_test_set()
  generate_train_set(test_ids)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
