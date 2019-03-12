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

"""Beam pipeline for MAESTRO dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import re

import apache_beam as beam
from apache_beam.metrics import Metrics

from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import split_audio_and_label_data
from magenta.protobuf import music_pb2

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_tfrecord', 'gs://magentadata/datasets/maestro/v1.0.0/'
    'maestro-v1.0.0_ns_wav_train.tfrecord@10', 'Path to training tfrecord')
tf.app.flags.DEFINE_string(
    'test_tfrecord', 'gs://magentadata/datasets/maestro/v1.0.0/'
    'maestro-v1.0.0_ns_wav_test.tfrecord@10', 'Path to training tfrecord')
tf.app.flags.DEFINE_string(
    'validation_tfrecord', 'gs://magentadata/datasets/maestro/v1.0.0/'
    'maestro-v1.0.0_ns_wav_validation.tfrecord@10', 'Path to training tfrecord')
tf.app.flags.DEFINE_string('output_directory', None, 'Path to output_directory')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum length for a segment')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum length for a segment')
tf.app.flags.DEFINE_integer('sample_rate', 16000,
                            'sample_rate of the output files')
tf.app.flags.DEFINE_boolean('preprocess_examples', False,
                            'Whether to preprocess examples.')
tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_string(
    'pipeline_options', '--runner=DirectRunner',
    'Command line flags to use in constructing the Beam pipeline options.')


def split_wav(input_example, min_length, max_length, sample_rate,
              output_directory, chunk_files):
  """Splits wav and midi files for the dataset."""
  tf.logging.info('Splitting %s',
                  input_example.features.feature['id'].bytes_list.value[0])

  wav_data = input_example.features.feature['audio'].bytes_list.value[0]

  ns = music_pb2.NoteSequence.FromString(
      input_example.features.feature['sequence'].bytes_list.value[0])

  Metrics.counter('split_wav', 'read_midi_wav_to_split').inc()

  if not chunk_files:
    split_examples = split_audio_and_label_data.process_record(
        wav_data,
        ns,
        ns.id,
        min_length=0,
        max_length=-1,
        sample_rate=sample_rate)

    for example in split_examples:
      Metrics.counter('split_wav', 'full_example').inc()
      yield example
  else:
    try:
      split_examples = split_audio_and_label_data.process_record(
          wav_data, ns, ns.id, min_length, max_length, sample_rate)

      for example in split_examples:
        Metrics.counter('split_wav', 'split_example').inc()
        yield example
    except AssertionError:
      output_file = 'badexample-' + hashlib.md5(ns.id).hexdigest() + '.proto'
      output_path = os.path.join(output_directory, output_file)
      tf.logging.error('Exception processing %s. Writing file to %s', ns.id,
                       output_path)
      with tf.gfile.Open(output_path, 'w') as f:
        f.write(input_example.SerializeToString())
      raise


def preprocess_data(input_example, hparams):
  """Preprocess example using data.preprocess_data."""
  with tf.Graph().as_default():
    audio = tf.constant(
        input_example.features.feature['audio'].bytes_list.value[0])

    sequence = tf.constant(
        input_example.features.feature['sequence'].bytes_list.value[0])
    sequence_id = tf.constant(
        input_example.features.feature['id'].bytes_list.value[0])
    velocity_range = tf.constant(
        input_example.features.feature['velocity_range'].bytes_list.value[0])

    input_tensors = data.preprocess_data(
        sequence_id, sequence, audio, velocity_range, hparams, is_training=True)

    with tf.Session() as sess:
      preprocessed = sess.run(input_tensors)

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'spec':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=preprocessed.spec.flatten())),
              'labels':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=preprocessed.labels.flatten())),
              'label_weights':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=preprocessed.label_weights.flatten())),
              'length':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(
                          value=[preprocessed.length])),
              'onsets':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=preprocessed.onsets.flatten())),
              'offsets':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=preprocessed.offsets.flatten())),
              'velocities':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=preprocessed.velocities.flatten())),
              'sequence_id':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[preprocessed.sequence_id])),
              'note_sequence':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[preprocessed.note_sequence])),
          }))
  Metrics.counter('preprocess_data', 'preprocess_example').inc()
  return example


def generate_sharded_filenames(filenames):
  for filename in filenames.split(','):
    match = re.match(r'^([^@]+)@(\d+)$', filename)
    if not match:
      yield filename
    else:
      num_shards = int(match.group(2))
      base = match.group(1)
      for i in range(num_shards):
        yield '{}-{:0=5d}-of-{:0=5d}'.format(base, i, num_shards)


def pipeline(config_map):
  """Pipeline for dataset creation."""
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(','))

  config = config_map[FLAGS.config]
  hparams = config.hparams
  hparams.parse(FLAGS.hparams)

  with beam.Pipeline(options=pipeline_options) as p:
    tf.flags.mark_flags_as_required(['output_directory'])

    splits = [
        ('train', generate_sharded_filenames(FLAGS.train_tfrecord), True),
        ('train-nosplit', generate_sharded_filenames(FLAGS.train_tfrecord),
         False),
        ('validation', generate_sharded_filenames(FLAGS.validation_tfrecord),
         False),
        ('test', generate_sharded_filenames(FLAGS.test_tfrecord), False),
    ]

    for split_name, split_tfrecord, chunk_files in splits:
      split_p = p | 'tfrecord_list_%s' % split_name >> beam.Create(
          split_tfrecord)
      split_p |= 'read_tfrecord_%s' % split_name >> (
          beam.io.tfrecordio.ReadAllFromTFRecord(
              coder=beam.coders.ProtoCoder(tf.train.Example)))
      split_p |= 'shuffle_input_%s' % split_name >> beam.Reshuffle()
      split_p |= 'split_wav_%s' % split_name >> beam.FlatMap(
          split_wav, FLAGS.min_length, FLAGS.max_length, FLAGS.sample_rate,
          FLAGS.output_directory, chunk_files)
      if FLAGS.preprocess_examples:
        split_p |= 'preprocess_%s' % split_name >> beam.Map(
            preprocess_data, hparams)
      split_p |= 'shuffle_output_%s' % split_name >> beam.Reshuffle()
      split_p |= 'write_%s' % split_name >> beam.io.WriteToTFRecord(
          os.path.join(FLAGS.output_directory, '%s.tfrecord' % split_name),
          coder=beam.coders.ProtoCoder(tf.train.Example))
