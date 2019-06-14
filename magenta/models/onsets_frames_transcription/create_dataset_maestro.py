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

from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import data
from magenta.protobuf import music_pb2

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_directory', None, 'Path to output_directory')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum length for a segment')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum length for a segment')
tf.app.flags.DEFINE_integer('sample_rate', 16000,
                            'sample_rate of the output files')
tf.app.flags.DEFINE_boolean('preprocess_examples', False,
                            'Whether to preprocess examples.')
tf.app.flags.DEFINE_integer(
    'preprocess_train_example_multiplier', 1,
    'How many times to run data preprocessing on each training example. '
    'Useful if preprocessing involves a stochastic process that is useful to '
    'sample multiple times.')
tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('dataset_config', 'maestro',
                           'Name of the dataset config to use.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_string(
    'pipeline_options', '--runner=DirectRunner',
    'Command line flags to use in constructing the Beam pipeline options.')
tf.app.flags.DEFINE_boolean(
    'load_audio_with_librosa', False,
    'Whether to use librosa for sampling audio')


def split_wav(input_example, min_length, max_length, sample_rate,
              output_directory, process_for_training, load_audio_with_librosa):
  """Splits wav and midi files for the dataset."""
  tf.logging.info('Splitting %s',
                  input_example.features.feature['id'].bytes_list.value[0])

  wav_data = input_example.features.feature['audio'].bytes_list.value[0]

  ns = music_pb2.NoteSequence.FromString(
      input_example.features.feature['sequence'].bytes_list.value[0])

  Metrics.counter('split_wav', 'read_midi_wav_to_split').inc()

  if not process_for_training:
    split_examples = audio_label_data_utils.process_record(
        wav_data,
        ns,
        ns.id,
        min_length=0,
        max_length=-1,
        sample_rate=sample_rate,
        load_audio_with_librosa=load_audio_with_librosa)

    for example in split_examples:
      Metrics.counter('split_wav', 'full_example').inc()
      yield example
  else:
    try:
      split_examples = audio_label_data_utils.process_record(
          wav_data,
          ns,
          ns.id,
          min_length=min_length,
          max_length=max_length,
          sample_rate=sample_rate,
          load_audio_with_librosa=load_audio_with_librosa)

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


def multiply_example(ex, num_times):
  return [ex] * num_times


def preprocess_data(input_example, hparams, process_for_training):
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
        sequence_id, sequence, audio, velocity_range, hparams,
        is_training=process_for_training)

    with tf.Session() as sess:
      preprocessed = sess.run(input_tensors)

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'spec':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=preprocessed.spec.flatten())),
              'spectrogram_hash':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(
                          value=[preprocessed.spectrogram_hash])),
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


def pipeline(config_map, dataset_config_map):
  """Pipeline for dataset creation."""
  tf.flags.mark_flags_as_required(['output_directory'])

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(','))

  config = config_map[FLAGS.config]
  hparams = config.hparams
  hparams.parse(FLAGS.hparams)

  datasets = dataset_config_map[FLAGS.dataset_config]

  if tf.gfile.Exists(FLAGS.output_directory):
    raise ValueError(
        'Output directory %s already exists!' % FLAGS.output_directory)
  tf.gfile.MakeDirs(FLAGS.output_directory)
  with tf.gfile.Open(
      os.path.join(FLAGS.output_directory, 'config.txt'), 'w') as f:
    f.write('\n\n'.join([
        'min_length: {}'.format(FLAGS.min_length),
        'max_length: {}'.format(FLAGS.max_length),
        'sample_rate: {}'.format(FLAGS.sample_rate),
        'preprocess_examples: {}'.format(FLAGS.preprocess_examples),
        'preprocess_train_example_multiplier: {}'.format(
            FLAGS.preprocess_train_example_multiplier),
        'config: {}'.format(FLAGS.config),
        'hparams: {}'.format(hparams.to_json(sort_keys=True)),
        'dataset_config: {}'.format(FLAGS.dataset_config),
        'datasets: {}'.format(datasets),
    ]))

  with beam.Pipeline(options=pipeline_options) as p:
    for dataset in datasets:
      split_p = p | 'tfrecord_list_%s' % dataset.name >> beam.Create(
          generate_sharded_filenames(dataset.path))
      split_p |= 'read_tfrecord_%s' % dataset.name >> (
          beam.io.tfrecordio.ReadAllFromTFRecord(
              coder=beam.coders.ProtoCoder(tf.train.Example)))
      split_p |= 'shuffle_input_%s' % dataset.name >> beam.Reshuffle()
      split_p |= 'split_wav_%s' % dataset.name >> beam.FlatMap(
          split_wav, FLAGS.min_length, FLAGS.max_length, FLAGS.sample_rate,
          FLAGS.output_directory, dataset.process_for_training,
          FLAGS.load_audio_with_librosa)
      if FLAGS.preprocess_examples:
        if dataset.process_for_training:
          mul_name = 'preprocess_multiply_%dx_%s' % (
              FLAGS.preprocess_train_example_multiplier, dataset.name)
          split_p |= mul_name >> beam.FlatMap(
              multiply_example, FLAGS.preprocess_train_example_multiplier)
        split_p |= 'preprocess_%s' % dataset.name >> beam.Map(
            preprocess_data, hparams, dataset.process_for_training)
      split_p |= 'shuffle_output_%s' % dataset.name >> beam.Reshuffle()
      split_p |= 'write_%s' % dataset.name >> beam.io.WriteToTFRecord(
          os.path.join(FLAGS.output_directory, '%s.tfrecord' % dataset.name),
          coder=beam.coders.ProtoCoder(tf.train.Example))
