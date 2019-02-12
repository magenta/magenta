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
tf.app.flags.DEFINE_string(
    'pipeline_options', '--runner=DirectRunner',
    'Command line flags to use in constructing the Beam pipeline options.')


class SplitWavDoFn(beam.DoFn):
  """Splits wav and midi files for the dataset."""

  def __init__(self, min_length, max_length, sample_rate, split,
               output_directory):
    self._min_length = min_length
    self._max_length = max_length
    self._sample_rate = sample_rate
    self._split = split
    self._output_directory = output_directory

  def process(self, input_example):
    tf.logging.info('Splitting %s',
                    input_example.features.feature['id'].bytes_list.value[0])

    wav_data = input_example.features.feature['audio'].bytes_list.value[0]

    ns = music_pb2.NoteSequence.FromString(
        input_example.features.feature['sequence'].bytes_list.value[0])

    Metrics.counter('split_wav', 'read_midi_wav_to_split').inc()

    if self._split == 'test' or self._split == 'validation':
      # For the 'test' and 'validation' splits, use the full length audio and
      # midi.
      split_examples = split_audio_and_label_data.process_record(
          wav_data,
          ns,
          ns.id,
          min_length=0,
          max_length=-1,
          sample_rate=self._sample_rate)

      for example in split_examples:
        Metrics.counter('split_wav', 'full_example').inc()
        yield example
    else:
      try:
        split_examples = split_audio_and_label_data.process_record(
            wav_data, ns, ns.id, self._min_length, self._max_length,
            self._sample_rate)

        for example in split_examples:
          Metrics.counter('split_wav', 'split_example').inc()
          yield example
      except AssertionError:
        output_file = 'badexample-' + hashlib.md5(ns.id).hexdigest() + '.proto'
        output_path = os.path.join(self._output_directory, output_file)
        tf.logging.error('Exception processing %s. Writing file to %s',
                         ns.id, output_path)
        with tf.gfile.Open(output_path, 'w') as f:
          f.write(input_example.SerializeToString())
        raise


def generate_sharded_filenames(filename):
  match = re.match(r'^([^@]+)@(\d+)$', filename)
  if not match:
    yield filename
  else:
    num_shards = int(match.group(2))
    base = match.group(1)
    for i in range(num_shards):
      yield '{}-{:0=5d}-of-{:0=5d}'.format(base, i, num_shards)


def pipeline():
  """Pipeline for dataset creation."""
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(','))

  with beam.Pipeline(options=pipeline_options) as p:
    tf.flags.mark_flags_as_required(['output_directory'])

    splits = [
        ('train', generate_sharded_filenames(FLAGS.train_tfrecord)),
        ('validation', generate_sharded_filenames(FLAGS.validation_tfrecord)),
        ('test', generate_sharded_filenames(FLAGS.test_tfrecord)),
    ]

    for split_name, split_tfrecord in splits:
      split_p = p | 'tfrecord_list_%s' % split_name >> beam.Create(
          split_tfrecord)
      split_p |= 'read_tfrecord_%s' % split_name >> (
          beam.io.tfrecordio.ReadAllFromTFRecord(
              coder=beam.coders.ProtoCoder(tf.train.Example)))
      split_p |= 'shuffle_input_%s' % split_name >> beam.Reshuffle()
      split_p |= 'split_wav_%s' % split_name >> beam.ParDo(
          SplitWavDoFn(FLAGS.min_length, FLAGS.max_length, FLAGS.sample_rate,
                       split_name, FLAGS.output_directory))
      split_p |= 'shuffle_output_%s' % split_name >> beam.Reshuffle()
      split_p |= 'write_%s' % split_name >> beam.io.WriteToTFRecord(
          os.path.join(FLAGS.output_directory, '%s.tfrecord' % split_name),
          coder=beam.coders.ProtoCoder(tf.train.Example))
