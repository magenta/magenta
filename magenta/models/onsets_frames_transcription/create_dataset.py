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
"""Beam pipeline for dataset creation."""

import hashlib
import os

import apache_beam as beam
from apache_beam.metrics import Metrics

from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import create_dataset_lib
from magenta.models.onsets_frames_transcription import data
from note_seq import audio_io
from note_seq.protobuf import music_pb2

import tensorflow.compat.v1 as tf

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
              debug_output_directory, split_example, load_audio_with_librosa):
  """Splits wav and midi files for the dataset."""
  tf.logging.info('Splitting %s',
                  input_example.features.feature['id'].bytes_list.value[0])

  wav_data = input_example.features.feature['audio'].bytes_list.value[0]

  ns = music_pb2.NoteSequence.FromString(
      input_example.features.feature['sequence'].bytes_list.value[0])

  Metrics.counter('split_wav', 'read_midi_wav_to_split').inc()

  if not split_example:
    split_examples = audio_label_data_utils.process_record(
        wav_data,
        ns,
        ns.id,
        min_length=0,
        max_length=-1,
        sample_rate=sample_rate,
        allow_empty_notesequence=True,
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
      output_file = 'badexample-' + hashlib.md5(
          ns.id.encode('utf-8')).hexdigest() + '.proto'
      output_path = os.path.join(debug_output_directory, output_file)
      tf.logging.error('Exception processing %s. Writing file to %s', ns.id,
                       output_path)
      with tf.gfile.Open(output_path, 'w') as f:
        f.write(input_example.SerializeToString())
      raise


def multiply_example(ex, num_times):
  return [ex] * num_times


def preprocess_data(
    input_example, preprocess_example_fn, input_tensors_to_example_fn, hparams,
    process_for_training):
  """Preprocess example using data.preprocess_data."""
  with tf.Graph().as_default():
    example_proto = tf.constant(input_example.SerializeToString())

    input_tensors = preprocess_example_fn(
        example_proto=example_proto, hparams=hparams,
        is_training=process_for_training)

    with tf.Session() as sess:
      preprocessed = sess.run(input_tensors)

  example = input_tensors_to_example_fn(preprocessed, hparams)
  Metrics.counter('preprocess_data', 'preprocess_example').inc()
  return example


def mix_examples(mixid_exs, sample_rate, load_audio_with_librosa):
  """Mix several Examples together to create a new example."""
  mixid, exs = mixid_exs
  del mixid

  example_samples = []
  example_sequences = []

  for ex_str in exs:
    ex = tf.train.Example.FromString(ex_str)
    wav_data = ex.features.feature['audio'].bytes_list.value[0]
    if load_audio_with_librosa:
      samples = audio_io.wav_data_to_samples_librosa(wav_data, sample_rate)
    else:
      samples = audio_io.wav_data_to_samples(wav_data, sample_rate)
    example_samples.append(samples)
    ns = music_pb2.NoteSequence.FromString(
        ex.features.feature['sequence'].bytes_list.value[0])
    example_sequences.append(ns)

  mixed_samples, mixed_sequence = audio_label_data_utils.mix_sequences(
      individual_samples=example_samples, sample_rate=sample_rate,
      individual_sequences=example_sequences)

  mixed_wav_data = audio_io.samples_to_wav_data(mixed_samples, sample_rate)

  mixed_id = '::'.join(['mixed'] + [ns.id for ns in example_sequences])
  mixed_sequence.id = mixed_id
  mixed_filename = '::'.join(
      ['mixed'] + [ns.filename for ns in example_sequences])
  mixed_sequence.filename = mixed_filename

  examples = list(audio_label_data_utils.process_record(
      mixed_wav_data,
      mixed_sequence,
      mixed_id,
      min_length=0,
      max_length=-1,
      sample_rate=sample_rate))
  assert len(examples) == 1
  return examples[0]


def pipeline(config_map, dataset_config_map, preprocess_example_fn,
             input_tensors_to_example_fn):
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
      if isinstance(dataset.path, (list, tuple)):
        # If dataset.path is a list, then it's a list of sources to mix together
        # to form new examples. First, do the mixing, then pass the results to
        # the rest of the pipeline.
        id_exs = []
        sourceid_to_exids = []
        for source_id, stem_path in enumerate(dataset.path):
          if dataset.num_mixes is None:
            raise ValueError(
                'If path is not a list, num_mixes must not be None: {}'.format(
                    dataset))
          stem_p = p | 'tfrecord_list_%s_%d' % (dataset.name, source_id) >> (
              beam.Create(data.generate_sharded_filenames(stem_path)))

          # Note that we do not specify a coder when reading here.
          # This is so that the hashing in key_example below can work directly
          # on the serialized version instead of having to re-serialize it.
          # Also, deserializing with a coder and then re-serializing does not
          # always generate the same hash for the same example (likely due to
          # the map fields in tf.train.Example). This is important when reading
          # the same dataset multiple times to mix it with itself.
          stem_p |= 'read_tfrecord_%s_%d' % (dataset.name, source_id) >> (
              beam.io.tfrecordio.ReadAllFromTFRecord())
          stem_p |= 'shuffle_stems_%s_%d' % (dataset.name, source_id) >> (
              beam.Reshuffle())

          # Key all examples with a hash.
          def key_example(ex):
            return (hashlib.sha256(ex).hexdigest(), ex)
          stem_p |= 'add_id_key_%s_%d' % (dataset.name, source_id) >> (
              beam.Map(key_example))
          id_exs.append(stem_p)

          # Create a list of source_id to example id.
          def sourceid_to_exid(id_ex, source_id):
            return (source_id, id_ex[0])
          sourceid_to_exids.append(
              stem_p | 'key_%s_%d' % (dataset.name, source_id) >> (
                  beam.Map(sourceid_to_exid, source_id=source_id)))

        # ('example_hash', serialized_example)
        id_exs = (id_exs
                  | 'id_exs_flatten_%s' % dataset.name >> beam.Flatten()
                  | 'id_exs_distinct_%s' % dataset.name >> beam.Distinct())  # pylint:disable=no-value-for-parameter

        # ('source_id, 'example_hash')
        sourceid_to_exids = (
            sourceid_to_exids | 'sourceid_to_exids_flatten_%s' % dataset.name >>
            beam.Flatten())

        # Pass the list of source id to example IDs to generate_mixes,
        # which will create mixes by selecting random IDs from each source
        # (with replacement). This is represented as a list of example IDs
        # to Mix IDs.
        # Note: beam.Create([0]) is just a single dummy value to allow the
        # sourceid_to_exids to be passed in as a python list so we can do the
        # sampling with numpy.
        exid_to_mixids = (
            p
            | 'create_dummy_%s' % dataset.name >> beam.Create([0])
            | 'generate_mixes_%s' % dataset.name >> beam.Map(
                create_dataset_lib.generate_mixes, num_mixes=dataset.num_mixes,
                sourceid_to_exids=beam.pvalue.AsList(sourceid_to_exids)))
        # Create a list of (Mix ID, Full Example proto). Note: Examples may be
        # present in more than one mix. Then, group by Mix ID.
        def mixid_to_exs(id_ex, exid_to_mixids):
          exid, ex = id_ex
          for mixid in exid_to_mixids[exid]:
            yield mixid, ex
        mixid_exs = (
            id_exs
            | 'mixid_to_exs_%s' % dataset.name >> beam.FlatMap(
                mixid_to_exs,
                exid_to_mixids=beam.pvalue.AsSingleton(exid_to_mixids))
            | 'group_by_key_%s' % dataset.name >> beam.GroupByKey())
        # Take these groups of Examples, mix their audio and sequences to return
        # a single new Example. Then, carry on with the rest of the pipeline
        # like normal.
        split_p = (
            mixid_exs
            | 'mix_examples_%s' % dataset.name >> beam.Map(
                mix_examples, FLAGS.sample_rate, FLAGS.load_audio_with_librosa))
      else:
        if dataset.num_mixes is not None:
          raise ValueError(
              'If path is not a list, num_mixes must be None: {}'.format(
                  dataset))
        split_p = p | 'tfrecord_list_%s' % dataset.name >> beam.Create(
            data.generate_sharded_filenames(dataset.path))
        split_p |= 'read_tfrecord_%s' % dataset.name >> (
            beam.io.tfrecordio.ReadAllFromTFRecord(
                coder=beam.coders.ProtoCoder(tf.train.Example)))
      split_p |= 'shuffle_input_%s' % dataset.name >> beam.Reshuffle()
      split_p |= 'split_wav_%s' % dataset.name >> beam.FlatMap(
          split_wav, min_length=FLAGS.min_length, max_length=FLAGS.max_length,
          sample_rate=FLAGS.sample_rate,
          debug_output_directory=FLAGS.output_directory,
          split_example=dataset.process_for_training,
          load_audio_with_librosa=FLAGS.load_audio_with_librosa)
      if FLAGS.preprocess_examples:
        if dataset.process_for_training:
          mul_name = 'preprocess_multiply_%dx_%s' % (
              FLAGS.preprocess_train_example_multiplier, dataset.name)
          split_p |= mul_name >> beam.FlatMap(
              multiply_example, FLAGS.preprocess_train_example_multiplier)
        split_p |= 'preprocess_%s' % dataset.name >> beam.Map(
            preprocess_data, preprocess_example_fn, input_tensors_to_example_fn,
            hparams, dataset.process_for_training)
      split_p |= 'shuffle_output_%s' % dataset.name >> beam.Reshuffle()
      split_p |= 'write_%s' % dataset.name >> beam.io.WriteToTFRecord(
          os.path.join(FLAGS.output_directory, '%s.tfrecord' % dataset.name),
          coder=beam.coders.ProtoCoder(tf.train.Example))
