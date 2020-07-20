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
r"""Beam job to preprocess a TFRecord of NoteSequences for training a MusicVAE.

Splits a TFRecord of NoteSequences (as prepared by `convert_to_note_sequences`)
into individual examples using the DataConverter in the given config, outputting
the resulting unique sequences in a new TFRecord file. The ID field of each new
NoteSequence will contain a comma-separated list of the original NoteSequence(s)
it was extracted from.

NOTE: for large datasets you will want to use a distributed platform like
Google Cloud DataFlow (https://beam.apache.org/documentation/runners/dataflow/).

Example Usage:
====================
CONFIG=cat-mel_2bar_small

python -m magenta.models.music_vae.preprocess_tfrecord \
--input_tfrecord=/path/to/tfrecords/train.tfrecord \
--output_tfrecord=/path/to/tfrecords/train-$CONFIG.tfrecord \
--output_shards=10 \
--config=$CONFIG \
--alsologtostderr

If running on DataFlow, you'll need to set the `--pipeline_options` flag using
the execution parameters described at
https://cloud.google.com/dataflow/docs/guides/specifying-exec-params
E.g., `--pipeline_options=--runner=DataFlowRunner,--project=<my-project-id>`.

"""

import typing

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam import typehints
from apache_beam.metrics import Metrics as beam_metrics
from magenta.models.music_vae import configs
import note_seq
from note_seq.sequences_lib import split_note_sequence_on_time_changes
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_tfrecord', None,
    'Filepattern matching input TFRecord file(s).')
flags.DEFINE_string(
    'output_tfrecord', None,
    'The prefx for the output TFRecord file(s).')
flags.DEFINE_integer(
    'output_shards', 32,
    'The number of output shards.')
flags.DEFINE_string(
    'config', None,
    'The name of the model config to use.')
flags.DEFINE_bool(
    'enable_filtering', True,
    'If True, enables max_total_time, max_num_notes, min_velocities, '
    'min_metric_positions, is_drum, and drums_only flags.')
flags.DEFINE_integer(
    'max_total_time', 1800,
    'NoteSequences longer than this (in seconds) will be skipped.')
flags.DEFINE_integer(
    'max_num_notes', 10000,
    'NoteSequences with more than this many notes will be skipped.')
flags.DEFINE_integer(
    'min_velocities', 1,
    'NoteSequences with fewer unique velocities than this will be skipped.')
flags.DEFINE_integer(
    'min_metric_positions', 1,
    'NoteSequences with fewer unique metric positions between quarter notes '
    'than this will be skipped.')
flags.DEFINE_bool(
    'is_drum', None,
    'If None, filtering will consider drums and non-drums. If True, only drums '
    'will be considered. If False, only non-drums will be considered.')
flags.DEFINE_bool(
    'drums_only', False,
    'If True, NoteSequences with non-drum instruments will be skipped.')

flags.DEFINE_list(
    'pipeline_options', '--runner=DirectRunner',
    'A comma-separated list of command line arguments to be used as options '
    'for the Beam Pipeline.')


@typehints.with_input_types(note_seq.NoteSequence)
@typehints.with_output_types(typing.Tuple[note_seq.NoteSequence, str])
class ExtractExamplesDoFn(beam.DoFn):
  """Encodes each NoteSequence example and emits the results in a tf.Example."""

  def __init__(self, config_name, filters, *unused_args, **unused_kwargs):
    super(ExtractExamplesDoFn, self).__init__(*unused_args, **unused_kwargs)
    self._config = configs.CONFIG_MAP[config_name]
    self._config.data_converter.max_outputs_per_notesequence = None
    self._filters = filters

  def _process_ns(self, ns):
    if self._filters:
      if ns.total_time > self._filters['max_total_time']:
        logging.info('Skipping %s: total_time=%f', ns.id, ns.total_time)
        beam_metrics.counter('ExtractExamplesDoFn', 'filtered-too-long').inc()
        return
      if len(ns.notes) > self._filters['max_num_notes']:
        logging.info('Skipping %s: num_notes=%d', ns.id, len(ns.notes))
        beam_metrics.counter(
            'ExtractExamplesDoFn', 'filtered-too-many-notes').inc()
        return

      try:
        qns = note_seq.quantize_note_sequence(ns, steps_per_quarter=16)
      except (note_seq.BadTimeSignatureError,
              note_seq.NonIntegerStepsPerBarError, note_seq.NegativeTimeError):
        beam_metrics.counter('ExtractExamplesDoFn', 'quantize-failed').inc()
        return

      vels = set()
      metric_positions = set()
      drums_only = True
      for note in qns.notes:
        drums_only &= note.is_drum
        if ((self._filters['is_drum'] is None or
             note.is_drum == self._filters['is_drum'])
            and note.velocity > 0):
          vels.add(note.velocity)
          metric_positions.add(note.quantized_start_step % 16)

      if len(vels) < self._filters['min_velocities']:
        beam_metrics.counter(
            'ExtractExamplesDoFn', 'filtered-min-velocities').inc()
        return
      if len(metric_positions) < self._filters['min_metric_positions']:
        beam_metrics.counter(
            'ExtractExamplesDoFn', 'filtered-min-metric-positions').inc()
        return
      if self._filters['drums_only'] and not drums_only:
        beam_metrics.counter(
            'ExtractExamplesDoFn', 'filtered-drums-only').inc()
        return

    beam_metrics.counter('ExtractExamplesDoFn', 'unfiltered-sequences').inc()
    logging.info('Converting %s to tensors', ns.id)
    extracted_examples = self._config.data_converter.to_tensors(ns)
    if not extracted_examples.outputs:
      beam_metrics.counter('ExtractExamplesDoFn', 'empty-extractions').inc()
      return
    beam_metrics.counter('ExtractExamplesDoFn', 'extracted-examples').inc(
        len(extracted_examples.outputs))
    for _, outputs, controls, _ in zip(*extracted_examples):
      if controls.size:
        example_ns = self._config.data_converter.from_tensors(
            [outputs], [controls])[0]
      else:
        example_ns = self._config.data_converter.from_tensors([outputs])[0]
      # Try to re-encode.
      # TODO(adarob): For now we filter and count examples that cannot be
      # re-extracted, but ultimately the converter should filter these or avoid
      # producing them all together.
      reextracted_examples = self._config.data_converter.to_tensors(
          example_ns).inputs
      assert len(reextracted_examples) <= 1
      if not reextracted_examples:
        logging.warning(
            'Extracted example NoteSequence does not reproduce example. '
            'Skipping: %s', example_ns)
        beam_metrics.counter('ExtractExamplesDoFn', 'empty-reextraction').inc()
        continue
      # Extra checks if the code returns multiple segments.
      # TODO(fjord): should probably make this recursive for cases with more
      # than 1 level of hierarchy.
      if isinstance(outputs, list):
        if len(outputs) != len(reextracted_examples[0]):
          logging.warning(
              'Re-extracted example tensor has different number of segments. '
              'ID: %s. original %d, reextracted %d. Skipping.', ns.id,
              len(outputs), len(reextracted_examples[0]))
          beam_metrics.counter(
              'ExtractExamplesDoFn', 'different-reextraction-count').inc()
          continue
        for i in range(len(outputs)):
          if not np.array_equal(reextracted_examples[0][i], outputs[i]):
            logging.warning(
                'Re-extracted example tensor does not equal original example. '
                'ID: %s. Index %d. NoteSequence: %s', ns.id, i, example_ns)
            beam_metrics.counter(
                'ExtractExamplesDoFn', 'different-reextraction').inc()
      yield example_ns, ns.id

  def process(self, seq):
    split_seqs = split_note_sequence_on_time_changes(seq)
    if len(split_seqs) > 100:
      beam_metrics.counter(
          'ExtractExamplesDoFn', 'filtered-too-many-sequence-time-splits').inc()
      return
    beam_metrics.counter(
        'ExtractExamplesDoFn', 'sequence-time-splits').inc(len(split_seqs))
    for ns in split_seqs:
      for res in self._process_ns(ns):
        beam_metrics.counter('ExtractExamplesDoFn', 'all-examples').inc()
        yield res


def combine_matching_seqs(ns_ids):
  ns, ids = ns_ids
  beam_metrics.counter('ExtractExamplesDoFn', 'unique-examples').inc()
  ns.id = ','.join(ids)
  return ns


def run_pipeline(input_tfrecord, output_tfrecord, output_shards, config,
                 filters, pipeline_options):
  with beam.Pipeline(options=pipeline_options) as p:
    _ = (
        p
        | 'read_input' >> beam.io.ReadFromTFRecord(
            input_tfrecord, coder=beam.coders.ProtoCoder(note_seq.NoteSequence))
        | 'extract_sequences' >> beam.ParDo(
            ExtractExamplesDoFn(config, filters))
        | 'group' >> beam.GroupByKey()
        | 'uniquify' >> beam.Map(combine_matching_seqs)
        | 'shuffle' >> beam.Reshuffle()
        | 'write' >> beam.io.WriteToTFRecord(
            output_tfrecord,
            num_shards=output_shards,
            coder=beam.coders.ProtoCoder(note_seq.NoteSequence)))


def main(_):
  flags.mark_flags_as_required(['input_tfrecord', 'output_tfrecord', 'config'])

  filters = None if not FLAGS.enable_filtering else {  # pylint: disable=g-long-ternary
      'max_total_time': FLAGS.max_total_time,
      'max_num_notes': FLAGS.max_num_notes,
      'min_velocities': FLAGS.min_velocities,
      'min_metric_positions': FLAGS.min_metric_positions,
      'is_drum': FLAGS.is_drum,
      'drums_only': FLAGS.drums_only,
  }
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  run_pipeline(FLAGS.input_tfrecord, FLAGS.output_tfrecord, FLAGS.output_shards,
               FLAGS.config, filters, pipeline_options)


if __name__ == '__main__':
  app.run(main)
