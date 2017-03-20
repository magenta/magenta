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
"""Create a dataset of SequenceExamples from NoteSequence protos.

This script will extract melodies and chords from NoteSequence protos and save
them to TensorFlow's SequenceExample protos for input to the improv RNN models.
"""

import os

# internal imports
import tensorflow as tf
import magenta

from magenta.models.improv_rnn import improv_rnn_config_flags

from magenta.pipelines import dag_pipeline
from magenta.pipelines import lead_sheet_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics

from magenta.protobuf import music_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Directory to write training and eval TFRecord '
                           'files. The TFRecord files are populated with '
                           'SequenceExample protos.')
tf.app.flags.DEFINE_float('eval_ratio', 0.1,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


class EncoderPipeline(pipeline.Pipeline):
  """A Module that converts lead sheets to a model specific encoding."""

  def __init__(self, config, name):
    """Constructs an EncoderPipeline.

    Args:
      config: An ImprovRnnConfig that specifies the encoder/decoder,
          pitch range, and transposition behavior.
      name: A unique pipeline name.
    """
    super(EncoderPipeline, self).__init__(
        input_type=magenta.music.LeadSheet,
        output_type=tf.train.SequenceExample,
        name=name)
    self._conditional_encoder_decoder = config.encoder_decoder
    self._min_note = config.min_note
    self._max_note = config.max_note
    self._transpose_to_key = config.transpose_to_key

  def transform(self, lead_sheet):
    lead_sheet.squash(
        self._min_note,
        self._max_note,
        self._transpose_to_key)
    try:
      encoded = [self._conditional_encoder_decoder.encode(
          lead_sheet.chords, lead_sheet.melody)]
      stats = []
    except magenta.music.ChordEncodingException as e:
      tf.logging.warning('Skipped lead sheet: %s', e)
      encoded = []
      stats = [statistics.Counter('chord_encoding_exception', 1)]
    except magenta.music.ChordSymbolException as e:
      tf.logging.warning('Skipped lead sheet: %s', e)
      encoded = []
      stats = [statistics.Counter('chord_symbol_exception', 1)]
    self._set_stats(stats)
    return encoded

  def get_stats(self):
    return {}


def get_pipeline(config, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: An ImprovRnnConfig object.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  all_transpositions = config.transpose_to_key is None
  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_lead_sheets', 'training_lead_sheets'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    time_change_splitter = pipelines_common.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    quantizer = pipelines_common.Quantizer(
        steps_per_quarter=config.steps_per_quarter, name='Quantizer_' + mode)
    lead_sheet_extractor = lead_sheet_pipelines.LeadSheetExtractor(
        min_bars=7, max_steps=512, min_unique_pitches=3, gap_bars=1.0,
        ignore_polyphonic_notes=False, all_transpositions=all_transpositions,
        name='LeadSheetExtractor_' + mode)
    encoder_pipeline = EncoderPipeline(config, name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + '_lead_sheets']
    dag[quantizer] = time_change_splitter
    dag[lead_sheet_extractor] = quantizer
    dag[encoder_pipeline] = lead_sheet_extractor
    dag[dag_pipeline.DagOutput(mode + '_lead_sheets')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  config = improv_rnn_config_flags.config_from_flags()
  pipeline_instance = get_pipeline(
      config, FLAGS.eval_ratio)

  FLAGS.input = os.path.expanduser(FLAGS.input)
  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(FLAGS.input, pipeline_instance.input_type),
      FLAGS.output_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
