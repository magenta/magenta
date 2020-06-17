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

"""Pipeline to create ImprovRNN dataset."""

from magenta.pipelines import dag_pipeline
from magenta.pipelines import lead_sheet_pipelines
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
import note_seq
import tensorflow.compat.v1 as tf


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
        input_type=note_seq.LeadSheet,
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
    except note_seq.ChordEncodingError as e:
      tf.logging.warning('Skipped lead sheet: %s', e)
      encoded = []
      stats = [statistics.Counter('chord_encoding_exception', 1)]
    except note_seq.ChordSymbolError as e:
      tf.logging.warning('Skipped lead sheet: %s', e)
      encoded = []
      stats = [statistics.Counter('chord_symbol_exception', 1)]
    self._set_stats(stats)
    return [pipelines_common.make_sequence_example(*enc) for enc in encoded]

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
      note_seq.NoteSequence,
      ['eval_lead_sheets', 'training_lead_sheets'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(note_seq.NoteSequence)}

  for mode in ['eval', 'training']:
    time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
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
