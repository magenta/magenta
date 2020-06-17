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

"""Pipeline to create MelodyRNN dataset."""

from magenta.pipelines import dag_pipeline
from magenta.pipelines import melody_pipelines
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
import note_seq
import tensorflow.compat.v1 as tf


class EncoderPipeline(pipeline.Pipeline):
  """A Module that converts monophonic melodies to a model specific encoding."""

  def __init__(self, config, name):
    """Constructs an EncoderPipeline.

    Args:
      config: A MelodyRnnConfig that specifies the encoder/decoder, pitch range,
          and what key to transpose into.
      name: A unique pipeline name.
    """
    super(EncoderPipeline, self).__init__(
        input_type=note_seq.Melody,
        output_type=tf.train.SequenceExample,
        name=name)
    self._melody_encoder_decoder = config.encoder_decoder
    self._min_note = config.min_note
    self._max_note = config.max_note
    self._transpose_to_key = config.transpose_to_key

  def transform(self, melody):
    melody.squash(
        self._min_note,
        self._max_note,
        self._transpose_to_key)
    encoded = pipelines_common.make_sequence_example(
        *self._melody_encoder_decoder.encode(melody))
    return [encoded]


def get_pipeline(config, transposition_range=(0,), eval_ratio=0.0):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: A MelodyRnnConfig object.
    transposition_range: Collection of integer pitch steps to transpose.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  partitioner = pipelines_common.RandomPartition(
      note_seq.NoteSequence,
      ['eval_melodies', 'training_melodies'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(note_seq.NoteSequence)}

  for mode in ['eval', 'training']:
    time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range, name='TranspositionPipeline_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_quarter=config.steps_per_quarter, name='Quantizer_' + mode)
    melody_extractor = melody_pipelines.MelodyExtractor(
        min_bars=7, max_steps=512, min_unique_pitches=5,
        gap_bars=1.0, ignore_polyphonic_notes=False,
        name='MelodyExtractor_' + mode)
    encoder_pipeline = EncoderPipeline(config, name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + '_melodies']
    dag[quantizer] = time_change_splitter
    dag[transposition_pipeline] = quantizer
    dag[melody_extractor] = transposition_pipeline
    dag[encoder_pipeline] = melody_extractor
    dag[dag_pipeline.DagOutput(mode + '_melodies')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)
