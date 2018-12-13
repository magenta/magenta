# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Pipeline to create Pianoroll RNN-NADE dataset."""

import magenta.music as mm
from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.protobuf import music_pb2


class PianorollSequenceExtractor(pipeline.Pipeline):
  """Extracts pianoroll tracks from a quantized NoteSequence."""

  def __init__(self, min_steps, max_steps, name=None):
    super(PianorollSequenceExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=mm.PianorollSequence,
        name=name)
    self._min_steps = min_steps
    self._max_steps = max_steps

  def transform(self, quantized_sequence):
    pianoroll_seqs, stats = mm.extract_pianoroll_sequences(
        quantized_sequence,
        min_steps_discard=self._min_steps,
        max_steps_truncate=self._max_steps)
    self._set_stats(stats)
    return pianoroll_seqs


def get_pipeline(config, min_steps, max_steps, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: An EventSequenceRnnConfig.
    min_steps: Minimum number of steps for an extracted sequence.
    max_steps: Maximum number of steps for an extracted sequence.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  # Transpose up to a major third in either direction.
  transposition_range = range(-4, 5)

  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_pianoroll_tracks', 'training_pianoroll_tracks'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_quarter=config.steps_per_quarter, name='Quantizer_' + mode)
    transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range, name='TranspositionPipeline_' + mode)
    pianoroll_extractor = PianorollSequenceExtractor(
        min_steps=min_steps, max_steps=max_steps,
        name='PianorollExtractor_' + mode)
    encoder_pipeline = mm.EncoderPipeline(
        mm.PianorollSequence, config.encoder_decoder,
        name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + '_pianoroll_tracks']
    dag[quantizer] = time_change_splitter
    dag[transposition_pipeline] = quantizer
    dag[pianoroll_extractor] = transposition_pipeline
    dag[encoder_pipeline] = pianoroll_extractor
    dag[dag_pipeline.DagOutput(mode + '_pianoroll_tracks')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)
