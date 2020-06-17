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

"""Pipeline to create DrumsRNN dataset."""

from magenta.pipelines import dag_pipeline
from magenta.pipelines import drum_pipelines
from magenta.pipelines import event_sequence_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipelines_common
import note_seq


def get_pipeline(config, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: A DrumsRnnConfig object.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  partitioner = pipelines_common.RandomPartition(
      note_seq.NoteSequence,
      ['eval_drum_tracks', 'training_drum_tracks'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(note_seq.NoteSequence)}

  for mode in ['eval', 'training']:
    time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_quarter=config.steps_per_quarter, name='Quantizer_' + mode)
    drums_extractor = drum_pipelines.DrumsExtractor(
        min_bars=7, max_steps=512, gap_bars=1.0, name='DrumsExtractor_' + mode)
    encoder_pipeline = event_sequence_pipeline.EncoderPipeline(
        note_seq.DrumTrack,
        config.encoder_decoder,
        name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + '_drum_tracks']
    dag[quantizer] = time_change_splitter
    dag[drums_extractor] = quantizer
    dag[encoder_pipeline] = drums_extractor
    dag[dag_pipeline.DagOutput(mode + '_drum_tracks')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)
