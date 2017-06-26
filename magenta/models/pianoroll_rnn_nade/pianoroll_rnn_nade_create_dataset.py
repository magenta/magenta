# Copyright 2017 Google Inc. All Rights Reserved.
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

This script will extract pianoroll tracks from NoteSequence protos and save
them to TensorFlow's SequenceExample protos for input to the RNN-NADE models.
"""

import os

# internal imports

import tensorflow as tf

from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_model
import magenta.music as mm
from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
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
tf.app.flags.DEFINE_string('config', 'rnn-nade',
                           'Which config to use.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


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


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  pipeline_instance = get_pipeline(
      min_steps=80,  # 5 measures
      max_steps=2048,
      eval_ratio=FLAGS.eval_ratio,
      config=pianoroll_rnn_nade_model.default_configs[FLAGS.config])

  input_dir = os.path.expanduser(FLAGS.input)
  output_dir = os.path.expanduser(FLAGS.output_dir)
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(input_dir, pipeline_instance.input_type),
      output_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
