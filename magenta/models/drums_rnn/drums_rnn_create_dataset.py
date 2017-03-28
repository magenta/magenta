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

This script will extract drum tracks from NoteSequence protos and save them to
TensorFlow's SequenceExample protos for input to the drums RNN models.
"""

import os

# internal imports
import tensorflow as tf
import magenta

from magenta.models.drums_rnn import drums_rnn_config_flags

from magenta.music import encoder_decoder
from magenta.pipelines import dag_pipeline
from magenta.pipelines import drum_pipelines
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
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


def get_pipeline(config, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: A DrumsRnnConfig object.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_drum_tracks', 'training_drum_tracks'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    time_change_splitter = pipelines_common.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    quantizer = pipelines_common.Quantizer(
        steps_per_quarter=config.steps_per_quarter, name='Quantizer_' + mode)
    drums_extractor = drum_pipelines.DrumsExtractor(
        min_bars=7, max_steps=512, gap_bars=1.0, name='DrumsExtractor_' + mode)
    encoder_pipeline = encoder_decoder.EncoderPipeline(
        magenta.music.DrumTrack, config.encoder_decoder,
        name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + '_drum_tracks']
    dag[quantizer] = time_change_splitter
    dag[drums_extractor] = quantizer
    dag[encoder_pipeline] = drums_extractor
    dag[dag_pipeline.DagOutput(mode + '_drum_tracks')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  config = drums_rnn_config_flags.config_from_flags()
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
