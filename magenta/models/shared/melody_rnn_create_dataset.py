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

This script will extract melodies from NoteSequence protos and save them to
TensorFlow's SequenceExample protos for input to the melody RNN models.
"""

import os

# internal imports
import tensorflow as tf
import magenta

from magenta.pipelines import dag_pipeline
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
tf.app.flags.DEFINE_float('eval_ratio', 0.0,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


class EncoderPipeline(pipeline.Pipeline):
  """A Module that converts monophonic melodies to a model specific encoding."""

  def __init__(self, melody_encoder_decoder):
    """Constructs a EncoderPipeline.

    A magenta.music.MelodyEncoderDecoder is needed to provide the
    `encode` function.

    Args:
      melody_encoder_decoder: A magenta.music.MelodyEncoderDecoder object.
    """
    super(EncoderPipeline, self).__init__(
        input_type=magenta.music.Melody,
        output_type=tf.train.SequenceExample)
    self.melody_encoder_decoder = melody_encoder_decoder

  def transform(self, melody):
    encoded = self.melody_encoder_decoder.squash_and_encode(melody)
    return [encoded]

  def get_stats(self):
    return {}


def get_pipeline(melody_encoder_decoder):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    melody_encoder_decoder: A magenta.music.MelodyEncoderDecoder object.

  Returns:
    A pipeline.Pipeline instance.
  """
  quantizer = pipelines_common.Quantizer(steps_per_quarter=4)
  melody_extractor = pipelines_common.MelodyExtractor(
      min_bars=7, min_unique_pitches=5,
      gap_bars=1.0, ignore_polyphonic_notes=False)
  encoder_pipeline = EncoderPipeline(melody_encoder_decoder)
  partitioner = pipelines_common.RandomPartition(
      tf.train.SequenceExample,
      ['eval_melodies', 'training_melodies'],
      [FLAGS.eval_ratio])

  dag = {quantizer: dag_pipeline.Input(music_pb2.NoteSequence),
         melody_extractor: quantizer,
         encoder_pipeline: melody_extractor,
         partitioner: encoder_pipeline,
         dag_pipeline.Output(): partitioner}
  return dag_pipeline.DAGPipeline(dag)


def run_from_flags(pipeline_instance):
  tf.logging.set_verbosity(FLAGS.log)
  FLAGS.input = os.path.expanduser(FLAGS.input)
  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(FLAGS.input, pipeline_instance.input_type),
      FLAGS.output_dir)
