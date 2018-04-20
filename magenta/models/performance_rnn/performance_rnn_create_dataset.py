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

This script will extract polyphonic tracks from NoteSequence protos and save
them to TensorFlow's SequenceExample protos for input to the performance RNN
models. It will apply data augmentation, stretching and transposing each
NoteSequence within a limited range.
"""

import os

# internal imports

import tensorflow as tf
import magenta

from magenta.models.performance_rnn import performance_model

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
tf.app.flags.DEFINE_string('config', 'performance', 'The config to use')
tf.app.flags.DEFINE_float('eval_ratio', 0.1,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


class EncoderPipeline(pipeline.Pipeline):
  """A Pipeline that converts performances to a model specific encoding."""

  def __init__(self, config, name):
    """Constructs an EncoderPipeline.

    Args:
      config: A PerformanceRnnConfig that specifies the encoder/decoder and
          note density conditioning behavior.
      name: A unique pipeline name.
    """
    super(EncoderPipeline, self).__init__(
        input_type=magenta.music.Performance,
        output_type=tf.train.SequenceExample,
        name=name)
    self._encoder_decoder = config.encoder_decoder
    self._control_signals = config.control_signals
    self._optional_conditioning = config.optional_conditioning

  def transform(self, performance):
    if self._control_signals:
      # Encode conditional on control signals.
      control_sequences = []
      for control in self._control_signals:
        control_sequences.append(control.extract(performance))
      control_sequence = zip(*control_sequences)
      if self._optional_conditioning:
        # Create two copies, one with and one without conditioning.
        encoded = [
            self._encoder_decoder.encode(
                zip([disable] * len(control_sequence), control_sequence),
                performance)
            for disable in [False, True]]
      else:
        encoded = [self._encoder_decoder.encode(
            control_sequence, performance)]
    else:
      # Encode unconditional.
      encoded = [self._encoder_decoder.encode(performance)]
    return encoded


class PerformanceExtractor(pipeline.Pipeline):
  """Extracts polyphonic tracks from a quantized NoteSequence."""

  def __init__(self, min_events, max_events, num_velocity_bins, name=None):
    super(PerformanceExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=magenta.music.Performance,
        name=name)
    self._min_events = min_events
    self._max_events = max_events
    self._num_velocity_bins = num_velocity_bins

  def transform(self, quantized_sequence):
    performances, stats = magenta.music.extract_performances(
        quantized_sequence,
        min_events_discard=self._min_events,
        max_events_truncate=self._max_events,
        num_velocity_bins=self._num_velocity_bins)
    self._set_stats(stats)
    return performances


def get_pipeline(config, min_events, max_events, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: A PerformanceRnnConfig.
    min_events: Minimum number of events for an extracted sequence.
    max_events: Maximum number of events for an extracted sequence.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
  stretch_factors = [0.95, 0.975, 1.0, 1.025, 1.05]

  # Transpose no more than a major third.
  transposition_range = range(-3, 4)

  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_performances', 'training_performances'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    sustain_pipeline = note_sequence_pipelines.SustainPipeline(
        name='SustainPipeline_' + mode)
    stretch_pipeline = note_sequence_pipelines.StretchPipeline(
        stretch_factors if mode == 'training' else [1.0],
        name='StretchPipeline_' + mode)
    splitter = note_sequence_pipelines.Splitter(
        hop_size_seconds=30.0, name='Splitter_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_second=config.steps_per_second, name='Quantizer_' + mode)
    transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range if mode == 'training' else [0],
        name='TranspositionPipeline_' + mode)
    perf_extractor = PerformanceExtractor(
        min_events=min_events, max_events=max_events,
        num_velocity_bins=config.num_velocity_bins,
        name='PerformanceExtractor_' + mode)
    encoder_pipeline = EncoderPipeline(config, name='EncoderPipeline_' + mode)

    dag[sustain_pipeline] = partitioner[mode + '_performances']
    dag[stretch_pipeline] = sustain_pipeline
    dag[splitter] = stretch_pipeline
    dag[quantizer] = splitter
    dag[transposition_pipeline] = quantizer
    dag[perf_extractor] = transposition_pipeline
    dag[encoder_pipeline] = perf_extractor
    dag[dag_pipeline.DagOutput(mode + '_performances')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  pipeline_instance = get_pipeline(
      min_events=32,
      max_events=512,
      eval_ratio=FLAGS.eval_ratio,
      config=performance_model.default_configs[FLAGS.config])

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
