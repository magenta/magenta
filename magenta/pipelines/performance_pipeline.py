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
"""Pipeline to create Performance dataset."""

from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
from note_seq import MetricPerformance
from note_seq import Performance
from note_seq import sequences_lib
from note_seq.performance_lib import BasePerformance
from note_seq.performance_lib import NotePerformance
from note_seq.performance_lib import TooManyDurationStepsError
from note_seq.performance_lib import TooManyTimeShiftStepsError
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf


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
        input_type=BasePerformance,
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
      control_sequence = list(zip(*control_sequences))
      if self._optional_conditioning:
        # Create two copies, one with and one without conditioning.
        # pylint: disable=g-complex-comprehension
        encoded = [
            self._encoder_decoder.encode(
                list(zip([disable] * len(control_sequence), control_sequence)),
                performance) for disable in [False, True]
        ]
        # pylint: enable=g-complex-comprehension
      else:
        encoded = [self._encoder_decoder.encode(
            control_sequence, performance)]
    else:
      # Encode unconditional.
      encoded = [self._encoder_decoder.encode(performance)]
    return [pipelines_common.make_sequence_example(*enc) for enc in encoded]


class PerformanceExtractor(pipeline.Pipeline):
  """Extracts polyphonic tracks from a quantized NoteSequence."""

  def __init__(self, min_events, max_events, num_velocity_bins,
               note_performance, name=None):
    super(PerformanceExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=BasePerformance,
        name=name)
    self._min_events = min_events
    self._max_events = max_events
    self._num_velocity_bins = num_velocity_bins
    self._note_performance = note_performance

  def transform(self, quantized_sequence):
    performances, stats = extract_performances(
        quantized_sequence,
        min_events_discard=self._min_events,
        max_events_truncate=self._max_events,
        num_velocity_bins=self._num_velocity_bins,
        note_performance=self._note_performance)
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
  transposition_range = list(range(-3, 4))

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
        note_performance=config.note_performance,
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


def extract_performances(
    quantized_sequence, start_step=0, min_events_discard=None,
    max_events_truncate=None, max_steps_truncate=None, num_velocity_bins=0,
    split_instruments=False, note_performance=False):
  """Extracts one or more performances from the given quantized NoteSequence.

  Args:
    quantized_sequence: A quantized NoteSequence.
    start_step: Start extracting a sequence at this time step.
    min_events_discard: Minimum length of tracks in events. Shorter tracks are
        discarded.
    max_events_truncate: Maximum length of tracks in events. Longer tracks are
        truncated.
    max_steps_truncate: Maximum length of tracks in quantized time steps. Longer
        tracks are truncated.
    num_velocity_bins: Number of velocity bins to use. If 0, velocity events
        will not be included at all.
    split_instruments: If True, will extract a performance for each instrument.
        Otherwise, will extract a single performance.
    note_performance: If True, will create a NotePerformance object. If
        False, will create either a MetricPerformance or Performance based on
        how the sequence was quantized.

  Returns:
    performances: A python list of Performance or MetricPerformance (if
        `quantized_sequence` is quantized relative to meter) instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  """
  sequences_lib.assert_is_quantized_sequence(quantized_sequence)

  # pylint: disable=g-complex-comprehension
  stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
               ['performances_discarded_too_short',
                'performances_truncated', 'performances_truncated_timewise',
                'performances_discarded_more_than_1_program',
                'performance_discarded_too_many_time_shift_steps',
                'performance_discarded_too_many_duration_steps'])
  # pylint: enable=g-complex-comprehension

  if sequences_lib.is_absolute_quantized_sequence(quantized_sequence):
    steps_per_second = quantized_sequence.quantization_info.steps_per_second
    # Create a histogram measuring lengths in seconds.
    stats['performance_lengths_in_seconds'] = statistics.Histogram(
        'performance_lengths_in_seconds',
        [5, 10, 20, 30, 40, 60, 120])
  else:
    steps_per_bar = sequences_lib.steps_per_bar_in_quantized_sequence(
        quantized_sequence)
    # Create a histogram measuring lengths in bars.
    stats['performance_lengths_in_bars'] = statistics.Histogram(
        'performance_lengths_in_bars',
        [1, 10, 20, 30, 40, 50, 100, 200, 500])

  if split_instruments:
    instruments = set(note.instrument for note in quantized_sequence.notes)
  else:
    instruments = set([None])
    # Allow only 1 program.
    programs = set()
    for note in quantized_sequence.notes:
      programs.add(note.program)
    if len(programs) > 1:
      stats['performances_discarded_more_than_1_program'].increment()
      return [], list(stats.values())

  performances = []

  for instrument in instruments:
    # Translate the quantized sequence into a Performance.
    if note_performance:
      try:
        performance = NotePerformance(
            quantized_sequence, start_step=start_step,
            num_velocity_bins=num_velocity_bins, instrument=instrument)
      except TooManyTimeShiftStepsError:
        stats['performance_discarded_too_many_time_shift_steps'].increment()
        continue
      except TooManyDurationStepsError:
        stats['performance_discarded_too_many_duration_steps'].increment()
        continue
    elif sequences_lib.is_absolute_quantized_sequence(quantized_sequence):
      performance = Performance(quantized_sequence, start_step=start_step,
                                num_velocity_bins=num_velocity_bins,
                                instrument=instrument)
    else:
      performance = MetricPerformance(quantized_sequence, start_step=start_step,
                                      num_velocity_bins=num_velocity_bins,
                                      instrument=instrument)

    if (max_steps_truncate is not None and
        performance.num_steps > max_steps_truncate):
      performance.set_length(max_steps_truncate)
      stats['performances_truncated_timewise'].increment()

    if (max_events_truncate is not None and
        len(performance) > max_events_truncate):
      performance.truncate(max_events_truncate)
      stats['performances_truncated'].increment()

    if min_events_discard is not None and len(performance) < min_events_discard:
      stats['performances_discarded_too_short'].increment()
    else:
      performances.append(performance)
      if sequences_lib.is_absolute_quantized_sequence(quantized_sequence):
        stats['performance_lengths_in_seconds'].increment(
            performance.num_steps // steps_per_second)
      else:
        stats['performance_lengths_in_bars'].increment(
            performance.num_steps // steps_per_bar)

  return performances, list(stats.values())
