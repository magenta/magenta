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
"""Pipeline to create Pianoroll dataset."""

from magenta.pipelines import dag_pipeline
from magenta.pipelines import event_sequence_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
from note_seq import PianorollSequence
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2


class PianorollSequenceExtractor(pipeline.Pipeline):
  """Extracts pianoroll tracks from a quantized NoteSequence."""

  def __init__(self, min_steps, max_steps, name=None):
    super(PianorollSequenceExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=PianorollSequence,
        name=name)
    self._min_steps = min_steps
    self._max_steps = max_steps

  def transform(self, quantized_sequence):
    pianoroll_seqs, stats = extract_pianoroll_sequences(
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
  transposition_range = list(range(-4, 5))

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
    encoder_pipeline = event_sequence_pipeline.EncoderPipeline(
        PianorollSequence,
        config.encoder_decoder,
        name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + '_pianoroll_tracks']
    dag[quantizer] = time_change_splitter
    dag[transposition_pipeline] = quantizer
    dag[pianoroll_extractor] = transposition_pipeline
    dag[encoder_pipeline] = pianoroll_extractor
    dag[dag_pipeline.DagOutput(mode + '_pianoroll_tracks')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)


def extract_pianoroll_sequences(
    quantized_sequence, start_step=0, min_steps_discard=None,
    max_steps_discard=None, max_steps_truncate=None):
  """Extracts a polyphonic track from the given quantized NoteSequence.

  Currently, this extracts only one pianoroll from a given track.

  Args:
    quantized_sequence: A quantized NoteSequence.
    start_step: Start extracting a sequence at this time step. Assumed
        to be the beginning of a bar.
    min_steps_discard: Minimum length of tracks in steps. Shorter tracks are
        discarded.
    max_steps_discard: Maximum length of tracks in steps. Longer tracks are
        discarded. Mutually exclusive with `max_steps_truncate`.
    max_steps_truncate: Maximum length of tracks in steps. Longer tracks are
        truncated. Mutually exclusive with `max_steps_discard`.

  Returns:
    pianoroll_seqs: A python list of PianorollSequence instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.

  Raises:
    ValueError: If both `max_steps_discard` and `max_steps_truncate` are
        specified.
  """

  if (max_steps_discard, max_steps_truncate).count(None) == 0:
    raise ValueError(
        'Only one of `max_steps_discard` and `max_steps_truncate` can be '
        'specified.')
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

  # pylint: disable=g-complex-comprehension
  stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
               ['pianoroll_tracks_truncated_too_long',
                'pianoroll_tracks_discarded_too_short',
                'pianoroll_tracks_discarded_too_long',
                'pianoroll_tracks_discarded_more_than_1_program'])
  # pylint: enable=g-complex-comprehension

  steps_per_bar = sequences_lib.steps_per_bar_in_quantized_sequence(
      quantized_sequence)

  # Create a histogram measuring lengths (in bars not steps).
  stats['pianoroll_track_lengths_in_bars'] = statistics.Histogram(
      'pianoroll_track_lengths_in_bars',
      [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, 1000])

  # Allow only 1 program.
  programs = set()
  for note in quantized_sequence.notes:
    programs.add(note.program)
  if len(programs) > 1:
    stats['pianoroll_tracks_discarded_more_than_1_program'].increment()
    return [], list(stats.values())

  # Translate the quantized sequence into a PianorollSequence.
  pianoroll_seq = PianorollSequence(quantized_sequence=quantized_sequence,
                                    start_step=start_step)

  pianoroll_seqs = []
  num_steps = pianoroll_seq.num_steps

  if min_steps_discard is not None and num_steps < min_steps_discard:
    stats['pianoroll_tracks_discarded_too_short'].increment()
  elif max_steps_discard is not None and num_steps > max_steps_discard:
    stats['pianoroll_tracks_discarded_too_long'].increment()
  else:
    if max_steps_truncate is not None and num_steps > max_steps_truncate:
      stats['pianoroll_tracks_truncated_too_long'].increment()
      pianoroll_seq.set_length(max_steps_truncate)
    pianoroll_seqs.append(pianoroll_seq)
    stats['pianoroll_track_lengths_in_bars'].increment(
        num_steps // steps_per_bar)
  return pianoroll_seqs, list(stats.values())
