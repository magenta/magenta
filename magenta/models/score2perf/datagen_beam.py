# pylint: skip-file
# TODO(iansimon): Enable when Apache Beam supports Python 3.
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
"""Beam pipeline to generate examples for a Score2Perf dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import hashlib
import logging
import os
import random

from absl import flags

import apache_beam as beam
from apache_beam import typehints
from apache_beam.metrics import Metrics

import numpy as np
from tensor2tensor.data_generators import generator_utils
import tensorflow as tf

from magenta.music import chord_inference
from magenta.music import melody_inference
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

# TODO(iansimon): this should probably be defined in the problem
SCORE_BPM = 120.0

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'pipeline_options', '',
    'Command line flags to use in constructing the Beam pipeline options.')

# TODO(iansimon): Figure out how to avoid explicitly serializing and
# deserializing NoteSequence protos.


@typehints.with_output_types(typehints.KV[str, str])
class ReadNoteSequencesFromTFRecord(beam.PTransform):
  """Beam PTransform that reads NoteSequence protos from TFRecord."""

  def __init__(self, tfrecord_path):
    super(ReadNoteSequencesFromTFRecord, self).__init__()
    self._tfrecord_path = tfrecord_path

  def expand(self, pcoll):
    pcoll |= beam.io.tfrecordio.ReadFromTFRecord(self._tfrecord_path)
    pcoll |= beam.Map(
        lambda ns_str: (music_pb2.NoteSequence.FromString(ns_str).id, ns_str))
    return pcoll


def select_split(cumulative_splits, kv, unused_num_partitions):
  """Select split for an `(id, _)` tuple using a hash of `id`."""
  key, _ = kv
  m = hashlib.md5(key)
  r = int(m.hexdigest(), 16) / (2 ** (8 * m.digest_size))
  for i, (name, p) in enumerate(cumulative_splits):
    if r < p:
      Metrics.counter('select_split', name).inc()
      return i
  assert False


def filter_invalid_notes(min_pitch, max_pitch, kv):
  """Filter notes with out-of-range pitch from NoteSequence protos."""
  key, ns_str = kv
  ns = music_pb2.NoteSequence.FromString(ns_str)
  valid_notes = [note for note in ns.notes
                 if min_pitch <= note.pitch <= max_pitch]
  if len(valid_notes) < len(ns.notes):
    del ns.notes[:]
    ns.notes.extend(valid_notes)
    Metrics.counter('filter_invalid_notes', 'out_of_range_pitch').inc()
  return key, ns.SerializeToString()


class DataAugmentationError(Exception):
  """Exception to be raised by augmentation functions on known failure."""
  pass


class ExtractExamplesDoFn(beam.DoFn):
  """Extracts Score2Perf examples from NoteSequence protos."""

  def __init__(self, min_hop_size_seconds, max_hop_size_seconds,
               num_replications, encode_performance_fn, encode_score_fns,
               augment_fns, absolute_timing, *unused_args, **unused_kwargs):
    """Initialize an ExtractExamplesDoFn.

    If any of the `encode_score_fns` or `encode_performance_fn` returns an empty
    encoding for a particular example, the example will be discarded.

    Args:
      min_hop_size_seconds: Minimum hop size in seconds at which input
          NoteSequence protos can be split.
      max_hop_size_seconds: Maximum hop size in seconds at which input
          NoteSequence protos can be split. If zero or None, will not split at
          all.
      num_replications: Number of times input NoteSequence protos will be
          replicated prior to splitting.
      encode_performance_fn: Performance encoding function. Will be applied to
          the performance NoteSequence and the resulting encoding will be stored
          as 'targets' in each example.
      encode_score_fns: Optional dictionary of named score encoding functions.
          If provided, each function will be applied to the score NoteSequence
          and the resulting encodings will be stored in each example.
      augment_fns: Optional list of data augmentation functions. If provided,
          each function will be applied to each performance NoteSequence (and
          score, when using scores), creating a separate example per
          augmentation function. Should not modify the NoteSequence.
      absolute_timing: If True, each score will use absolute instead of tempo-
          relative timing. Since chord inference depends on having beats, the
          score will only contain melody.

    Raises:
      ValueError: If the maximum hop size is less than twice the minimum hop
          size.
    """
    if (max_hop_size_seconds and
        max_hop_size_seconds != min_hop_size_seconds and
        max_hop_size_seconds < 2 * min_hop_size_seconds):
      raise ValueError(
          'Maximum hop size must be at least twice minimum hop size.')

    super(ExtractExamplesDoFn, self).__init__(*unused_args, **unused_kwargs)
    self._min_hop_size_seconds = min_hop_size_seconds
    self._max_hop_size_seconds = max_hop_size_seconds
    self._num_replications = num_replications
    self._encode_performance_fn = encode_performance_fn
    self._encode_score_fns = encode_score_fns
    self._augment_fns = augment_fns if augment_fns else [lambda ns: ns]
    self._absolute_timing = absolute_timing

  def process(self, kv):
    # Seed random number generator based on key so that hop times are
    # deterministic.
    key, ns_str = kv
    m = hashlib.md5(key)
    random.seed(int(m.hexdigest(), 16))

    # Deserialize NoteSequence proto.
    ns = music_pb2.NoteSequence.FromString(ns_str)

    # Apply sustain pedal.
    ns = sequences_lib.apply_sustain_control_changes(ns)

    # Remove control changes as there are potentially a lot of them and they are
    # no longer needed.
    del ns.control_changes[:]

    if (self._min_hop_size_seconds and
        ns.total_time < self._min_hop_size_seconds):
      Metrics.counter('extract_examples', 'sequence_too_short').inc()
      return

    sequences = []
    for _ in range(self._num_replications):
      if self._max_hop_size_seconds:
        if self._max_hop_size_seconds == self._min_hop_size_seconds:
          # Split using fixed hop size.
          sequences += sequences_lib.split_note_sequence(
              ns, self._max_hop_size_seconds)
        else:
          # Sample random hop positions such that each segment size is within
          # the specified range.
          hop_times = [0.0]
          while hop_times[-1] <= ns.total_time - self._min_hop_size_seconds:
            if hop_times[-1] + self._max_hop_size_seconds < ns.total_time:
              # It's important that we get a valid hop size here, since the
              # remainder of the sequence is too long.
              max_offset = min(
                  self._max_hop_size_seconds,
                  ns.total_time - self._min_hop_size_seconds - hop_times[-1])
            else:
              # It's okay if the next hop time is invalid (in which case we'll
              # just stop).
              max_offset = self._max_hop_size_seconds
            offset = random.uniform(self._min_hop_size_seconds, max_offset)
            hop_times.append(hop_times[-1] + offset)
          # Split at the chosen hop times (ignoring zero and the final invalid
          # time).
          sequences += sequences_lib.split_note_sequence(ns, hop_times[1:-1])
      else:
        sequences += [ns]

    for performance_sequence in sequences:
      if self._encode_score_fns:
        # We need to extract a score.
        if not self._absolute_timing:
          # Beats are required to extract a score with metric timing.
          beats = [
              ta for ta in performance_sequence.text_annotations
              if (ta.annotation_type ==
                  music_pb2.NoteSequence.TextAnnotation.BEAT)
              and ta.time <= performance_sequence.total_time
          ]
          if len(beats) < 2:
            Metrics.counter('extract_examples', 'not_enough_beats').inc()
            continue

          # Ensure the sequence starts and ends on a beat.
          performance_sequence = sequences_lib.extract_subsequence(
              performance_sequence,
              start_time=min(beat.time for beat in beats),
              end_time=max(beat.time for beat in beats)
          )

          # Infer beat-aligned chords (only for relative timing).
          try:
            chord_inference.infer_chords_for_sequence(
                performance_sequence,
                chord_change_prob=0.25,
                chord_note_concentration=50.0,
                add_key_signatures=True)
          except chord_inference.ChordInferenceException:
            Metrics.counter('extract_examples', 'chord_inference_failed').inc()
            continue

        # Infer melody regardless of relative/absolute timing.
        try:
          melody_instrument = melody_inference.infer_melody_for_sequence(
              performance_sequence,
              melody_interval_scale=2.0,
              rest_prob=0.1,
              instantaneous_non_max_pitch_prob=1e-15,
              instantaneous_non_empty_rest_prob=0.0,
              instantaneous_missing_pitch_prob=1e-15)
        except melody_inference.MelodyInferenceException:
          Metrics.counter('extract_examples', 'melody_inference_failed').inc()
          continue

        if not self._absolute_timing:
          # Now rectify detected beats to occur at fixed tempo.
          # TODO(iansimon): also include the alignment
          score_sequence, unused_alignment = sequences_lib.rectify_beats(
              performance_sequence, beats_per_minute=SCORE_BPM)
        else:
          # Score uses same timing as performance.
          score_sequence = copy.deepcopy(performance_sequence)

        # Remove melody notes from performance.
        performance_notes = []
        for note in performance_sequence.notes:
          if note.instrument != melody_instrument:
            performance_notes.append(note)
        del performance_sequence.notes[:]
        performance_sequence.notes.extend(performance_notes)

        # Remove non-melody notes from score.
        score_notes = []
        for note in score_sequence.notes:
          if note.instrument == melody_instrument:
            score_notes.append(note)
        del score_sequence.notes[:]
        score_sequence.notes.extend(score_notes)

        # Remove key signatures and beat/chord annotations from performance.
        del performance_sequence.key_signatures[:]
        del performance_sequence.text_annotations[:]

        Metrics.counter('extract_examples', 'extracted_score').inc()

      for augment_fn in self._augment_fns:
        # Augment and encode the performance.
        try:
          augmented_performance_sequence = augment_fn(performance_sequence)
        except DataAugmentationError:
          Metrics.counter(
              'extract_examples', 'augment_performance_failed').inc()
          continue
        example_dict = {
            'targets': self._encode_performance_fn(
                augmented_performance_sequence)
        }
        if not example_dict['targets']:
          Metrics.counter('extract_examples', 'skipped_empty_targets').inc()
          continue

        if self._encode_score_fns:
          # Augment the extracted score.
          try:
            augmented_score_sequence = augment_fn(score_sequence)
          except DataAugmentationError:
            Metrics.counter('extract_examples', 'augment_score_failed').inc()
            continue

          # Apply all score encoding functions.
          skip = False
          for name, encode_score_fn in self._encode_score_fns.items():
            example_dict[name] = encode_score_fn(augmented_score_sequence)
            if not example_dict[name]:
              Metrics.counter('extract_examples',
                              'skipped_empty_%s' % name).inc()
              skip = True
              break
          if skip:
            continue

        Metrics.counter('extract_examples', 'encoded_example').inc()
        Metrics.distribution(
            'extract_examples', 'performance_length_in_seconds').update(
                int(augmented_performance_sequence.total_time))

        yield generator_utils.to_example(example_dict)


def generate_examples(input_transform, output_dir, problem_name, splits,
                      min_hop_size_seconds, max_hop_size_seconds,
                      num_replications, min_pitch, max_pitch,
                      encode_performance_fn, encode_score_fns=None,
                      augment_fns=None, absolute_timing=False):
  """Generate data for a Score2Perf problem.

  Args:
    input_transform: The input PTransform object that reads input NoteSequence
        protos, or dictionary mapping split names to such PTransform objects.
        Should produce `(id, NoteSequence)` tuples.
    output_dir: The directory to write the resulting TFRecord file containing
        examples.
    problem_name: Name of the Tensor2Tensor problem, used as a base filename
        for generated data.
    splits: A dictionary of split names and their probabilities. Probabilites
        should add up to 1. If `input_filename` is a dictionary, this argument
        will be ignored.
    min_hop_size_seconds: Minimum hop size in seconds at which input
        NoteSequence protos can be split. Can also be a dictionary mapping split
        name to minimum hop size.
    max_hop_size_seconds: Maximum hop size in seconds at which input
        NoteSequence protos can be split. If zero or None, will not split at
        all. Can also be a dictionary mapping split name to maximum hop size.
    num_replications: Number of times input NoteSequence protos will be
        replicated prior to splitting.
    min_pitch: Minimum MIDI pitch value; notes with lower pitch will be dropped.
    max_pitch: Maximum MIDI pitch value; notes with greater pitch will be
        dropped.
    encode_performance_fn: Required performance encoding function.
    encode_score_fns: Optional dictionary of named score encoding functions.
    augment_fns: Optional list of data augmentation functions. Only applied in
        the 'train' split.
    absolute_timing: If True, each score will use absolute instead of tempo-
        relative timing. Since chord inference depends on having beats, the
        score will only contain melody.

  Raises:
    ValueError: If split probabilities do not add up to 1, or if splits are not
        provided but `input_filename` is not a dictionary.
  """
  # Make sure Beam's log messages are not filtered.
  logging.getLogger().setLevel(logging.INFO)

  if isinstance(input_transform, dict):
    split_names = input_transform.keys()
  else:
    if not splits:
      raise ValueError(
          'Split probabilities must be provided if input is not presplit.')
    split_names, split_probabilities = zip(*splits.items())
    cumulative_splits = zip(split_names, np.cumsum(split_probabilities))
    if cumulative_splits[-1][1] != 1.0:
      raise ValueError('Split probabilities must sum to 1; got %f' %
                       cumulative_splits[-1][1])

  # Check for existence of prior outputs. Since the number of shards may be
  # different, the prior outputs will not necessarily be overwritten and must
  # be deleted explicitly.
  output_filenames = [
      os.path.join(output_dir, '%s-%s.tfrecord' % (problem_name, split_name))
      for split_name in split_names
  ]
  for split_name, output_filename in zip(split_names, output_filenames):
    existing_output_filenames = tf.gfile.Glob(output_filename + '*')
    if existing_output_filenames:
      tf.logging.info(
          'Data files already exist for split %s in problem %s, deleting.',
          split_name, problem_name)
      for filename in existing_output_filenames:
        tf.gfile.Remove(filename)

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(','))

  with beam.Pipeline(options=pipeline_options) as p:
    if isinstance(input_transform, dict):
      # Input data is already partitioned into splits.
      split_partitions = [
          p | 'input_transform_%s' % split_name >> input_transform[split_name]
          for split_name in split_names
      ]
    else:
      # Read using a single PTransform.
      p |= 'input_transform' >> input_transform
      split_partitions = p | 'partition' >> beam.Partition(
          functools.partial(select_split, cumulative_splits),
          len(cumulative_splits))

    for split_name, output_filename, s in zip(
        split_names, output_filenames, split_partitions):
      if isinstance(min_hop_size_seconds, dict):
        min_hop = min_hop_size_seconds[split_name]
      else:
        min_hop = min_hop_size_seconds
      if isinstance(max_hop_size_seconds, dict):
        max_hop = max_hop_size_seconds[split_name]
      else:
        max_hop = max_hop_size_seconds
      s |= 'preshuffle_%s' % split_name >> beam.Reshuffle()
      s |= 'filter_invalid_notes_%s' % split_name >> beam.Map(
          functools.partial(filter_invalid_notes, min_pitch, max_pitch))
      s |= 'extract_examples_%s' % split_name >> beam.ParDo(
          ExtractExamplesDoFn(
              min_hop, max_hop,
              num_replications if split_name == 'train' else 1,
              encode_performance_fn, encode_score_fns,
              augment_fns if split_name == 'train' else None, absolute_timing))
      s |= 'shuffle_%s' % split_name >> beam.Reshuffle()
      s |= 'write_%s' % split_name >> beam.io.WriteToTFRecord(
          output_filename, coder=beam.coders.ProtoCoder(tf.train.Example))
