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
"""Defines Module base class and implementations.

Modules are data processing building blocks for creating datasets.
"""


import random

# internal imports
import numpy as np
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import sequences_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


class Quantizer(pipeline.Pipeline):
  """A Module that quantizes NoteSequence data."""

  def __init__(self, steps_per_beat=4):
    super(Quantizer, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=sequences_lib.QuantizedSequence)
    self.steps_per_beat = steps_per_beat

  def transform(self, note_sequence):
    quantized_sequence = sequences_lib.QuantizedSequence()
    try:
      quantized_sequence.from_note_sequence(note_sequence, self.steps_per_beat)
      return [quantized_sequence]
    except sequences_lib.MultipleTimeSignatureException:
      tf.logging.debug('Multiple time signatures found in NoteSequence')
      return []


class MonophonicMelodyExtractor(pipeline.Pipeline):
  """Extracts monophonic melodies from a QuantizedSequence."""

  def __init__(self, min_bars=7, min_unique_pitches=5, gap_bars=1.0,
               ignore_polyphonic_notes=False):
    super(MonophonicMelodyExtractor, self).__init__(
        input_type=sequences_lib.QuantizedSequence,
        output_type=melodies_lib.MonophonicMelody)
    self.min_bars = min_bars
    self.min_unique_pitches = min_unique_pitches
    self.gap_bars = gap_bars
    self.ignore_polyphonic_notes = False
    self.stats = {}

  def transform(self, quantized_sequence):
    melodies, stats = melodies_lib.extract_melodies(
        quantized_sequence,
        min_bars=self.min_bars,
        min_unique_pitches=self.min_unique_pitches,
        gap_bars=self.gap_bars,
        ignore_polyphonic_notes=self.ignore_polyphonic_notes)
    self.stats = stats
    return melodies

  def get_stats(self):
    return self.stats


class RandomPartition(pipeline.Pipeline):
  """Outputs multiple datasets.

  This Pipeline will take a single input feed and randomly partition the inputs
  into multiple output datasets. The probabilities of an input landing in each
  dataset are given by `partition_probabilities`. Use this Pipeline to partition
  previous Pipeline outputs into training and test sets, or training, eval, and
  test sets.
  """

  def __init__(self, type_, partition_names, partition_probabilities):
    super(RandomPartition, self).__init__(
        type_, dict([(name, type_) for name in partition_names]))
    if len(partition_probabilities) != len(partition_names) - 1:
      raise ValueError('len(partition_probabilities) != '
                       'len(partition_names) - 1. '
                       'Last probability is implicity.')
    self.partition_names = partition_names
    self.cumulative_density = np.cumsum(partition_probabilities).tolist()
    self.rand_func = random.random
    self.stats = self._make_stats_dict()

  def transform(self, input_object):
    r = self.rand_func()
    if r >= self.cumulative_density[-1]:
      bucket = len(self.cumulative_density)
    else:
      for i, cpd in enumerate(self.cumulative_density):
        if r < cpd:
          bucket = i
          break
    self.stats = self._make_stats_dict(self.partition_names[bucket])
    return dict([(name, [] if i != bucket else [input_object])
                 for i, name in enumerate(self.partition_names)])

  def get_stats(self):
    return self.stats

  def _make_stats_dict(self, increment_partition=None):
    stats = dict([(name + '_count', statistics.Counter())
                  for name in self.partition_names])
    if increment_partition is not None:
      stats[increment_partition + '_count'].increment()
    return stats

