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
"""Common data processing pipelines."""

import random

# internal imports
import numpy as np
import tensorflow as tf

from magenta.music import sequences_lib
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


class TimeChangeSplitter(pipeline.Pipeline):
  """A Pipeline that splits NoteSequences on time signature & tempo changes."""

  def __init__(self, name=None):
    super(TimeChangeSplitter, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=music_pb2.NoteSequence,
        name=name)

  def transform(self, note_sequence):
    return sequences_lib.split_note_sequence_on_time_changes(note_sequence)


class Quantizer(pipeline.Pipeline):
  """A Module that quantizes NoteSequence data."""

  def __init__(self, steps_per_quarter=None, steps_per_second=None, name=None):
    super(Quantizer, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=music_pb2.NoteSequence,
        name=name)
    if (steps_per_quarter is not None) == (steps_per_second is not None):
      raise ValueError(
          'Exactly one of steps_per_quarter or steps_per_second must be set.')
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_second = steps_per_second

  def transform(self, note_sequence):
    try:
      if self._steps_per_quarter is not None:
        quantized_sequence = sequences_lib.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
      else:
        quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
            note_sequence, self._steps_per_second)
      return [quantized_sequence]
    except sequences_lib.MultipleTimeSignatureException as e:
      tf.logging.warning('Multiple time signatures in NoteSequence %s: %s',
                         note_sequence.filename, e)
      self._set_stats([statistics.Counter(
          'sequences_discarded_because_multiple_time_signatures', 1)])
      return []
    except sequences_lib.MultipleTempoException as e:
      tf.logging.warning('Multiple tempos found in NoteSequence %s: %s',
                         note_sequence.filename, e)
      self._set_stats([statistics.Counter(
          'sequences_discarded_because_multiple_tempos', 1)])
      return []
    except sequences_lib.BadTimeSignatureException as e:
      tf.logging.warning('Bad time signature in NoteSequence %s: %s',
                         note_sequence.filename, e)
      self._set_stats([statistics.Counter(
          'sequences_discarded_because_bad_time_signature', 1)])
      return []


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

  def transform(self, input_object):
    r = self.rand_func()
    if r >= self.cumulative_density[-1]:
      bucket = len(self.cumulative_density)
    else:
      for i, cpd in enumerate(self.cumulative_density):
        if r < cpd:
          bucket = i
          break
    self._set_stats(self._make_stats(self.partition_names[bucket]))
    return dict([(name, [] if i != bucket else [input_object])
                 for i, name in enumerate(self.partition_names)])

  def _make_stats(self, increment_partition=None):
    return [statistics.Counter(increment_partition + '_count', 1)]
