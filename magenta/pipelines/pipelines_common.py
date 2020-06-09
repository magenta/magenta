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

"""Common data processing pipelines."""

import numbers
import random

from magenta.pipelines import pipeline
from magenta.pipelines import statistics
import numpy as np
import tensorflow.compat.v1 as tf


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
        type_, dict((name, type_) for name in partition_names))
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
    return dict((name, [] if i != bucket else [input_object])
                for i, name in enumerate(self.partition_names))

  def _make_stats(self, increment_partition=None):
    return [statistics.Counter(increment_partition + '_count', 1)]


def make_sequence_example(inputs, labels):
  """Returns a SequenceExample for the given inputs and labels.

  Args:
    inputs: A list of input vectors. Each input vector is a list of floats.
    labels: A list of ints.

  Returns:
    A tf.train.SequenceExample containing inputs and labels.
  """
  input_features = [
      tf.train.Feature(float_list=tf.train.FloatList(value=input_))
      for input_ in inputs]
  label_features = []
  for label in labels:
    if isinstance(label, numbers.Number):
      label = [label]
    label_features.append(
        tf.train.Feature(int64_list=tf.train.Int64List(value=label)))
  feature_list = {
      'inputs': tf.train.FeatureList(feature=input_features),
      'labels': tf.train.FeatureList(feature=label_features)
  }
  feature_lists = tf.train.FeatureLists(feature_list=feature_list)
  return tf.train.SequenceExample(feature_lists=feature_lists)
