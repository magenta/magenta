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

from magenta.pipelines import pipeline
from magenta.pipelines import statistics


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
