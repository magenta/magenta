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
"""Defines statistics objects for pipelines."""

import abc
import bisect


# https://docs.python.org/2/library/bisect.html#searching-sorted-lists
def _find_le(a, x):
  """Find rightmost value less than or equal to x."""
  i = bisect.bisect_right(a, x)
  if i:
    return a[i-1]
  raise ValueError


class Statistic(object):

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def merge_from(self, other):
    pass

  @abc.abstractmethod
  def pretty_print(self, name):
    pass


class Counter(Statistic):

  def __init__(self, start_value=0):
    super(Counter, self).__init__()
    self.count = start_value

  def increment(self, inc=1):
    self.count += inc

  def merge_from(self, other):
    self.count += other.count

  def pretty_print(self, name):
    return '%s: %d' % (name, self.count)


class Histogram(Statistic):

  def __init__(self, buckets):
    super(Histogram, self).__init__()

    # List of inclusive lowest values in each bucket.
    self.buckets = [float('-inf')] + sorted(buckets) 

    self.counters = dict([(bucket_lower, list())
                          for bucket_lower in self.buckets])

  def increment(self, value, inc=1):
    bucket_lower = _find_le(self.buckets, value)
    self.counters[bucket_lower] += inc

  def merge_from(self, other):
    assert self.buckets == other.buckets
    for bucket_lower, count in other.counters.items():
      self.counters[bucket_lower] += count

  def pretty_print(self, name):
    b = self.buckets + [float('inf')]
    return '%s:\n' + '\n'.join(
        ['  [%s,%s): %d' % (lower, b[i+1], self.counters[lower])
         for i, lower in enumerate(self.buckets)])
