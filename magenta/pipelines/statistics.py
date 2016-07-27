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


class MergeStatisticsException(Exception):
  pass


class Statistic(object):
  """Holds statistics about a Pipeline run.

  Pipelines produce statistics on each call to `transform`.
  Statistic objects can be merged together to aggregate
  statistics over the course of many calls to `transforms`.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def merge_from(self, other):
    pass

  @abc.abstractmethod
  def pretty_print(self, name):
    pass


def merge_statistics_dicts(merge_to, merge_from):
  for name in merge_from:
    if name in merge_to:
      merge_to[name].merge_from(merge_from[name])
    else:
      merge_to[name] = merge_from[name]


# https://docs.python.org/2/library/bisect.html#searching-sorted-lists
def _find_le(a, x):
  """Find rightmost value less than or equal to x."""
  i = bisect.bisect_right(a, x)
  if i:
    return a[i-1]
  raise ValueError


class Counter(Statistic):
  """Holds a count.

  Use `Counter` to count occurrences or sum values together.
  """

  def __init__(self, start_value=0):
    super(Counter, self).__init__()
    self.count = start_value

  def increment(self, inc=1):
    self.count += inc

  def merge_from(self, other):
    if not isinstance(other, Counter):
      raise MergeStatisticsException(
          'Cannot merge %s into Counter' % other.__class__.__name__)
    self.count += other.count

  def pretty_print(self, name):
    return '%s: %d' % (name, self.count)


class Histogram(Statistic):
  """Represents a histogram.

  A histogram is a list of counts, each over a range of values.
  For example, given this list of values [0.5, 0.0, 1.0, 0.6, 1.5, 2.4, 0.1],
  a histogram over 3 ranges [0, 1), [1, 2), [2, 3) would be:
    [0, 1): 4
    [1, 2): 2
    [2, 3): 1
  Each range is inclusive in the lower bound and exclusive in the upper bound 
  (hence the square open bracket but curved close bracket).
  """

  def __init__(self, buckets):
    """Initializes the histogram with the given ranges.

    Args:
      buckets: The ranges the histogram counts over. This is a list of values,
          where each value is the inclusive lower bound of the range. An extra
          range will be implicitly defined which spans from negative infinity
          to the lowest given lower bound. The highest given lower bound
          defines a range spaning to positive infinity. This way any value will
          be included in the histogram counts. For example, if `buckets` is
          [4, 6, 10] the histogram will have ranges
          [-inf, 4), [4, 6), [6, 10), [10, inf).
    """
    super(Histogram, self).__init__()

    # List of inclusive lowest values in each bucket.
    self.buckets = [float('-inf')] + sorted(buckets) 
    self.counters = dict([(bucket_lower, 0)
                          for bucket_lower in self.buckets])

  def increment(self, value, inc=1):
    bucket_lower = _find_le(self.buckets, value)
    self.counters[bucket_lower] += inc

  def merge_from(self, other):
    if not isinstance(other, Histogram):
      raise MergeStatisticsException(
          'Cannot merge %s into Histogram' % other.__class__.__name__)
    if self.buckets != other.buckets:
      raise MergeStatisticsException(
          'Histogram buckets do not match. Expected %s, got %s'
          % (self.buckets, other.buckets))
    for bucket_lower, count in other.counters.items():
      self.counters[bucket_lower] += count

  def pretty_print(self, name):
    b = self.buckets + [float('inf')]
    return ('%s:\n' % name) + '\n'.join(
        ['  [%s,%s): %d' % (lower, b[i+1], self.counters[lower])
         for i, lower in enumerate(self.buckets)])
