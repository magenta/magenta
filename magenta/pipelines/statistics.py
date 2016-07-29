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

# internal imports
import tensorflow as tf


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
    """Merge another Statistic into this instance.

    Takes another Statistic of the same type, and merges its information into
    this instance.

    Args:
      other: Another Statistic instance.
    """
    pass

  @abc.abstractmethod
  def pretty_print(self, name):
    """Return a string representation of this instance using the given name.

    Returns a human readable and nicely presented representation of this
    instance. Since this instance does not know what its measuring, a string
    name is given to use in the string representation.

    For example, if this Statistic held a count, say 5, and the given name was
    'error_count', then the string representation might be 'error_count: 5'.

    Args:
      name: A string name for this instance.

    Returns:
      A human readable and preferably a nicely presented string representation
      of this instance.
    """
    pass


def merge_statistics_dicts(merge_to, merge_from):
  for name in merge_from:
    if name in merge_to:
      merge_to[name].merge_from(merge_from[name])
    else:
      merge_to[name] = merge_from[name]


def is_valid_statistics_dict(stats_dict):
  if not isinstance(stats_dict, dict):
    return False
  if [key for key in stats_dict if not isinstance(key, basestring)]:
    return False
  if [val for val in stats_dict.values() if not isinstance(val, Statistic)]:
    return False
  return True


def log_statistics_dict(stats_dict, logger_fn=tf.logging.info):
  for name, stat in stats_dict.items():
    logger_fn(stat.pretty_print(name))


class Counter(Statistic):
  """Holds a count.

  Use `Counter` to count occurrences or sum values together.
  """

  def __init__(self, start_value=0):
    """Constructs a Counter.

    Args:
      start_value: What value to start the count at.
    """
    super(Counter, self).__init__()
    self.count = start_value

  def increment(self, inc=1):
    """Increment the count.

    Args:
      inc: (defaults to 1) How much to increment the count by.
    """
    self.count += inc

  def merge_from(self, other):
    """Adds the count of another Counter into this instance."""
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

  def __init__(self, buckets, verbose_pretty_print=False):
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
      verbose_pretty_print: If True, self.pretty_print will print the count for
          every bucket. If False, only buckets with positive counts will be
          printed.
    """
    super(Histogram, self).__init__()

    # List of inclusive lowest values in each bucket.
    self.buckets = [float('-inf')] + sorted(set(buckets))
    self.counters = dict([(bucket_lower, 0)
                          for bucket_lower in self.buckets])
    self.verbose_pretty_print = verbose_pretty_print

  # https://docs.python.org/2/library/bisect.html#searching-sorted-lists
  def _find_le(self, x):
    """Find rightmost bucket less than or equal to x."""
    i = bisect.bisect_right(self.buckets, x)
    if i:
      return self.buckets[i-1]
    raise ValueError

  def increment(self, value, inc=1):
    """Increment the bucket containing the given value.

    The bucket count for which ever range `value` falls in will be incremented.

    Args:
      value: Any number.
      inc: An integer. How much to increment the bucket count by.
    """
    bucket_lower = self._find_le(value)
    self.counters[bucket_lower] += inc

  def merge_from(self, other):
    """Adds the counts of another Histogram into this instance.

    `other` must have the same buckets as this instance. The counts
    from `other` are added to the counts for this instance.

    Args:
      other: Another Histogram instance with the same buckets as this instance.

    Raises:
      MergeStatisticsException: If `other` is not a Histogram or the buckets
          are not the same.
    """
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
         for i, lower in enumerate(self.buckets)
         if self.verbose_pretty_print or self.counters[lower]])
