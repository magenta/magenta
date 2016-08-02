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
import copy

# internal imports
import tensorflow as tf


class MergeStatisticsException(Exception):
  pass


class Statistic(object):
  """Holds statistics about a Pipeline run.

  Pipelines produce statistics on each call to `transform`.
  Statistic objects can be merged together to aggregate
  statistics over the course of many calls to `transform`.

  A `Statistic` also has a string name which is used during merging. Any two
  `Statistic` instances with the same name may be merged together. The name
  should also be informative about what the `Statistic` is measuring. Names
  do not need to be unique globally (outside of the `Pipeline` objects that
  produce them) because a `Pipeline` that returns statistics will prepend
  its own name, effectively creating a namespace for each `Pipeline`.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, name):
    """Constructs a `Statistic`.

    Subclass constructors are expected to call this constructor.

    Args:
      name: The string name for this `Statistic`. Any two `Statistic` objects
          with the same name will be merged together. The name should also
          describe what this statistic is measuring.
    """
    self._name = name

  @abc.abstractmethod
  def _merge_from(self, other):
    """Merge another Statistic into this instance.

    Takes another Statistic of the same type, and merges its information into
    this instance.

    Args:
      other: Another Statistic instance.
    """
    pass

  @abc.abstractmethod
  def _pretty_print(self, name):
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

  @abc.abstractmethod
  def copy(self):
    """Returns a new copy of `self`."""
    pass

  def merge_from(self, other):
    if not isinstance(other, Statistic):
      raise MergeStatisticsException(
          'Cannot merge with non-Statistic of type %s' % type(other))
    if self.name != other.name:
      raise MergeStatisticsException(
          'Name "%s" does not match this name "%s"' % (other.name, self.name))
    self._merge_from(other)

  @property
  def name(self):
    """String name of this statistic.

    This name is used to uniquely identify a statistic.

    Returns:
      The string name of `self`.
    """
    return self._name

  @name.setter
  def name(self, value):
    assert isinstance(value, basestring) and value
    self._name = value

  def __str__(self):
    return self._pretty_print(self._name)


def merge_statistics(stats_list):
  """Merge together statistics of the same name in the given list.

  Any two statistics in the list with the same name will be merged into a
  single statistic using the `merge_from` method.

  Args:
    stats_list: A list of `Statistic` objects.

  Returns:
    A list of merged statistics. Each name will appear only once.
  """
  name_map = {}
  for stat in stats_list:
    if stat.name not in name_map:
      name_map[stat.name] = stat
    else:
      name_map[stat.name].merge_from(stat)
  return name_map.values()


def log_statistics_list(stats_list, logger_fn=tf.logging.info):
  """Calls the given logger function on each `Statistic` in the list.

  Args:
    stats_list: A list of `Statistic` objects.
    logger_fn: The function which will be called on the string representation
        of each `Statistic`.
  """
  for stat in stats_list:
    logger_fn(str(stat))


class Counter(Statistic):
  """Holds a count.

  Use `Counter` to count occurrences or sum values together.
  """

  def __init__(self, name, start_value=0):
    """Constructs a Counter.

    Args:
      name: String name of this counter.
      start_value: What value to start the count at.
    """
    super(Counter, self).__init__(name)
    self.count = start_value

  def increment(self, inc=1):
    """Increment the count.

    Args:
      inc: (defaults to 1) How much to increment the count by.
    """
    self.count += inc

  def _merge_from(self, other):
    """Adds the count of another Counter into this instance."""
    if not isinstance(other, Counter):
      raise MergeStatisticsException(
          'Cannot merge %s into Counter' % other.__class__.__name__)
    self.count += other.count

  def _pretty_print(self, name):
    return '%s: %d' % (name, self.count)

  def copy(self):
    return copy.copy(self)


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

  def __init__(self, name, buckets, verbose_pretty_print=False):
    """Initializes the histogram with the given ranges.

    Args:
      name: String name of this histogram.
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
    super(Histogram, self).__init__(name)

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

  def _merge_from(self, other):
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

  def _pretty_print(self, name):
    b = self.buckets + [float('inf')]
    return ('%s:\n' % name) + '\n'.join(
        ['  [%s,%s): %d' % (lower, b[i+1], self.counters[lower])
         for i, lower in enumerate(self.buckets)
         if self.verbose_pretty_print or self.counters[lower]])

  def copy(self):
    return copy.copy(self)
