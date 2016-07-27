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
"""Tests for statistics."""

# internal imports
import tensorflow as tf

from magenta.pipelines import statistics


class StatisticsTest(tf.test.TestCase):

  def testCounter(self):
    counter = statistics.Counter()
    self.assertEqual(counter.count, 0)
    counter.increment()
    self.assertEqual(counter.count, 1)
    counter.increment(10)
    self.assertEqual(counter.count, 11)

    counter_2 = statistics.Counter(5)
    self.assertEqual(counter_2.count, 5)
    counter.merge_from(counter_2)
    self.assertEqual(counter.count, 16)

    class ABC(object):
      pass

    with self.assertRaises(
        statistics.MergeStatisticsException):
      counter.merge_from(ABC())

    self.assertEqual(counter.pretty_print('name_123'), 'name_123: 16')

  def testHistogram(self):
    histo = statistics.Histogram([1, 2, 10])
    self.assertEqual(histo.counters, {float('-inf'): 0, 1: 0, 2: 0, 10: 0})
    histo.increment(1)
    self.assertEqual(histo.counters, {float('-inf'): 0, 1: 1, 2: 0, 10: 0})
    histo.increment(3, 3)
    self.assertEqual(histo.counters, {float('-inf'): 0, 1: 1, 2: 3, 10: 0})
    histo.increment(20)
    histo.increment(100)
    self.assertEqual(histo.counters, {float('-inf'): 0, 1: 1, 2: 3, 10: 2})
    histo.increment(0)
    histo.increment(-10)
    self.assertEqual(histo.counters, {float('-inf'): 2, 1: 1, 2: 3, 10: 2})

    histo_2 = statistics.Histogram([1, 2, 10])
    histo_2.increment(0, 4)
    histo_2.increment(2, 10)
    histo_2.increment(10, 1)
    histo.merge_from(histo_2)
    self.assertEqual(histo.counters, {float('-inf'): 6, 1: 1, 2: 13, 10: 3})

    histo_3 = statistics.Histogram([1, 2, 7])
    with self.assertRaisesRegexp(
        statistics.MergeStatisticsException,
        r'Histogram buckets do not match. '
        r'Expected \[-inf, 1, 2, 10\], got \[-inf, 1, 2, 7\]'):
      histo.merge_from(histo_3)

    class ABC(object):
      pass

    with self.assertRaises(
        statistics.MergeStatisticsException):
      histo.merge_from(ABC())

    self.assertEqual(
        histo.pretty_print('name_123'),
        'name_123:\n  [-inf,1): 6\n  [1,2): 1\n  [2,10): 13\n  [10,inf): 3')


if __name__ == '__main__':
  tf.test.main()
