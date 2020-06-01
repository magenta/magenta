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

"""Tests for pipelines_common."""

import functools

from absl.testing import absltest
from magenta.common import testing_lib as common_testing_lib
from magenta.pipelines import pipelines_common


class PipelineUnitsCommonTest(absltest.TestCase):

  def _unit_transform_test(self, unit, input_instance,  # pylint:disable=g-unreachable-test-method
                           expected_outputs):
    outputs = unit.transform(input_instance)
    self.assertIsInstance(outputs, list)
    common_testing_lib.assert_set_equality(self, expected_outputs, outputs)
    self.assertEqual(unit.input_type, type(input_instance))
    if outputs:
      self.assertEqual(unit.output_type, type(outputs[0]))

  def testRandomPartition(self):
    random_partition = pipelines_common.RandomPartition(
        str, ['a', 'b', 'c'], [0.1, 0.4])
    random_nums = [0.55, 0.05, 0.34, 0.99]
    choices = ['c', 'a', 'b', 'c']
    random_partition.rand_func = functools.partial(next, iter(random_nums))
    self.assertEqual(random_partition.input_type, str)
    self.assertEqual(random_partition.output_type,
                     {'a': str, 'b': str, 'c': str})
    for i, s in enumerate(['hello', 'qwerty', '1234567890', 'zxcvbnm']):
      results = random_partition.transform(s)
      self.assertIsInstance(results, dict)
      self.assertEqual(set(results.keys()), set(['a', 'b', 'c']))
      self.assertLen(results.values(), 3)
      self.assertLen([l for l in results.values() if l == []], 2)  # pylint: disable=g-explicit-bool-comparison
      self.assertEqual(results[choices[i]], [s])


if __name__ == '__main__':
  absltest.main()
