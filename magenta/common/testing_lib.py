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

"""Testing support code."""

import numpy as np
import six
from google.protobuf import text_format


def assert_set_equality(test_case, expected, actual):
  """Asserts that two lists are equal without order.

  Given two lists, treat them as sets and test equality. This function only
  requires an __eq__ method to be defined on the objects, and not __hash__
  which set comparison requires. This function removes the burden of defining
  a __hash__ method just for testing.

  This function calls into tf.test.TestCase.assert* methods and behaves
  like a test assert. The function returns if `expected` and `actual`
  contain the same objects regardless of ordering.

  Note, this is an O(n^2) operation and is not suitable for large lists.

  Args:
    test_case: A tf.test.TestCase instance from a test.
    expected: A list of objects.
    actual: A list of objects.
  """
  actual_found = np.zeros(len(actual), dtype=bool)
  for expected_obj in expected:
    found = False
    for i, actual_obj in enumerate(actual):
      if expected_obj == actual_obj:
        actual_found[i] = True
        found = True
        break
    if not found:
      test_case.fail('Expected %s not found in actual collection' %
                     expected_obj)
  if not np.all(actual_found):
    test_case.fail('Actual objects %s not found in expected collection' %
                   np.array(actual)[np.invert(actual_found)])


def parse_test_proto(proto_type, proto_string):
  instance = proto_type()
  text_format.Merge(proto_string, instance)
  return instance


class MockStringProto(object):
  """Provides common methods for a protocol buffer object.

  Wraps a single string value. This makes testing equality easy.
  """

  def __init__(self, string=''):
    self.string = string

  @staticmethod
  def FromString(string):  # pylint: disable=invalid-name
    return MockStringProto(string)

  def SerializeToString(self):  # pylint: disable=invalid-name
    # protobuf's SerializeToString returns binary string
    if six.PY3:
      return ('serialized:' + self.string).encode('utf-8')
    else:
      return 'serialized:' + self.string

  def __eq__(self, other):
    return isinstance(other, MockStringProto) and self.string == other.string

  def __hash__(self):
    return hash(self.string)
