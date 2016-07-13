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
"""Tests for sequences_lib."""

# internal imports
import tensorflow as tf

from google.protobuf import text_format
from magenta.lib import sequences_lib


def assert_set_equality(test_case, expected, actual):
  """Asserts that two lists are equal without order.

  Given two lists, treat them as sets and test equality. This function only
  requires an __eq__ method to be defined on the objects, and not __hash__
  which set comparison requires. This function removes the burdon of defining
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
  actual_copy = list(actual)
  for expected_obj in expected:
    found = False
    for actual_obj in actual_copy:
      if expected_obj == actual_obj:
        actual_copy.remove(actual_obj)
        found = True
        break
    if not found:
      test_case.fail('Expected %s not found in actual collection' %
                     expected_obj)
  if actual_copy:
    test_case.fail('Actual objects %s not found in expected collection' %
                   actual_copy)


def parse_test_proto(proto_type, proto_string):
  instance = proto_type()
  text_format.Merge(proto_string, instance)
  return instance


def add_track(note_sequence, instrument, notes):
  for pitch, velocity, start_time, end_time in notes:
    note = note_sequence.notes.add()
    note.pitch = pitch
    note.velocity = velocity
    note.start_time = start_time
    note.end_time = end_time
    note.instrument = instrument


def add_quantized_track(quantized_sequence, instrument, notes):
  if instrument not in quantized_sequence.tracks:
    quantized_sequence.tracks[instrument] = []
  track = quantized_sequence.tracks[instrument]
  for pitch, velocity, start_step, end_step in notes:
    note = sequences_lib.Note(pitch=pitch,
                              velocity=velocity,
                              start=start_step,
                              end=end_step,
                              instrument=instrument,
                              program=0)
    track.append(note)