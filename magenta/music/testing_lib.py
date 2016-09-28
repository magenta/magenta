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
import numpy as np

from google.protobuf import text_format
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


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
    return 'serialized:' + self.string

  def __eq__(self, other):
    return isinstance(other, MockStringProto) and self.string == other.string

  def __hash__(self):
    return hash(self.string)


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


def add_track(note_sequence, instrument, notes):
  for pitch, velocity, start_time, end_time in notes:
    note = note_sequence.notes.add()
    note.pitch = pitch
    note.velocity = velocity
    note.start_time = start_time
    note.end_time = end_time
    note.instrument = instrument


def add_chords(note_sequence, chords):
  for figure, time in chords:
    annotation = note_sequence.text_annotations.add()
    annotation.time = time
    annotation.text = figure
    annotation.annotation_type = CHORD_SYMBOL


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


def add_quantized_chords(quantized_sequence, chords):
  for figure, step in chords:
    chord = sequences_lib.ChordSymbol(step=step, figure=figure)
    quantized_sequence.chords.append(chord)
