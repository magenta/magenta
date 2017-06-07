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
"""Tests for pipelines_common."""

# internal imports
import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.pipelines import pipelines_common
from magenta.protobuf import music_pb2


class PipelineUnitsCommonTest(tf.test.TestCase):

  def _unit_transform_test(self, unit, input_instance,
                           expected_outputs):
    outputs = unit.transform(input_instance)
    self.assertTrue(isinstance(outputs, list))
    common_testing_lib.assert_set_equality(self, expected_outputs, outputs)
    self.assertEqual(unit.input_type, type(input_instance))
    if outputs:
      self.assertEqual(unit.output_type, type(outputs[0]))

  def testTimeChangeSplitter(self):
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          time: 2.0
          numerator: 3
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    expected_sequences = sequences_lib.split_note_sequence_on_time_changes(
        note_sequence)

    unit = pipelines_common.TimeChangeSplitter()
    self._unit_transform_test(unit, note_sequence, expected_sequences)

  def testQuantizer(self):
    steps_per_quarter = 4
    note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 60}""")
    testing_lib.add_track_to_sequence(
        note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    expected_quantized_sequence = sequences_lib.quantize_note_sequence(
        note_sequence, steps_per_quarter)

    unit = pipelines_common.Quantizer(steps_per_quarter)
    self._unit_transform_test(unit, note_sequence,
                              [expected_quantized_sequence])

  def testRandomPartition(self):
    random_partition = pipelines_common.RandomPartition(
        str, ['a', 'b', 'c'], [0.1, 0.4])
    random_nums = [0.55, 0.05, 0.34, 0.99]
    choices = ['c', 'a', 'b', 'c']
    random_partition.rand_func = iter(random_nums).next
    self.assertEqual(random_partition.input_type, str)
    self.assertEqual(random_partition.output_type,
                     {'a': str, 'b': str, 'c': str})
    for i, s in enumerate(['hello', 'qwerty', '1234567890', 'zxcvbnm']):
      results = random_partition.transform(s)
      self.assertTrue(isinstance(results, dict))
      self.assertEqual(set(results.keys()), set(['a', 'b', 'c']))
      self.assertEqual(len(results.values()), 3)
      self.assertEqual(len([l for l in results.values() if l == []]), 2)  # pylint: disable=g-explicit-bool-comparison
      self.assertEqual(results[choices[i]], [s])


if __name__ == '__main__':
  tf.test.main()
