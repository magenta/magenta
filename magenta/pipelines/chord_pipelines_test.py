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
"""Tests for chord_pipelines."""

# internal imports
import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import chords_lib
from magenta.music import constants
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.pipelines import chord_pipelines

NO_CHORD = constants.NO_CHORD


class ChordPipelinesTest(tf.test.TestCase):

  def _unit_transform_test(self, unit, input_instance,
                           expected_outputs):
    outputs = unit.transform(input_instance)
    self.assertTrue(isinstance(outputs, list))
    common_testing_lib.assert_set_equality(self, expected_outputs, outputs)
    self.assertEqual(unit.input_type, type(input_instance))
    if outputs:
      self.assertEqual(unit.output_type, type(outputs[0]))

  def testChordsExtractor(self):
    quantized_sequence = sequences_lib.QuantizedSequence()
    quantized_sequence.steps_per_quarter = 1
    testing_lib.add_quantized_chords_to_sequence(
        quantized_sequence, [('C', 2), ('Am', 4), ('F', 5)])
    quantized_sequence.total_steps = 8
    expected_events = [[NO_CHORD, NO_CHORD, 'C', 'C', 'Am', 'F', 'F', 'F']]
    expected_chord_progressions = []
    for events_list in expected_events:
      chords = chords_lib.ChordProgression(
          events_list, steps_per_quarter=1, steps_per_bar=4)
      expected_chord_progressions.append(chords)
    unit = chord_pipelines.ChordsExtractor(all_transpositions=False)
    self._unit_transform_test(unit, quantized_sequence,
                              expected_chord_progressions)


if __name__ == '__main__':
  tf.test.main()
