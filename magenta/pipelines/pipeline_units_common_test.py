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
"""Tests for pipeline_units_common."""

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import sequences_lib
from magenta.lib import testing_lib
from magenta.pipelines import pipeline_units_common
from magenta.protobuf import music_pb2


NOTE_OFF = melodies_lib.NOTE_OFF
NO_EVENT = melodies_lib.NO_EVENT


class PipelineUnitsCommonTest(tf.test.TestCase):

  def _unit_transform_test(self, unit, input_instance,
                           expected_outputs):
    outputs = unit.transform(input_instance)
    self.assertTrue(isinstance(outputs, list))
    testing_lib.assert_set_equality(self, expected_outputs, outputs)
    self.assertEqual(unit.input_type, type(input_instance))
    if outputs:
      self.assertEqual(unit.output_type, type(outputs[0]))

  def testQuantizer(self):
    steps_per_beat = 4
    note_sequence = testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          bpm: 60}""")
    testing_lib.add_track(
        note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])
    expected_quantized_sequence = sequences_lib.QuantizedSequence()
    expected_quantized_sequence.bpm = 60.0
    expected_quantized_sequence.steps_per_beat = steps_per_beat
    testing_lib.add_quantized_track(
        expected_quantized_sequence, 0,
        [(12, 100, 0, 40), (11, 55, 1, 2), (40, 45, 10, 14),
         (55, 120, 16, 17), (52, 99, 19, 20)])

    unit = pipeline_units_common.Quantizer(steps_per_beat)
    self._unit_transform_test(unit, note_sequence,
                              [expected_quantized_sequence])

  def testMonophonicMelodyExtractor(self):
    quantized_sequence = sequences_lib.QuantizedSequence()
    quantized_sequence.steps_per_beat = 1
    testing_lib.add_quantized_track(
        quantized_sequence, 0,
        [(12, 100, 2, 4), (11, 1, 6, 7)])
    testing_lib.add_quantized_track(
        quantized_sequence, 1,
        [(12, 127, 2, 4), (14, 50, 6, 8)])
    expected_events = [
        [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 11, NOTE_OFF],
        [NO_EVENT, NO_EVENT, 12, NO_EVENT, NOTE_OFF, NO_EVENT, 14, NO_EVENT,
         NOTE_OFF]]
    expected_melodies = []
    for events_list in expected_events:
      melody = melodies_lib.MonophonicMelody()
      melody.from_event_list(events_list)
      melody.steps_per_bar = 4
      expected_melodies.append(melody)
    expected_melodies[0].end_step = 8
    expected_melodies[1].end_step = 12

    unit = pipeline_units_common.MonophonicMelodyExtractor(
        min_bars=1, min_unique_pitches=1, gap_bars=1)
    self._unit_transform_test(unit, quantized_sequence, expected_melodies)


if __name__ == '__main__':
  tf.test.main()
