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
"""Tests for melody_rnn_create_dataset."""

# internal imports
import tensorflow as tf

from magenta.lib import testing_lib
from magenta.models.shared import melody_rnn_create_dataset
from magenta.pipelines import pipeline_units_common
from magenta.protobuf import music_pb2


class MockProto(object):

  def __init__(self, string=''):
    self.string = string

  @staticmethod
  def FromString(string):
    return MockProto(string)

  def SerializeToString(self):
    return 'serialized:' + self.string

  def __eq__(self, other):
    return isinstance(other, MockProto) and self.string == other.string

  def __hash__(self):
    return hash(self.string)


class PipelineTest(tf.test.TestCase):

  def testBasicRNNPipeline(self):
    note_sequence = testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          bpm: 120}""")
    testing_lib.add_track(
        note_sequence, 0,
        [(12, 100, 0.01, 10.0), (11, 55, 0.22, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.01), (52, 99, 4.75, 5.0)])

    quantizer = pipeline_units_common.Quantizer(steps_per_beat=4)
    melody_extractor = pipeline_units_common.MonophonicMelodyExtractor(
        min_bars=7, min_unique_pitches=5,
        gap_bars=1.0)
    one_hot_encoder = melody_rnn_create_dataset.OneHotEncoder()
    quantized = quantizer.transform(note_sequence)[0]
    melody = melody_extractor.transform(quantized)[0]
    one_hot = one_hot_encoder.transform(melody)[0]
    expected_result = {'basic_rnn_train': [one_hot], 'basic_rnn_eval': []}

    pipeline_inst = melody_rnn_create_dataset.BasicRNNPipeline(eval_ratio=0)
    result = pipeline_inst.transform(note_sequence)
    self.assertEqual(expected_result, result)
