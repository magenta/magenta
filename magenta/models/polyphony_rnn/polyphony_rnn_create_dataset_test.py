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

"""Tests for polyphony_rnn_create_dataset."""

import magenta
from magenta.contrib import training as contrib_training
from magenta.models.polyphony_rnn import polyphony_encoder_decoder
from magenta.models.polyphony_rnn import polyphony_rnn_pipeline
from magenta.models.shared import events_rnn_model
import note_seq
import note_seq.testing_lib
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

FLAGS = tf.app.flags.FLAGS


class PolySeqPipelineTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.config = events_rnn_model.EventSequenceRnnConfig(
        None,
        note_seq.OneHotEventSequenceEncoderDecoder(
            polyphony_encoder_decoder.PolyphonyOneHotEncoding()),
        contrib_training.HParams())

  def testPolyRNNPipeline(self):
    note_sequence = magenta.common.testing_lib.parse_test_proto(
        note_seq.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 120}""")
    note_seq.testing_lib.add_track_to_sequence(note_sequence, 0,
                                               [(36, 100, 0.00, 2.0),
                                                (40, 55, 2.1, 5.0),
                                                (44, 80, 3.6, 5.0),
                                                (41, 45, 5.1, 8.0),
                                                (64, 100, 6.6, 10.0),
                                                (55, 120, 8.1, 11.0),
                                                (39, 110, 9.6, 9.7),
                                                (53, 99, 11.1, 14.1),
                                                (51, 40, 12.6, 13.0),
                                                (55, 100, 14.1, 15.0),
                                                (54, 90, 15.6, 17.0),
                                                (60, 100, 17.1, 18.0)])

    pipeline_inst = polyphony_rnn_pipeline.get_pipeline(
        min_steps=80,  # 5 measures
        max_steps=512,
        eval_ratio=0,
        config=self.config)
    result = pipeline_inst.transform(note_sequence)
    self.assertTrue(len(result['training_poly_tracks']))


if __name__ == '__main__':
  tf.test.main()
