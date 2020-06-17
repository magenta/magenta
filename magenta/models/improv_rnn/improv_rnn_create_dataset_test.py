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

"""Tests for improv_rnn_create_dataset."""

import magenta
from magenta.contrib import training as contrib_training
from magenta.models.improv_rnn import improv_rnn_model
from magenta.models.improv_rnn import improv_rnn_pipeline
from magenta.pipelines import lead_sheet_pipelines
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipelines_common
import note_seq
import note_seq.testing_lib
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

FLAGS = tf.app.flags.FLAGS


class ImprovRNNPipelineTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.config = improv_rnn_model.ImprovRnnConfig(
        None,
        note_seq.ConditionalEventSequenceEncoderDecoder(
            note_seq.OneHotEventSequenceEncoderDecoder(
                note_seq.MajorMinorChordOneHotEncoding()),
            note_seq.OneHotEventSequenceEncoderDecoder(
                note_seq.MelodyOneHotEncoding(0, 127))),
        contrib_training.HParams(),
        min_note=0,
        max_note=127,
        transpose_to_key=0)

  def testMelodyRNNPipeline(self):
    note_sequence = magenta.common.testing_lib.parse_test_proto(
        note_seq.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4}
        tempos: {
          qpm: 120}""")
    note_seq.testing_lib.add_track_to_sequence(note_sequence, 0,
                                               [(12, 100, 0.00, 2.0),
                                                (11, 55, 2.1, 5.0),
                                                (40, 45, 5.1, 8.0),
                                                (55, 120, 8.1, 11.0),
                                                (53, 99, 11.1, 14.1)])
    note_seq.testing_lib.add_chords_to_sequence(note_sequence, [('N.C.', 0.0),
                                                                ('Am9', 5.0),
                                                                ('D7', 10.0)])

    quantizer = note_sequence_pipelines.Quantizer(steps_per_quarter=4)
    lead_sheet_extractor = lead_sheet_pipelines.LeadSheetExtractor(
        min_bars=7, min_unique_pitches=5, gap_bars=1.0,
        ignore_polyphonic_notes=False, all_transpositions=False)
    conditional_encoding = note_seq.ConditionalEventSequenceEncoderDecoder(
        note_seq.OneHotEventSequenceEncoderDecoder(
            note_seq.MajorMinorChordOneHotEncoding()),
        note_seq.OneHotEventSequenceEncoderDecoder(
            note_seq.MelodyOneHotEncoding(self.config.min_note,
                                          self.config.max_note)))
    quantized = quantizer.transform(note_sequence)[0]
    lead_sheet = lead_sheet_extractor.transform(quantized)[0]
    lead_sheet.squash(
        self.config.min_note,
        self.config.max_note,
        self.config.transpose_to_key)
    encoded = pipelines_common.make_sequence_example(
        *conditional_encoding.encode(lead_sheet.chords, lead_sheet.melody))
    expected_result = {'training_lead_sheets': [encoded],
                       'eval_lead_sheets': []}

    pipeline_inst = improv_rnn_pipeline.get_pipeline(
        self.config, eval_ratio=0.0)
    result = pipeline_inst.transform(note_sequence)
    self.assertEqual(expected_result, result)


if __name__ == '__main__':
  tf.test.main()
