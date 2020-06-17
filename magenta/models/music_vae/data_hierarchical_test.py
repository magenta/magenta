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

"""Tests for hierarchical data converters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from magenta.models.music_vae import data_hierarchical
import note_seq
from note_seq import testing_lib
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class MultiInstrumentPerformanceConverterTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.sequence = note_seq.NoteSequence()
    self.sequence.ticks_per_quarter = 220
    self.sequence.tempos.add().qpm = 120.0

  def testToNoteSequence(self):
    sequence = copy.deepcopy(self.sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(64, 100, 0, 2), (60, 100, 0, 4), (67, 100, 2, 4),
         (62, 100, 4, 6), (59, 100, 4, 8), (67, 100, 6, 8),
        ])
    testing_lib.add_track_to_sequence(
        sequence, 1,
        [(40, 100, 0, 0.125), (50, 100, 0, 0.125), (50, 100, 2, 2.125),
         (40, 100, 4, 4.125), (50, 100, 4, 4.125), (50, 100, 6, 6.125),
        ],
        is_drum=True)
    converter = data_hierarchical.MultiInstrumentPerformanceConverter(
        hop_size_bars=2, chunk_size_bars=2)
    tensors = converter.to_tensors(sequence)
    self.assertEqual(2, len(tensors.outputs))
    sequences = converter.from_tensors(tensors.outputs)
    self.assertEqual(2, len(sequences))

    sequence1 = copy.deepcopy(self.sequence)
    testing_lib.add_track_to_sequence(
        sequence1, 0, [(64, 100, 0, 2), (60, 100, 0, 4), (67, 100, 2, 4)])
    testing_lib.add_track_to_sequence(
        sequence1, 1,
        [(40, 100, 0, 0.125), (50, 100, 0, 0.125), (50, 100, 2, 2.125)],
        is_drum=True)
    self.assertProtoEquals(sequence1, sequences[0])

    sequence2 = copy.deepcopy(self.sequence)
    testing_lib.add_track_to_sequence(
        sequence2, 0, [(62, 100, 0, 2), (59, 100, 0, 4), (67, 100, 2, 4)])
    testing_lib.add_track_to_sequence(
        sequence2, 1,
        [(40, 100, 0, 0.125), (50, 100, 0, 0.125), (50, 100, 2, 2.125)],
        is_drum=True)
    self.assertProtoEquals(sequence2, sequences[1])

  def testToNoteSequenceWithChords(self):
    sequence = copy.deepcopy(self.sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(64, 100, 0, 2), (60, 100, 0, 4), (67, 100, 2, 4),
         (62, 100, 4, 6), (59, 100, 4, 8), (67, 100, 6, 8),
        ])
    testing_lib.add_track_to_sequence(
        sequence, 1,
        [(40, 100, 0, 0.125), (50, 100, 0, 0.125), (50, 100, 2, 2.125),
         (40, 100, 4, 4.125), (50, 100, 4, 4.125), (50, 100, 6, 6.125),
        ],
        is_drum=True)
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 0), ('G', 4)])
    converter = data_hierarchical.MultiInstrumentPerformanceConverter(
        hop_size_bars=2,
        chunk_size_bars=2,
        chord_encoding=note_seq.MajorMinorChordOneHotEncoding())
    tensors = converter.to_tensors(sequence)
    self.assertEqual(2, len(tensors.outputs))
    sequences = converter.from_tensors(tensors.outputs, tensors.controls)
    self.assertEqual(2, len(sequences))

    sequence1 = copy.deepcopy(self.sequence)
    testing_lib.add_track_to_sequence(
        sequence1, 0, [(64, 100, 0, 2), (60, 100, 0, 4), (67, 100, 2, 4)])
    testing_lib.add_track_to_sequence(
        sequence1, 1,
        [(40, 100, 0, 0.125), (50, 100, 0, 0.125), (50, 100, 2, 2.125)],
        is_drum=True)
    testing_lib.add_chords_to_sequence(
        sequence1, [('C', 0)])
    self.assertProtoEquals(sequence1, sequences[0])

    sequence2 = copy.deepcopy(self.sequence)
    testing_lib.add_track_to_sequence(
        sequence2, 0, [(62, 100, 0, 2), (59, 100, 0, 4), (67, 100, 2, 4)])
    testing_lib.add_track_to_sequence(
        sequence2, 1,
        [(40, 100, 0, 0.125), (50, 100, 0, 0.125), (50, 100, 2, 2.125)],
        is_drum=True)
    testing_lib.add_chords_to_sequence(
        sequence2, [('G', 0)])
    self.assertProtoEquals(sequence2, sequences[1])

  def testToNoteSequenceMultipleChunks(self):
    sequence = copy.deepcopy(self.sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(64, 100, 0, 2), (60, 100, 0, 4), (67, 100, 2, 4),
         (62, 100, 4, 6), (59, 100, 4, 8), (67, 100, 6, 8),
        ])
    testing_lib.add_track_to_sequence(
        sequence, 1,
        [(40, 100, 0, 0.125), (50, 100, 0, 0.125), (50, 100, 2, 2.125),
         (40, 100, 4, 4.125), (50, 100, 4, 4.125), (50, 100, 6, 6.125),
        ],
        is_drum=True)
    converter = data_hierarchical.MultiInstrumentPerformanceConverter(
        hop_size_bars=4, chunk_size_bars=2)
    tensors = converter.to_tensors(sequence)
    self.assertEqual(1, len(tensors.outputs))
    sequences = converter.from_tensors(tensors.outputs)
    self.assertEqual(1, len(sequences))
    self.assertProtoEquals(sequence, sequences[0])

  def testToNoteSequenceMultipleChunksWithChords(self):
    sequence = copy.deepcopy(self.sequence)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(64, 100, 0, 2), (60, 100, 0, 4), (67, 100, 2, 4),
         (62, 100, 4, 6), (59, 100, 4, 8), (67, 100, 6, 8),
        ])
    testing_lib.add_track_to_sequence(
        sequence, 1,
        [(40, 100, 0, 0.125), (50, 100, 0, 0.125), (50, 100, 2, 2.125),
         (40, 100, 4, 4.125), (50, 100, 4, 4.125), (50, 100, 6, 6.125),
        ],
        is_drum=True)
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 0), ('G', 4)])
    converter = data_hierarchical.MultiInstrumentPerformanceConverter(
        hop_size_bars=4,
        chunk_size_bars=2,
        chord_encoding=note_seq.MajorMinorChordOneHotEncoding())
    tensors = converter.to_tensors(sequence)
    self.assertEqual(1, len(tensors.outputs))
    sequences = converter.from_tensors(tensors.outputs, tensors.controls)
    self.assertEqual(1, len(sequences))
    self.assertProtoEquals(sequence, sequences[0])


if __name__ == '__main__':
  tf.test.main()
