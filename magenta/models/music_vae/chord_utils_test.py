# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for chord_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# internal imports
import tensorflow as tf

from magenta.models.music_vae import chord_utils
import magenta.music as mm
from magenta.music import testing_lib
from magenta.protobuf import music_pb2


class ChordUtilsTest(tf.test.TestCase):

  def setUp(self):
    self.note_sequence = music_pb2.NoteSequence()
    self.note_sequence.ticks_per_quarter = 220
    self.note_sequence.tempos.add(qpm=60.0)

    testing_lib.add_chords_to_sequence(
        self.note_sequence, [('C', 2), ('G7', 6)])
    self.note_sequence.total_time = 8.0

  def testMelodyChords(self):
    melodies = [
        mm.Melody([60, -2, -2, -1],
                  start_step=0, steps_per_quarter=1, steps_per_bar=4),
        mm.Melody([62, -2, -2, -1],
                  start_step=4, steps_per_quarter=1, steps_per_bar=4),
    ]

    quantized_sequence = mm.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    chords = chord_utils.event_list_chords(quantized_sequence, melodies)

    expected_chords = [
        [mm.NO_CHORD, mm.NO_CHORD, 'C', 'C'],
        ['C', 'C', 'G7', 'G7']
    ]

    self.assertEqual(expected_chords, chords)

  def testAddChordsToSequence(self):
    expected_sequence = copy.deepcopy(self.note_sequence)
    del self.note_sequence.text_annotations[:]

    chords = [mm.NO_CHORD, 'C', 'C', 'G7']
    chord_times = [0.0, 2.0, 4.0, 6.0]
    chord_utils.add_chords_to_sequence(
        self.note_sequence, chords, chord_times)

    self.assertEqual(expected_sequence, self.note_sequence)


if __name__ == '__main__':
  tf.test.main()
