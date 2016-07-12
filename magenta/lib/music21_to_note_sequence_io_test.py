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

"""Test of ensure correct import of Music21 scores"""

import tensorflow as tf
from music21 import corpus
import pretty_music21
from music21_to_note_sequence_io import music21_to_sequence_proto


class Music21ScoretoNoteSequenceTest(tf.test.TestCase):
  def testMusic21ToSequence1(self):
    simple_score = pretty_music21.PrettyMusic21(corpus.parse('bwv66.6'))
    sequence_proto = music21_to_sequence_proto(simple_score)
    self.CompareNoteSequenceAndMusic21Score(sequence_proto, simple_score)

  def testMusic21ToSequence2(self):
    simple_score = pretty_music21.PrettyMusic21(corpus.parse('bwv117.4.mxl'))
    sequence_proto = music21_to_sequence_proto(simple_score)
    self.CompareNoteSequenceAndMusic21Score(sequence_proto, simple_score)

  def CompareNoteSequenceAndMusic21Score(self, sequence_proto, score):
    """Compares a sequence proto to a Music21 Score object

    Args:
      sequence_proto: A tensorflow.magenta.Sequence proto.
      score: A music21.Score object.
    """
    # Test time signature changes.
    self.assertEqual(len(score.time_signature_changes),
                     len(sequence_proto.time_signatures))
    for score_time, sequence_time in zip(score.time_signature_changes,
                                         sequence_proto.time_signatures):
      self.assertEqual(score_time.numerator, sequence_time.numerator)
      self.assertEqual(score_time.denominator, sequence_time.denominator)
      self.assertAlmostEqual(score_time.time, sequence_time.time)

    # Test key signature changes.
    self.assertEqual(len(score.key_signature_changes),
                     len(sequence_proto.key_signatures))
    for score_key, sequence_key in zip(score.key_signature_changes,
                                       sequence_proto.key_signatures):
      self.assertEqual(score_key.key_number, sequence_key.key)
      self.assertEqual(score_key.mode, sequence_key.mode)
      self.assertAlmostEqual(score_key.time, sequence_key.time)

    # Test tempos.
    self.assertEqual(len(score.tempo_changes),
                     len(sequence_proto.tempos))
    for score_tempo, sequence_tempo in zip(
        score.tempo_changes, sequence_proto.tempos):
      self.assertAlmostEqual(score_tempo.bpm, sequence_tempo.bpm)
      self.assertAlmostEqual(score_tempo.time, sequence_tempo.time)

    # Test part info.
    self.assertEqual(len(score.part_infos), len(sequence_proto.part_info))
    for score_part_info, sequence_part_info in zip(
        score.part_infos, sequence_proto.part_info):
      self.assertEqual(score_part_info.index, sequence_part_info.part)
      self.assertEqual(score_part_info.name, sequence_part_info.name)

    # Test parts and notes.
    for score_note, sequence_note in zip(score.sorted_notes, sequence_proto.notes):
      self.assertAlmostEqual(score_note.pitch, sequence_note.pitch)
      self.assertAlmostEqual(score_note.start, sequence_note.start_time)
      self.assertAlmostEqual(score_note.end, sequence_note.end_time)
      self.assertEqual(score_note.part, sequence_note.part)


if __name__ == '__main__':
  tf.test.main()
