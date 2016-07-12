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

"""Tests to ensure correct extraction of music21 score objects"""

from music21 import stream, note, meter, key
import pretty_music21
import tensorflow as tf


class SimplifyMusic21Test(tf.test.TestCase):

  def setUp(self):
    """Make a simple score with pick up and two voices"""
    sc = stream.Score()
    num_voices = 2
    pitches = ['C', 'A-']
    for i in range(num_voices):
      part = stream.Part()
      part.id = 'part %d' % i
      time_sig = meter.TimeSignature('4/4')
      key_sig = key.Key('c')

      # Add pickup measure
      pickup = stream.Measure()
      pickup.append(time_sig)
      pickup.append(key_sig)
      n1 = note.Note(pitches[i])
      n1.duration.quarterLength = 1
      pickup.append(n1)
      part.append(pickup)

      # Add full measure
      full_m = stream.Measure()
      full_m.append(n1)
      n2 = n1.transpose('M2')
      full_m.append(n2)
      full_m.repeatAppend(n1, 2)
      part.append(full_m)

      sc.insert(0, part)

    # Shows the full score and all score elements in indented text
    #sc.show('text')

    self.source = sc
    self.score = pretty_music21.PrettyMusic21(sc)


  def testCompareScores(self):
    """Test pretty_music21 score by comparing to music21 score"""

    # Check overall length.
    self.assertAlmostEqual(self.source.duration.quarterLength, self.score.total_time)
    
    # Check number of parts.
    self.assertEqual(len(self.source.parts), len(self.score.parts))

    # Check the notes.
    # TODO: have not included pretty_music21.convert_time to convert time yet
    for part_num in range(len(self.source.parts)):
      part_flat = self.source.parts[part_num].flat
      for note, simple_note in zip(part_flat.getElementsByClass('Note'), self.score.parts[part_num]):
        self.assertEqual(note.pitch.midi, simple_note.pitch)
        note_start = note.getOffsetBySite(part_flat)
        self.assertEqual(note_start, simple_note.start)
        self.assertEqual(note_start + note.duration.quarterLength, simple_note.end)
        # TODO: compare other note attributes

    # Check the time signature.
    self.assertEqual(len(self.score.time_signature_changes), 2)
    # pickup measure of 1/4, and then a full measure of 4/4
    correct_time_sigs = [(0.0, 1, 4), (1.0, 4, 4)]
    for i, time_sig in enumerate(self.score.time_signature_changes):
      self.assertAlmostEqual(time_sig.time, correct_time_sigs[i][0])
      self.assertEqual(time_sig.numerator, correct_time_sigs[i][1])
      self.assertEqual(time_sig.denominator, correct_time_sigs[i][2])

    # Check the key signature.
    retrieved_key_sigs = self.score.key_signature_changes
    self.assertEqual(len(retrieved_key_sigs), 1)
    self.assertEqual(retrieved_key_sigs[0].time, 0.0)
    self.assertEqual(retrieved_key_sigs[0].key_number, 0)  # 0 for c
    self.assertEqual(retrieved_key_sigs[0].mode, 1)  # 1 for minor

    # TODO: Check tempo


  def testSortedNotes(self):
    """Test if notes are sorted by start time"""
    notes = self.score.sorted_notes
    assert all(notes[i].start <= notes[i+1].start for i in range(len(notes)-1))


if __name__ == "__main__":
  tf.test.main()

