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
"""Tests to ensure correct extraction of music21 score objects."""

# internal imports
from music21 import key
from music21 import meter
from music21 import note as music21_note
from music21 import pitch
from music21 import stream
import tensorflow as tf
from magenta.music import pretty_music21


class PrettyMusic21Test(tf.test.TestCase):

  def setUp(self):
    self.sources = {'pickup_score': self.makeScoreWithPickup(),
                    'scores': self.makeScore()}
    self.simple_scores = {sc_type: pretty_music21.PrettyMusic21(sc)
                          for sc_type, sc in self.sources.iteritems()}

  def makeScoreWithPickup(self):
    """Make a short score with pick up and two voices."""
    sc = stream.Score()
    num_voices = 2
    pitches = ['C', 'A-']
    for i in range(num_voices):
      part = stream.Part()
      part.id = 'part %d' % i
      time_sig = meter.TimeSignature('4/4')
      key_sig = key.Key('c')

      # Add pickup measure.
      pickup = stream.Measure()
      pickup.append(time_sig)
      pickup.append(key_sig)
      n1 = music21_note.Note(pitches[i])
      n1.duration.quarterLength = 1
      pickup.append(n1)
      part.append(pickup)

      # Add full measure.
      full_m = stream.Measure()
      full_m.append(n1)
      n2 = n1.transpose('M2')
      full_m.append(n2)
      full_m.repeatAppend(n1, 2)
      part.append(full_m)

      sc.insert(0, part)

    # Show the full score and all score elements in indented text.
    # sc.show('text')
    return sc

  def makeScore(self):
    """Make a short score with pick up and two voices."""
    sc = stream.Score()
    num_voices = 2
    pitches = ['C', 'A-']
    for i in range(num_voices):
      part = stream.Part()
      part.id = 'part %d' % i
      time_sig = meter.TimeSignature('4/4')
      key_sig = key.Key('c')

      # Make a note.
      n1 = music21_note.Note(pitches[i])
      n1.duration.quarterLength = 1

      # Add full measure.
      full_m = stream.Measure()
      full_m.append(time_sig)
      full_m.append(key_sig)
      full_m.append(n1)
      n2 = n1.transpose('M2')
      full_m.append(n2)
      full_m.repeatAppend(n1, 2)
      part.append(full_m)

      # Add another full measure.
      full_m = stream.Measure()
      full_m.append(n1)
      n2 = n1.transpose('M2')
      full_m.append(n2)
      full_m.repeatAppend(n1, 2)
      part.append(full_m)

      sc.insert(0, part)

    # Show the full score and all score elements in indented text.
    # sc.show('text')
    return sc

  def testExtractionOfKeySignatureAttributes(self):
    """Check the key, mode, tonic pitch class extraction from key signature."""
    num_to_major_key = {0: 'C',
                        1: 'G',
                        2: 'D',
                        3: 'A',
                        4: 'E',
                        5: 'B',
                        6: 'F#',
                        7: 'C#',
                        8: 'G#',
                        9: 'D#',
                        10: 'A#',
                        11: 'E#',
                        12: 'B#',
                        -2: 'Bb',
                        -12: 'Dbb',
                        -11: 'Abb',
                        -10: 'Ebb',
                        -9: 'Bbb',
                        -8: 'Fb',
                        -7: 'Cb',
                        -6: 'Gb',
                        -5: 'Db',
                        -4: 'Ab',
                        -3: 'Eb',
                        -1: 'F'}
    num_to_minor_key = {0: 'a',
                        1: 'e',
                        2: 'b',
                        3: 'f#',
                        4: 'c#',
                        5: 'g#',
                        6: 'd#',
                        7: 'a#',
                        8: 'e#',
                        9: 'b#',
                        10: 'f##',
                        11: 'c##',
                        12: 'g##',
                        -2: 'g',
                        -12: 'bbb',
                        -11: 'fb',
                        -10: 'cb',
                        -9: 'gb',
                        -8: 'db',
                        -7: 'ab',
                        -6: 'eb',
                        -5: 'bb',
                        -4: 'f',
                        -3: 'c',
                        -1: 'd'}

    for test_mode in ['major', 'minor']:
      for i in range(-12, 13):
        ks = key.KeySignature(i)
        ks.mode = test_mode
        if test_mode == 'major':
          key_map = num_to_major_key
        else:
          key_map = num_to_minor_key
        try:
          key_name, num_sharps, mode, tonic_pitchclass = (
              pretty_music21._extract_key_signature_attributes(ks))
        except pretty_music21.PrettyMusic21Error:
          self.assertTrue(i < 7 or i > 7)
          continue
        self.assertEqual(key_name, key_map[i])
        if mode == 'minor':
          self.assertEqual(
              key.sharpsToPitch(num_sharps + 3).name,
              key.convertKeyStringToMusic21KeyString(key_name).upper())
        else:
          self.assertEqual(
              key.sharpsToPitch(num_sharps).name,
              key.convertKeyStringToMusic21KeyString(key_name).upper())

        self.assertEqual(mode, ks.mode)
        check_pitch = pitch.Pitch(
            key.convertKeyStringToMusic21KeyString(key_map[i]))
        check_pitchclass = check_pitch.pitchClass
        self.assertEqual(tonic_pitchclass, check_pitchclass)

  def testCompareScores(self):
    """Test pretty_music21 score by comparing to music21 score."""
    for score_type, source in self.sources.iteritems():
      simple_score = self.simple_scores[score_type]
      # Check overall length.
      self.assertAlmostEqual(source.duration.quarterLength / 2.0,
                             simple_score.total_time)

      # Check number of parts.
      self.assertEqual(len(source.parts), len(simple_score.parts))

      # Check the notes.
      # TODO(annahuang): Don't assume note lengths are in quarter units.
      for part_num in range(len(source.parts)):
        part_flat = source.parts[part_num].flat
        for note, simple_note in zip(
            part_flat.getElementsByClass('Note'), simple_score.parts[part_num]):
          self.assertEqual(note.pitch.midi, simple_note.pitch_midi)
          self.assertEqual(
              note.pitch.name.replace('-', 'b'), simple_note.pitch_name)
          note_start = note.getOffsetBySite(part_flat)
          self.assertEqual(note_start / 2.0, simple_note.start_time)
          self.assertEqual((note_start + note.duration.quarterLength) / 2.0,
                           simple_note.end_time)
          self.assertEqual(part_num, simple_note.part_index)

      # Check the time signature.
      if 'pickup' in score_type:
        self.assertEqual(len(simple_score.time_signature_changes), 2)
        # Pickup measure of 1/4, and then a full measure of 4/4.
        correct_time_sigs = [(0.0, 1, 4), (0.5, 4, 4)]
      else:
        self.assertEqual(len(simple_score.time_signature_changes), 1)
        correct_time_sigs = [(0.0, 4, 4)]
      for i, time_sig in enumerate(simple_score.time_signature_changes):
        self.assertAlmostEqual(time_sig.time, correct_time_sigs[i][0])
        self.assertEqual(time_sig.numerator, correct_time_sigs[i][1])
        self.assertEqual(time_sig.denominator, correct_time_sigs[i][2])

      # Check the key signature.
      retrieved_key_sigs = simple_score.key_signature_changes
      self.assertEqual(len(retrieved_key_sigs), 1)
      self.assertEqual(retrieved_key_sigs[0].time, 0.0)
      self.assertEqual(retrieved_key_sigs[0].key, 'c')
      self.assertEqual(retrieved_key_sigs[0].mode, 'minor')
      self.assertEqual(retrieved_key_sigs[0].tonic_pitchclass, 0)

      # TODO(annahuang): Check tempo.

  def testSortedNotes(self):
    """Test if notes are sorted by start time."""
    for simple_score in self.simple_scores.values():
      notes = simple_score.sorted_notes
      assert all(notes[i].start_time <= notes[i + 1].start_time
                 for i in range(len(notes) - 1))


if __name__ == '__main__':
  tf.test.main()
