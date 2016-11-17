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
"""Tests for polyphony_lib."""

# internal imports
import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.models.polyphonic_rnn import polyphony_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2

from magenta.models.polyphonic_rnn.polyphony_lib import EVENT_START
from magenta.models.polyphonic_rnn.polyphony_lib import EVENT_END
from magenta.models.polyphonic_rnn.polyphony_lib import EVENT_STEP_END
from magenta.models.polyphonic_rnn.polyphony_lib import EVENT_NEW_NOTE
from magenta.models.polyphonic_rnn.polyphony_lib import EVENT_CONTINUED_NOTE

class PolyphonyLibTest(tf.test.TestCase):

  def setUp(self):
    self.maxDiff = None

    self.note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        tempos: {
          qpm: 60
        }
        ticks_per_quarter: 220
        """)

  def testFromQuantizedNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    poly_seq = list(polyphony_lib.PolyphonicSequence(quantized_sequence))

    pe = polyphony_lib.PolyphonicEvent
    expected_poly_seq = [
        pe(EVENT_START, 0),
        # step 0
        pe(EVENT_NEW_NOTE, 60),
        pe(EVENT_NEW_NOTE, 64),
        pe(EVENT_STEP_END, 0),
        # step 1
        pe(EVENT_CONTINUED_NOTE, 60),
        pe(EVENT_CONTINUED_NOTE, 64),
        pe(EVENT_NEW_NOTE, 67),
        pe(EVENT_STEP_END, 0),
        # step 2
        pe(EVENT_CONTINUED_NOTE, 60),
        pe(EVENT_CONTINUED_NOTE, 64),
        pe(EVENT_CONTINUED_NOTE, 67),
        pe(EVENT_STEP_END, 0),
        # step 3
        pe(EVENT_CONTINUED_NOTE, 60),
        pe(EVENT_CONTINUED_NOTE, 64),
        pe(EVENT_STEP_END, 0),
        # step 4
        pe(EVENT_CONTINUED_NOTE, 60),
        pe(EVENT_STEP_END, 0),

        pe(EVENT_END, 0),
    ]
    self.assertEqual(expected_poly_seq, poly_seq)

  def testToSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    poly_seq = polyphony_lib.PolyphonicSequence(quantized_sequence)
    poly_seq_ns = poly_seq.to_sequence(qpm=60.0)

    # Make comparison easier
    poly_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, poly_seq_ns)

  def testToSequenceWithContinuedNotesNotStarted(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        pe(EVENT_START, 0),
        # step 0
        pe(EVENT_NEW_NOTE, 60),
        pe(EVENT_NEW_NOTE, 64),
        pe(EVENT_STEP_END, 0),
        # step 1
        pe(EVENT_CONTINUED_NOTE, 60),
        pe(EVENT_CONTINUED_NOTE, 64),
        pe(EVENT_CONTINUED_NOTE, 67),  # Was not started, should be ignored.
        pe(EVENT_STEP_END, 0),

        pe(EVENT_END, 0),
    ]
    for event in poly_events:
      poly_seq.append(event)

    poly_seq_ns = poly_seq.to_sequence(qpm=60.0)

    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 1.0), (64, 100, 0.0, 1.0)])

    # Make comparison easier
    poly_seq_ns.notes.sort(key=lambda n: (n.start_time, n.pitch))
    self.note_sequence.notes.sort(key=lambda n: (n.start_time, n.pitch))

    self.assertEqual(self.note_sequence, poly_seq_ns)

  def testNumSteps(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        pe(EVENT_START, 0),
        # step 0
        pe(EVENT_NEW_NOTE, 60),
        pe(EVENT_NEW_NOTE, 64),
        pe(EVENT_STEP_END, 0),
        # step 1
        pe(EVENT_CONTINUED_NOTE, 60),
        pe(EVENT_CONTINUED_NOTE, 64),
        pe(EVENT_STEP_END, 0),

        pe(EVENT_END, 0),
    ]
    for event in poly_events:
      poly_seq.append(event)

    self.assertEqual(2, poly_seq.num_steps)

  def testExtractPolyphonicSequences(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)

    seqs, _ = polyphony_lib.extract_polyphonic_sequences(quantized_sequence)
    self.assertEqual(1, len(seqs))

    seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_sequence, min_steps_discard=2, max_steps_discard=5)
    self.assertEqual(1, len(seqs))

    self.note_sequence.notes[0].end_time = 1.0
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_sequence, min_steps_discard=3, max_steps_discard=5)
    self.assertEqual(0, len(seqs))

    self.note_sequence.notes[0].end_time = 10.0
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_sequence, min_steps_discard=3, max_steps_discard=5)
    self.assertEqual(0, len(seqs))


if __name__ == '__main__':
  tf.test.main()
