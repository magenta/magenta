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

"""Tests for Score2Perf music encoders."""
import tempfile

from magenta.models.score2perf import music_encoders
import note_seq
from note_seq import testing_lib
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class MidiPerformanceEncoderTest(tf.test.TestCase):

  def testNumReservedIds(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108)
    self.assertEqual(2, encoder.num_reserved_ids)

  def testEncodeEmptyNoteSequence(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108)
    ids = encoder.encode_note_sequence(note_seq.NoteSequence())
    self.assertEqual([], ids)

  def testEncodeEmptyNoteSequenceAddEos(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108,
        add_eos=True)
    ids = encoder.encode_note_sequence(note_seq.NoteSequence())
    self.assertEqual([1], ids)

  def testEncodeNoteSequence(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108)

    ns = note_seq.NoteSequence()
    testing_lib.add_track_to_sequence(
        ns, 0, [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 127, 1.0, 2.0)])
    ids = encoder.encode_note_sequence(ns)

    expected_ids = [
        302,  # VELOCITY(25)
        41,   # NOTE-ON(60)
        45,   # NOTE-ON(64)
        277,  # TIME-SHIFT(100)
        309,  # VELOCITY(32)
        48,   # NOTE-ON(67)
        277,  # TIME-SHIFT(100)
        136,  # NOTE-OFF(67)
        277,  # TIME-SHIFT(100)
        133,  # NOTE-OFF(64
        277,  # TIME-SHIFT(100)
        129   # NOTE-OFF(60)
    ]

    self.assertEqual(expected_ids, ids)

  def testEncodeNoteSequenceAddEos(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108,
        add_eos=True)

    ns = note_seq.NoteSequence()
    testing_lib.add_track_to_sequence(
        ns, 0, [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 127, 1.0, 2.0)])
    ids = encoder.encode_note_sequence(ns)

    expected_ids = [
        302,  # VELOCITY(25)
        41,   # NOTE-ON(60)
        45,   # NOTE-ON(64)
        277,  # TIME-SHIFT(100)
        309,  # VELOCITY(32)
        48,   # NOTE-ON(67)
        277,  # TIME-SHIFT(100)
        136,  # NOTE-OFF(67)
        277,  # TIME-SHIFT(100)
        133,  # NOTE-OFF(64
        277,  # TIME-SHIFT(100)
        129,  # NOTE-OFF(60)
        1     # EOS
    ]

    self.assertEqual(expected_ids, ids)

  def testEncodeNoteSequenceNGrams(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108,
        ngrams=[(41, 45), (277, 309, 300), (309, 48), (277, 129, 130)])

    ns = note_seq.NoteSequence()
    testing_lib.add_track_to_sequence(
        ns, 0, [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 127, 1.0, 2.0)])
    ids = encoder.encode_note_sequence(ns)

    expected_ids = [
        302,  # VELOCITY(25)
        310,  # NOTE-ON(60), NOTE-ON(64)
        277,  # TIME-SHIFT(100)
        312,  # VELOCITY(32), NOTE-ON(67)
        277,  # TIME-SHIFT(100)
        136,  # NOTE-OFF(67)
        277,  # TIME-SHIFT(100)
        133,  # NOTE-OFF(64
        277,  # TIME-SHIFT(100)
        129   # NOTE-OFF(60)
    ]

    self.assertEqual(expected_ids, ids)

  def testEncode(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108,
        ngrams=[(277, 129)])

    ns = note_seq.NoteSequence()
    testing_lib.add_track_to_sequence(ns, 0, [(60, 97, 0.0, 1.0)])

    # Write NoteSequence to MIDI file as encoder takes in filename.
    with tempfile.NamedTemporaryFile(suffix='.mid') as f:
      note_seq.sequence_proto_to_midi_file(ns, f.name)
      ids = encoder.encode(f.name)

    expected_ids = [
        302,  # VELOCITY(25)
        41,   # NOTE-ON(60)
        310   # TIME-SHIFT(100), NOTE-OFF(60)
    ]

    self.assertEqual(expected_ids, ids)

  def testDecode(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108,
        ngrams=[(277, 129)])

    ids = [
        302,  # VELOCITY(25)
        41,   # NOTE-ON(60)
        310   # TIME-SHIFT(100), NOTE-OFF(60)
    ]

    # Decode method returns MIDI filename, read and convert to NoteSequence.
    filename = encoder.decode(ids)
    ns = note_seq.midi_file_to_sequence_proto(filename)

    # Remove default tempo & time signature.
    del ns.tempos[:]
    del ns.time_signatures[:]

    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    testing_lib.add_track_to_sequence(expected_ns, 0, [(60, 97, 0.0, 1.0)])

    # Add source info fields.
    expected_ns.source_info.encoding_type = (
        note_seq.NoteSequence.SourceInfo.MIDI)
    expected_ns.source_info.parser = (
        note_seq.NoteSequence.SourceInfo.PRETTY_MIDI)

    self.assertEqual(expected_ns, ns)

  def testVocabSize(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108)
    self.assertEqual(310, encoder.vocab_size)

  def testVocabSizeNGrams(self):
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100, num_velocity_bins=32, min_pitch=21, max_pitch=108,
        ngrams=[(41, 45), (277, 309, 300), (309, 48), (277, 129, 130)])
    self.assertEqual(314, encoder.vocab_size)


class TextChordsEncoderTest(tf.test.TestCase):

  def testEncodeNoteSequence(self):
    encoder = music_encoders.TextChordsEncoder(steps_per_quarter=1)

    ns = note_seq.NoteSequence()
    ns.tempos.add(qpm=60)
    testing_lib.add_chords_to_sequence(
        ns, [('C', 1), ('Dm', 3), ('Bdim', 4)])
    ns.total_time = 5.0
    ids = encoder.encode_note_sequence(ns)

    expected_ids = [
        2,   # no-chord
        3,   # C major
        3,   # C major
        17,  # D minor
        50   # B diminished
    ]

    self.assertEqual(expected_ids, ids)

  def testEncode(self):
    encoder = music_encoders.TextChordsEncoder(steps_per_quarter=1)

    ids = encoder.encode('C G Am F')
    expected_ids = [
        3,   # C major
        10,  # G major
        24,  # A minor
        8    # F major
    ]

    self.assertEqual(expected_ids, ids)

  def testVocabSize(self):
    encoder = music_encoders.TextChordsEncoder(steps_per_quarter=1)
    self.assertEqual(51, encoder.vocab_size)


class TextMelodyEncoderTest(tf.test.TestCase):

  def testEncodeNoteSequence(self):
    encoder = music_encoders.TextMelodyEncoder(
        steps_per_quarter=4, min_pitch=21, max_pitch=108)
    encoder_absolute = music_encoders.TextMelodyEncoderAbsolute(
        steps_per_second=4, min_pitch=21, max_pitch=108)

    ns = note_seq.NoteSequence()
    ns.tempos.add(qpm=60)
    testing_lib.add_track_to_sequence(
        ns, 0,
        [(60, 127, 0.0, 0.25), (62, 127, 0.25, 0.75), (64, 127, 1.25, 2.0)])
    ids = encoder.encode_note_sequence(ns)
    ids_absolute = encoder_absolute.encode_note_sequence(ns)

    expected_ids = [
        43,  # ON(60)
        45,  # ON(62)
        2,   # HOLD(62)
        3,   # OFF(62)
        2,   # REST
        47,  # ON(64)
        2,   # HOLD(64)
        2    # HOLD(64)
    ]

    self.assertEqual(expected_ids, ids)
    self.assertEqual(expected_ids, ids_absolute)

  def testEncode(self):
    encoder = music_encoders.TextMelodyEncoder(
        steps_per_quarter=4, min_pitch=21, max_pitch=108)

    ids = encoder.encode('60 -2 62 -1 64 -2')
    expected_ids = [
        43,  # ON(60)
        2,   # HOLD(60)
        45,  # ON(62)
        3,   # OFF(62)
        47,  # ON(64)
        2    # HOLD(64)
    ]

    self.assertEqual(expected_ids, ids)

  def testVocabSize(self):
    encoder = music_encoders.TextMelodyEncoder(
        steps_per_quarter=4, min_pitch=21, max_pitch=108)
    self.assertEqual(92, encoder.vocab_size)


class FlattenedTextMelodyEncoderTest(tf.test.TestCase):

  def testEncodeNoteSequence(self):
    encoder = music_encoders.FlattenedTextMelodyEncoderAbsolute(
        steps_per_second=4, num_velocity_bins=127)

    ns = note_seq.NoteSequence()
    ns.tempos.add(qpm=60)
    testing_lib.add_track_to_sequence(
        ns, 0,
        [(60, 127, 0.0, 0.25), (62, 15, 0.25, 0.75), (64, 32, 1.25, 2.0)])
    ids = encoder.encode_note_sequence(ns)
    expected_ids = [
        130,  # ON(vel=127)
        18,  # ON(vel=15)
        2,   # HOLD(62)
        2,   # REST
        2,   # REST
        35,  # ON(vel=32)
        2,   # HOLD(64)
        2    # HOLD(64)
    ]

    self.assertEqual(expected_ids, ids)

  def testVocabSize(self):
    num_vel_bins = 12
    encoder = music_encoders.FlattenedTextMelodyEncoderAbsolute(
        steps_per_second=4, num_velocity_bins=num_vel_bins)
    expected = num_vel_bins + encoder.num_reserved_ids + 2
    self.assertEqual(expected, encoder.vocab_size)


class CompositeScoreEncoderTest(tf.test.TestCase):

  def testEncodeNoteSequence(self):
    encoder = music_encoders.CompositeScoreEncoder([
        music_encoders.TextChordsEncoder(steps_per_quarter=4),
        music_encoders.TextMelodyEncoder(
            steps_per_quarter=4, min_pitch=21, max_pitch=108)
    ])

    ns = note_seq.NoteSequence()
    ns.tempos.add(qpm=60)
    testing_lib.add_chords_to_sequence(ns, [('C', 0.5), ('Dm', 1.0)])
    testing_lib.add_track_to_sequence(
        ns, 0,
        [(60, 127, 0.0, 0.25), (62, 127, 0.25, 0.75), (64, 127, 1.25, 2.0)])
    chord_ids, melody_ids = zip(*encoder.encode_note_sequence(ns))

    expected_chord_ids = [
        2,   # no-chord
        2,   # no-chord
        3,   # C major
        3,   # C major
        17,  # D minor
        17,  # D minor
        17,  # D minor
        17   # D minor
    ]

    expected_melody_ids = [
        43,  # ON(60)
        45,  # ON(62)
        2,   # HOLD(62)
        3,   # OFF(62)
        2,   # REST
        47,  # ON(64)
        2,   # HOLD(64)
        2    # HOLD(64)
    ]

    self.assertEqual(expected_chord_ids, list(chord_ids))
    self.assertEqual(expected_melody_ids, list(melody_ids))

  # TODO(iansimon): also test MusicXML encoding

  def testVocabSize(self):
    encoder = music_encoders.CompositeScoreEncoder([
        music_encoders.TextChordsEncoder(steps_per_quarter=4),
        music_encoders.TextMelodyEncoder(
            steps_per_quarter=4, min_pitch=21, max_pitch=108)
    ])
    self.assertEqual([51, 92], encoder.vocab_size)


if __name__ == '__main__':
  tf.test.main()
