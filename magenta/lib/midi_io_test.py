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
"""Test to ensure correct import of pretty_midi."""

from collections import defaultdict
import os.path
import tempfile

import pretty_midi
import tensorflow as tf

from magenta.lib import midi_io

# self.midi_simple_filename contains a c-major scale of 8 quarter notes each
# with a sustain of .95 of the entire note. Here are the first two notes dumped
# using mididump.py:
#   midi.NoteOnEvent(tick=0, channel=0, data=[60, 100]),
#   midi.NoteOnEvent(tick=209, channel=0, data=[60, 0]),
#   midi.NoteOnEvent(tick=11, channel=0, data=[62, 100]),
#   midi.NoteOnEvent(tick=209, channel=0, data=[62, 0]),
_SIMPLE_MIDI_FILE_VELO = 100
_SIMPLE_MIDI_FILE_NUM_NOTES = 8
_SIMPLE_MIDI_FILE_SUSTAIN = .95

# self.midi_complex_filename contains many instruments including percussion as
# well as control change and pitch bend events.


class MidiIoTest(tf.test.TestCase):

  def setUp(self):
    self.midi_simple_filename = os.path.join(
        tf.resource_loader.get_data_files_path(), '../testdata/example.mid')
    self.midi_complex_filename = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/example_complex.mid')

  def CheckPrettyMidiAndSequence(self, midi, sequence_proto):
    """Compares PrettyMIDI object against a sequence proto.

    Args:
      midi: A pretty_midi.PrettyMIDI object.
      sequence_proto: A tensorflow.magenta.Sequence proto.
    """
    # Test time signature changes.
    self.assertEqual(len(midi.time_signature_changes),
                     len(sequence_proto.time_signatures))
    for midi_time, sequence_time in zip(midi.time_signature_changes,
                                        sequence_proto.time_signatures):
      self.assertEqual(midi_time.numerator, sequence_time.numerator)
      self.assertEqual(midi_time.denominator, sequence_time.denominator)
      self.assertAlmostEqual(midi_time.time, sequence_time.time)

    # Test key signature changes.
    self.assertEqual(len(midi.key_signature_changes),
                     len(sequence_proto.key_signatures))
    for midi_key, sequence_key in zip(midi.key_signature_changes,
                                      sequence_proto.key_signatures):
      self.assertEqual(midi_key.key_number % 12, sequence_key.key)
      self.assertEqual(midi_key.key_number / 12, sequence_key.mode)
      self.assertAlmostEqual(midi_key.time, sequence_key.time)

    # Test tempos.
    midi_times, midi_bpms = midi.get_tempo_changes()
    self.assertEqual(len(midi_times),
                     len(sequence_proto.tempos))
    self.assertEqual(len(midi_bpms),
                     len(sequence_proto.tempos))
    for midi_time, midi_bpm, sequence_tempo in zip(
        midi_times, midi_bpms, sequence_proto.tempos):
      self.assertAlmostEqual(midi_bpm, sequence_tempo.bpm)
      self.assertAlmostEqual(midi_time, sequence_tempo.time)

    # Test instruments.
    seq_instruments = defaultdict(lambda: defaultdict(list))
    for seq_note in sequence_proto.notes:
      seq_instruments[
          (seq_note.instrument, seq_note.program)]['notes'].append(seq_note)
    for seq_bend in sequence_proto.pitch_bends:
      seq_instruments[
          (seq_bend.instrument, seq_bend.program)]['bends'].append(seq_bend)
    for seq_control in sequence_proto.control_changes:
      seq_instruments[
          (seq_control.instrument, seq_control.program)][
              'controls'].append(seq_control)

    sorted_seq_instrument_keys = sorted(
        seq_instruments.keys(),
        key=lambda (instrument_id, program_id): (instrument_id, program_id))

    self.assertEqual(len(midi.instruments), len(seq_instruments))
    for midi_instrument, seq_instrument_key in zip(
        midi.instruments, sorted_seq_instrument_keys):

      seq_instrument_notes = seq_instruments[seq_instrument_key]['notes']

      self.assertEqual(len(midi_instrument.notes), len(seq_instrument_notes))
      for midi_note, sequence_note in zip(midi_instrument.notes,
                                          seq_instrument_notes):
        self.assertEqual(midi_note.pitch, sequence_note.pitch)
        self.assertEqual(midi_note.velocity, sequence_note.velocity)
        self.assertAlmostEqual(midi_note.start, sequence_note.start_time)
        self.assertAlmostEqual(midi_note.end, sequence_note.end_time)

      seq_instrument_pitch_bends = seq_instruments[seq_instrument_key]['bends']
      self.assertEqual(len(midi_instrument.pitch_bends),
                       len(seq_instrument_pitch_bends))
      for midi_pitch_bend, sequence_pitch_bend in zip(
          midi_instrument.pitch_bends,
          seq_instrument_pitch_bends):
        self.assertEqual(midi_pitch_bend.pitch, sequence_pitch_bend.bend)
        self.assertAlmostEqual(midi_pitch_bend.time, sequence_pitch_bend.time)

  def CheckMidiToSequence(self, filename):
    """Test the translation from PrettyMIDI to Sequence proto."""
    source_midi = pretty_midi.PrettyMIDI(filename)
    sequence_proto = midi_io.midi_to_sequence_proto(source_midi)
    self.CheckPrettyMidiAndSequence(source_midi, sequence_proto)

  def CheckSequenceToPrettyMidi(self, filename):
    """Test the translation from Sequence proto to PrettyMIDI."""
    source_midi = pretty_midi.PrettyMIDI(filename)
    sequence_proto = midi_io.midi_to_sequence_proto(source_midi)
    translated_midi = midi_io.sequence_proto_to_pretty_midi(sequence_proto)
    self.CheckPrettyMidiAndSequence(translated_midi, sequence_proto)

  def CheckReadWriteMidi(self, filename):
    """Test writing to a MIDI file and comparing it to the original Sequence."""
    source_midi = pretty_midi.PrettyMIDI(filename)
    sequence_proto = midi_io.midi_to_sequence_proto(source_midi)
    translated_midi = midi_io.sequence_proto_to_pretty_midi(sequence_proto)

    # Write the translated midi to a file.
    with tempfile.NamedTemporaryFile(prefix='MidiIoTest') as temp_file:
      translated_midi.write(temp_file.name)

      # Read it back in and compare to source.
      created_midi = pretty_midi.PrettyMIDI(temp_file.name)
      created_sequence = midi_io.midi_to_sequence_proto(created_midi)

    self.CheckPrettyMidiAndSequence(translated_midi, created_sequence)

  def testSimplePrettyMidiToSequence(self):
    self.CheckMidiToSequence(self.midi_simple_filename)

  def testSimpleSequenceToPrettyMidi(self):
    self.CheckSequenceToPrettyMidi(self.midi_simple_filename)

  def testSimpleReadWriteMidi(self):
    self.CheckReadWriteMidi(self.midi_simple_filename)

  def testComplexPrettyMidiToSequence(self):
    self.CheckMidiToSequence(self.midi_complex_filename)

  def testComplexSequenceToPrettyMidi(self):
    self.CheckSequenceToPrettyMidi(self.midi_complex_filename)

  # TODO(adarob): Uncomment once
  # https://github.com/craffel/pretty-midi/pull/67 is merged.
  # def testComplexReadWriteMidi(self):
  #   self.CheckReadWriteMidi(self.midi_complex_filename)


if __name__ == '__main__':
  tf.test.main()
