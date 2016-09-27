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
"""Tests for midi_hub."""

import collections
import mido
import Queue
import time

# internal imports
import tensorflow as tf

from magenta.interfaces.midi import midi_hub
from magenta.lib import testing_lib
from magenta.protobuf import music_pb2


Note = collections.namedtuple('Note', ['pitch', 'velocity', 'start', 'end'])


class MockMidiPort(mido.ports.BaseIOPort):
  def __init__(self):
    self.message_queue = Queue.Queue()
    super(MockMidiPort, self).__init__()

  def _send(self, msg):
    msg.time = time.time()
    self.message_queue.put(msg)

  def _receive(self, msg, block):
    return self.message_queue.get(block=block)


class MidiHubTest(tf.test.TestCase):
  def setUp(self):
    self.capture_messages = [
        mido.Message(type='note_on', note=0, time=0.01),
        mido.Message(type='control_change', control=1, value=1, time=0.02),
        mido.Message(type='note_on', note=1, time=2.0),
        mido.Message(type='note_off', note=0, time=3.0),
        mido.Message(type='note_on', note=2, time=3.0),
        mido.Message(type='note_on', note=3, time=4.0),
        mido.Message(type='note_off', note=2, time=4.0),
        mido.Message(type='note_off', note=1, time=5.0),
        mido.Message(type='control_change', control=1, value=1, time=6.0),
        mido.Message(type='note_off', note=3, time=time.time() + 100)]

  def testMidiSignal_ValidityChecks(self):
    # Unsupported type.
    with self.assertRaises(midi_hub.MidiHubException):
      midi_hub.MidiSignal(type='sysex')
    with self.assertRaises(midi_hub.MidiHubException):
      midi_hub.MidiSignal(msg=mido.Message(type='sysex'))

    # Invalid arguments.
    with self.assertRaises(midi_hub.MidiHubException):
      midi_hub.MidiSignal(type='note_on', value=1)
    with self.assertRaises(midi_hub.MidiHubException):
      midi_hub.MidiSignal(type='control', note=1)
    with self.assertRaises(midi_hub.MidiHubException):
      midi_hub.MidiSignal(msg=mido.Message(type='control_change'), value=1)

    # Non-inferrale type.
    with self.assertRaises(midi_hub.MidiHubException):
      midi_hub.MidiSignal(note=1, value=1)

  def testMidiSignal_Message(self):
    sig = midi_hub.MidiSignal(msg=mido.Message(type='note_on', note=1))
    self.assertEquals(r'^note_on channel=0 note=1 velocity=64 time=\d+.\d+$',
                      str(sig))

    sig = midi_hub.MidiSignal(msg=mido.Message(type='note_off', velocity=127))
    self.assertEquals(r'^note_off channel=0 note=0 velocity=127 time=\d+.\d+$',
                      str(sig))

    sig = midi_hub.MidiSignal(
        msg=mido.Message(type='control_change', control=1, value=2))
    self.assertEquals(
        r'^control_change channel=0 control=1 value=2 time=\d+.\d+$', str(sig))

  def testMidiSignal_Args(self):
    sig = midi_hub.MidiSignal(type='note_on', note=1)
    self.assertEquals(r'^note_on channel=\d+ note=1 velocity=\d+ time=\d+.\d+$',
                      str(sig))

    sig = midi_hub.MidiSignal(type='note_off', velocity=127)
    self.assertEquals(
        r'^note_off channel=\d+ note=\d+ velocity=127 time=\d+.\d+$', str(sig))

    sig = midi_hub.MidiSignal(type='control_change', value=2)
    self.assertEquals(
        r'^control_change channel=\d+ control=\d+ value=2 time=\d+.\d+$',
        str(sig))

  def testMidiSignal_Args_InferredType(self):
    sig = midi_hub.MidiSignal(note=1)
    self.assertEquals(r'^.* channel=\d+ note=1 velocity=\d+ time=\d+.\d+$',
                      str(sig))

    sig = midi_hub.MidiSignal(value=2)
    self.assertEquals(
        r'^control_change channel=\d+ control=\d+ value=2 time=\d+.\d+$',
        str(sig))

  def testMidiPlayer_NoUpdates(self):
    # Use a time in the past to test handling of past notes.
    start_time = time.time() - 0.05
    outport = MockMidiPort()
    seq = music_pb2.NoteSequence()
    notes = [Note(12, 100, 0.0, 1.0), Note(11, 55, 0.1, 0.5),
             Note(40, 45, 0.2, 0.6)]
    notes = [Note(note.pitch, note.velocity, note.start + start_time,
                  note.end + start_time) for note in notes]
    testing_lib.add_track(seq, 0, notes)
    player = midi_hub.MidiPlayer(outport, seq, allow_updates=False)
    player.start()
    player.join()
    note_events = []
    for note in notes:
      note_events.append((note.start, 'note_on', note.pitch))
      note_events.append((note.end, 'note_off', note.pitch))
    note_events = collections.deque(sorted(note_events))
    # The first event occured in the past.
    note_events.popleft()
    while not outport.message_queue.empty():
      msg = outport.message_queue.get()
      note_event = note_events.popleft()
      self.assertEquals(msg.type, note_event[1])
      self.assertEquals(msg.note, note_event[2])
      self.assertAlmostEqual(msg.time, note_event[0], delta=0.01)

  def testMidiPlayer_NoUpdates_UpdateException(self):
    # Use a time in the past to test handling of past notes.
    start_time = time.time()
    outport = MockMidiPort()
    seq = music_pb2.NoteSequence()
    notes = [Note(0, 100, start_time + 100, start_time + 101)]
    testing_lib.add_track(seq, 0, notes)
    player = midi_hub.MidiPlayer(outport, seq, allow_updates=False)
    player.start()

    with self.assertRaises(midi_hub.MidiHubException):
      player.update_sequence(seq)

    player.stop()

  def testMidiPlayer_Updates(self):
    start_time = time.time() + 0.1
    outport = MockMidiPort()
    seq = music_pb2.NoteSequence()
    notes = [Note(0, 100, start_time, start_time + 101),
             Note(1, 100, start_time, start_time + 101)]
    testing_lib.add_track(seq, 0, notes)
    player = midi_hub.MidiPlayer(outport, seq, allow_updates=True)
    player.start()

    # Sleep past first note start.
    time.sleep(0.2)

    new_seq = music_pb2.NoteSequence()
    notes = [Note(1, 100, 0.0, 1.0), Note(11, 55, 0.2, 0.5),
             Note(40, 45, 0.3, 0.6)]
    notes = [Note(note.pitch, note.velocity, note.start + start_time,
                  note.end + start_time) for note in notes]
    testing_lib.add_track(new_seq, 0, notes)
    player.update_sequence(new_seq)

    # Start and end the unclosed note from the first sequence.
    note_events = [(start_time, 'note_on', 0),
                   (start_time + 0.1, 'note_off', 0)]
    for note in notes:
      note_events.append((note.start, 'note_on', note.pitch))
      note_events.append((note.end, 'note_off', note.pitch))
    note_events = collections.deque(sorted(note_events))
    while not outport.message_queue.empty():
      msg = outport.message_queue.get()
      note_event = note_events.popleft()
      self.assertEquals(msg.type, note_event[1])
      self.assertEquals(msg.note, note_event[2])
      self.assertAlmostEqual(msg.time, note_event[0], delta=0.01)

  def testPolyphonicMidiCaptor_StopSignal(self):
    start_time = 1.0
    captor = midi_hub.PolyphonicMidiCaptor(
        120, start_time,
        stop_signal=midi_hub.MidiSignal(type='control_change', control=1))
    captor.start()

    for msg in self.capture_messages:
      captor.receive(msg)
    captor.join()

    captured_seq = captor.captured_sequence()
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 6.0
    testing_lib.add_track(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seq, expected_seq)

  def testPolyphonicMidiCaptor_StopTime(self):
    start_time = 1.0
    stop_time = time.time() + 1.0
    captor = midi_hub.PolyphonicMidiCaptor(120, start_time, stop_time=stop_time)
    captor.start()

    for msg in self.capture_messages:
      captor.receive(msg)
    captor.join()

    captured_seq = captor.captured_sequence()
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = stop_time
    testing_lib.add_track(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, stop_time)])
    self.assertProtoEquals(captured_seq, expected_seq)

  def testPolyphonicMidiCaptor_StopMethod(self):
    start_time = 1.0
    captor = midi_hub.PolyphonicMidiCaptor(120, start_time)
    captor.start()

    for msg in self.capture_messages:
      captor.receive(msg)
    time.sleep(0.1)

    stop_time = 5.5
    captor.stop(stop_time=stop_time)

    captured_seq = captor.captured_sequence()
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = stop_time
    testing_lib.add_track(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, stop_time)])
    self.assertProtoEquals(captured_seq, expected_seq)

  def testPolyphonicMidiCaptor_MidCapture(self):
    start_time = 1.0
    captor = midi_hub.PolyphonicMidiCaptor(120, start_time)
    captor.start()

    # Recieve first 6 messages.
    for msg in self.capture_messages[0:6]:
      captor.receive(msg)
    time.sleep(0.1)

    end_time = 3.5
    captured_seq = captor.captured_sequence(end_time)
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = end_time
    testing_lib.add_track(
        expected_seq, 0, [Note(1, 64, 2, 3.5), Note(2, 64, 3, 3.5)])
    self.assertProtoEquals(captured_seq, expected_seq)

    end_time = 4.5
    captured_seq = captor.captured_sequence(end_time)
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = end_time
    testing_lib.add_track(
        expected_seq, 0, [Note(1, 64, 2, 4.5), Note(2, 64, 3, 4.5),
        Note(3, 64, 4, 4.5)])
    self.assertProtoEquals(captured_seq, expected_seq)

    end_time = 6.0
    captured_seq = captor.captured_sequence(end_time)
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = end_time
    testing_lib.add_track(
        expected_seq, 0,
        [Note(1, 64, 2, 6), Note(2, 64, 3, 6), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seq, expected_seq)

    # Recieve the rest of the messages.
    for msg in self.capture_messages[6:]:
      captor.receive(msg)
    time.sleep(0.1)

    end_time = 6.0
    captured_seq = captor.captured_sequence(end_time)
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = end_time
    testing_lib.add_track(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seq, expected_seq)

    captor.stop()


if __name__ == '__main__':
  tf.test.main()
