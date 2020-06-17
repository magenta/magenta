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

"""Tests for midi_hub."""

import collections
import queue
import threading
import time

from magenta.common import concurrency
from magenta.interfaces.midi import midi_hub
import mido
from note_seq import testing_lib
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf

Note = collections.namedtuple('Note', ['pitch', 'velocity', 'start', 'end'])


class MockMidiPort(mido.ports.BaseIOPort):

  def __init__(self):
    super(MockMidiPort, self).__init__()
    self.message_queue = queue.Queue()

  def send(self, msg):
    msg.time = time.time()
    self.message_queue.put(msg)


class MidiHubTest(tf.test.TestCase):

  def setUp(self):
    self.maxDiff = None  # pylint:disable=invalid-name
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
        mido.Message(type='note_off', note=3, time=100)]

    self.port = MockMidiPort()
    self.midi_hub = midi_hub.MidiHub([self.port], [self.port],
                                     midi_hub.TextureType.POLYPHONIC)

    # Burn in Sleeper for calibration.
    for _ in range(5):
      concurrency.Sleeper().sleep(0.01)

  def tearDown(self):
    self.midi_hub.__del__()

  def send_capture_messages(self):
    for msg in self.capture_messages:
      self.port.callback(msg)

  def testMidiSignal_ValidityChecks(self):
    # Unsupported type.
    with self.assertRaises(midi_hub.MidiHubError):
      midi_hub.MidiSignal(type='sysex')
    with self.assertRaises(midi_hub.MidiHubError):
      midi_hub.MidiSignal(msg=mido.Message(type='sysex'))

    # Invalid arguments.
    with self.assertRaises(midi_hub.MidiHubError):
      midi_hub.MidiSignal()
    with self.assertRaises(midi_hub.MidiHubError):
      midi_hub.MidiSignal(type='note_on', value=1)
    with self.assertRaises(midi_hub.MidiHubError):
      midi_hub.MidiSignal(type='control', note=1)
    with self.assertRaises(midi_hub.MidiHubError):
      midi_hub.MidiSignal(msg=mido.Message(type='control_change'), value=1)

    # Non-inferrale type.
    with self.assertRaises(midi_hub.MidiHubError):
      midi_hub.MidiSignal(note=1, value=1)

  def testMidiSignal_Message(self):
    sig = midi_hub.MidiSignal(msg=mido.Message(type='note_on', note=1))
    self.assertEqual(
        r'^note_on channel=0 note=1 velocity=64 time=\d+.\d+$', str(sig))

    sig = midi_hub.MidiSignal(msg=mido.Message(type='note_off', velocity=127))
    self.assertEqual(r'^note_off channel=0 note=0 velocity=127 time=\d+.\d+$',
                     str(sig))

    sig = midi_hub.MidiSignal(
        msg=mido.Message(type='control_change', control=1, value=2))
    self.assertEqual(
        r'^control_change channel=0 control=1 value=2 time=\d+.\d+$', str(sig))

  def testMidiSignal_Args(self):
    sig = midi_hub.MidiSignal(type='note_on', note=1)
    self.assertEqual(
        r'^note_on channel=\d+ note=1 velocity=\d+ time=\d+.\d+$', str(sig))

    sig = midi_hub.MidiSignal(type='note_off', velocity=127)
    self.assertEqual(
        r'^note_off channel=\d+ note=\d+ velocity=127 time=\d+.\d+$', str(sig))

    sig = midi_hub.MidiSignal(type='control_change', value=2)
    self.assertEqual(
        r'^control_change channel=\d+ control=\d+ value=2 time=\d+.\d+$',
        str(sig))

  def testMidiSignal_Args_InferredType(self):
    sig = midi_hub.MidiSignal(note=1)
    self.assertEqual(
        r'^.* channel=\d+ note=1 velocity=\d+ time=\d+.\d+$', str(sig))

    sig = midi_hub.MidiSignal(value=2)
    self.assertEqual(
        r'^control_change channel=\d+ control=\d+ value=2 time=\d+.\d+$',
        str(sig))

  def testMetronome(self):
    start_time = time.time() + 0.1
    qpm = 180
    self.midi_hub.start_metronome(start_time=start_time, qpm=qpm)
    time.sleep(0.8)

    self.midi_hub.stop_metronome()
    self.assertEqual(7, self.port.message_queue.qsize())

    msg = self.port.message_queue.get()
    self.assertEqual(msg.type, 'program_change')
    next_tick_time = start_time
    while not self.port.message_queue.empty():
      msg = self.port.message_queue.get()
      if self.port.message_queue.qsize() % 2:
        self.assertEqual(msg.type, 'note_on')
        self.assertAlmostEqual(msg.time, next_tick_time, delta=0.01)
        next_tick_time += 60. / qpm
      else:
        self.assertEqual(msg.type, 'note_off')

  def testStartPlayback_NoUpdates(self):
    # Use a time in the past to test handling of past notes.
    start_time = time.time() - 0.01
    seq = music_pb2.NoteSequence()
    notes = [Note(12, 100, 0.0, 1.0), Note(11, 55, 0.1, 0.5),
             Note(40, 45, 0.2, 0.6)]
    notes = [Note(note.pitch, note.velocity, note.start + start_time,
                  note.end + start_time) for note in notes]
    testing_lib.add_track_to_sequence(seq, 0, notes)
    player = self.midi_hub.start_playback(seq, allow_updates=False)
    player.join()

    note_events = []
    for note in notes:
      note_events.append((note.start, 'note_on', note.pitch))
      note_events.append((note.end, 'note_off', note.pitch))

    # The first note on will not be sent since it started before
    # `start_playback` is called.
    del note_events[0]

    note_events = collections.deque(sorted(note_events))
    while not self.port.message_queue.empty():
      msg = self.port.message_queue.get()
      note_event = note_events.popleft()
      self.assertEqual(msg.type, note_event[1])
      self.assertEqual(msg.note, note_event[2])
      self.assertAlmostEqual(msg.time, note_event[0], delta=0.01)

    self.assertFalse(note_events)

  def testStartPlayback_NoUpdates_UpdateError(self):
    # Use a time in the past to test handling of past notes.
    start_time = time.time()
    seq = music_pb2.NoteSequence()
    notes = [Note(0, 100, start_time + 100, start_time + 101)]
    testing_lib.add_track_to_sequence(seq, 0, notes)
    player = self.midi_hub.start_playback(seq, allow_updates=False)

    with self.assertRaises(midi_hub.MidiHubError):
      player.update_sequence(seq)

    player.stop()

  def testStartPlayback_Updates(self):
    start_time = time.time() + 0.1
    seq = music_pb2.NoteSequence()
    notes = [Note(0, 100, start_time, start_time + 101),
             Note(1, 100, start_time, start_time + 101)]
    testing_lib.add_track_to_sequence(seq, 0, notes)
    player = self.midi_hub.start_playback(seq, allow_updates=True)

    # Sleep past first note start.
    concurrency.Sleeper().sleep_until(start_time + 0.2)

    new_seq = music_pb2.NoteSequence()
    notes = [Note(1, 100, 0.0, 0.8), Note(2, 100, 0.0, 1.0),
             Note(11, 55, 0.3, 0.5), Note(40, 45, 0.4, 0.6)]
    notes = [Note(note.pitch, note.velocity, note.start + start_time,
                  note.end + start_time) for note in notes]
    testing_lib.add_track_to_sequence(new_seq, 0, notes)
    player.update_sequence(new_seq)

    # Finish playing sequence.
    concurrency.Sleeper().sleep(0.8)

    # Start and end the unclosed note from the first sequence.
    note_events = [(start_time, 'note_on', 0),
                   (start_time + 0.3, 'note_off', 0)]
    # The second note will not be played since it started before the update
    # and was not in the original sequence.
    del notes[1]
    for note in notes:
      note_events.append((note.start, 'note_on', note.pitch))
      note_events.append((note.end, 'note_off', note.pitch))
    note_events = collections.deque(sorted(note_events))
    while not self.port.message_queue.empty():
      msg = self.port.message_queue.get()
      note_event = note_events.popleft()
      self.assertEqual(msg.type, note_event[1])
      self.assertEqual(msg.note, note_event[2])
      self.assertAlmostEqual(msg.time, note_event[0], delta=0.01)

    self.assertFalse(note_events)
    player.stop()

  def testCaptureSequence_StopSignal(self):
    start_time = 1.0

    threading.Timer(0.1, self.send_capture_messages).start()

    captured_seq = self.midi_hub.capture_sequence(
        120, start_time,
        stop_signal=midi_hub.MidiSignal(type='control_change', control=1))

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 6.0
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seq, expected_seq)

  def testCaptureSequence_StopTime(self):
    start_time = 1.0
    stop_time = time.time() + 1.0

    self.capture_messages[-1].time += time.time()
    threading.Timer(0.1, self.send_capture_messages).start()

    captured_seq = self.midi_hub.capture_sequence(
        120, start_time, stop_time=stop_time)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = stop_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, stop_time)])
    self.assertProtoEquals(captured_seq, expected_seq)

  def testCaptureSequence_Mono(self):
    start_time = 1.0

    threading.Timer(0.1, self.send_capture_messages).start()
    self.midi_hub = midi_hub.MidiHub([self.port], [self.port],
                                     midi_hub.TextureType.MONOPHONIC)
    captured_seq = self.midi_hub.capture_sequence(
        120, start_time,
        stop_signal=midi_hub.MidiSignal(type='control_change', control=1))

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 6
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 3), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seq, expected_seq)

  def testStartCapture_StopMethod(self):
    start_time = 1.0
    captor = self.midi_hub.start_capture(120, start_time)

    self.send_capture_messages()
    time.sleep(0.1)

    stop_time = 5.5
    captor.stop(stop_time=stop_time)

    captured_seq = captor.captured_sequence()
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = stop_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, stop_time)])
    self.assertProtoEquals(captured_seq, expected_seq)

  def testStartCapture_Multiple(self):
    captor_1 = self.midi_hub.start_capture(
        120, 0.0, stop_signal=midi_hub.MidiSignal(note=3))
    captor_2 = self.midi_hub.start_capture(
        120, 1.0,
        stop_signal=midi_hub.MidiSignal(type='control_change', control=1))

    self.send_capture_messages()

    captor_1.join()
    captor_2.join()

    captured_seq_1 = captor_1.captured_sequence()
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 4.0
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(0, 64, 0.01, 3), Note(1, 64, 2, 4), Note(2, 64, 3, 4)])
    self.assertProtoEquals(captured_seq_1, expected_seq)

    captured_seq_2 = captor_2.captured_sequence()
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 6.0
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seq_2, expected_seq)

  def testStartCapture_IsDrum(self):
    start_time = 1.0
    captor = self.midi_hub.start_capture(120, start_time)

    # Channels are 0-indexed in mido.
    self.capture_messages[2].channel = 9
    self.send_capture_messages()
    time.sleep(0.1)

    stop_time = 5.5
    captor.stop(stop_time=stop_time)

    captured_seq = captor.captured_sequence()
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = stop_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, stop_time)])
    expected_seq.notes[0].is_drum = True
    self.assertProtoEquals(captured_seq, expected_seq)

  def testStartCapture_MidCapture(self):
    start_time = 1.0
    captor = self.midi_hub.start_capture(120, start_time)

    # Receive the first 6 messages.
    for msg in self.capture_messages[0:6]:
      self.port.callback(msg)
    time.sleep(0.1)

    end_time = 3.5
    captured_seq = captor.captured_sequence(end_time)
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0, [Note(1, 64, 2, 3.5), Note(2, 64, 3, 3.5)])
    self.assertProtoEquals(captured_seq, expected_seq)

    end_time = 4.5
    captured_seq = captor.captured_sequence(end_time)
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 4.5), Note(2, 64, 3, 4.5), Note(3, 64, 4, 4.5)])
    self.assertProtoEquals(captured_seq, expected_seq)

    end_time = 6.0
    captured_seq = captor.captured_sequence(end_time)
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 6), Note(2, 64, 3, 6), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seq, expected_seq)

    # Receive the rest of the messages.
    for msg in self.capture_messages[6:]:
      self.port.callback(msg)
    time.sleep(0.1)

    end_time = 6.0
    captured_seq = captor.captured_sequence(end_time)
    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seq, expected_seq)

    captor.stop()

  def testStartCapture_Iterate_Signal(self):
    start_time = 1.0
    captor = self.midi_hub.start_capture(
        120, start_time,
        stop_signal=midi_hub.MidiSignal(type='control_change', control=1))

    for msg in self.capture_messages[:-1]:
      threading.Timer(0.2 * msg.time, self.port.callback, args=[msg]).start()

    captured_seqs = []
    for captured_seq in captor.iterate(
        signal=midi_hub.MidiSignal(type='note_off')):
      captured_seqs.append(captured_seq)

    self.assertLen(captured_seqs, 4)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 3
    testing_lib.add_track_to_sequence(expected_seq, 0, [Note(1, 64, 2, 3)])
    self.assertProtoEquals(captured_seqs[0], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 4
    testing_lib.add_track_to_sequence(
        expected_seq, 0, [Note(1, 64, 2, 4), Note(2, 64, 3, 4)])
    self.assertProtoEquals(captured_seqs[1], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 5
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 5)])
    self.assertProtoEquals(captured_seqs[2], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 6
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seqs[3], expected_seq)

  def testStartCapture_Iterate_Period(self):
    start_time = 1.0
    captor = self.midi_hub.start_capture(
        120, start_time,
        stop_signal=midi_hub.MidiSignal(type='control_change', control=1))

    for msg in self.capture_messages[:-1]:
      threading.Timer(0.1 * msg.time, self.port.callback, args=[msg]).start()

    period = 0.26
    captured_seqs = []
    wall_start_time = time.time()
    for captured_seq in captor.iterate(period=period):
      if len(captured_seqs) < 2:
        self.assertAlmostEqual(0, (time.time() - wall_start_time) % period,
                               delta=0.01)
      time.sleep(0.1)
      captured_seqs.append(captured_seq)

    self.assertLen(captured_seqs, 3)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    end_time = captured_seqs[0].total_time
    self.assertAlmostEqual(wall_start_time + period, end_time, delta=0.01)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0, [Note(1, 64, 2, end_time)])
    self.assertProtoEquals(captured_seqs[0], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    end_time = captured_seqs[1].total_time
    self.assertAlmostEqual(wall_start_time + 2 * period, end_time, delta=0.01)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, end_time)])
    self.assertProtoEquals(captured_seqs[1], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 6
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seqs[2], expected_seq)

  def testStartCapture_Iterate_Period_Overrun(self):
    start_time = 1.0
    captor = self.midi_hub.start_capture(
        120, start_time,
        stop_signal=midi_hub.MidiSignal(type='control_change', control=1))

    for msg in self.capture_messages[:-1]:
      threading.Timer(0.1 * msg.time, self.port.callback, args=[msg]).start()

    period = 0.26
    captured_seqs = []
    wall_start_time = time.time()
    for captured_seq in captor.iterate(period=period):
      time.sleep(0.5)
      captured_seqs.append(captured_seq)

    self.assertLen(captured_seqs, 2)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    end_time = captured_seqs[0].total_time
    self.assertAlmostEqual(wall_start_time + period, end_time, delta=0.01)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0, [Note(1, 64, 2, end_time)])
    self.assertProtoEquals(captured_seqs[0], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    expected_seq.total_time = 6
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, 6)])
    self.assertProtoEquals(captured_seqs[1], expected_seq)

  def testStartCapture_Callback_Period(self):
    start_time = 1.0
    captor = self.midi_hub.start_capture(120, start_time)

    for msg in self.capture_messages[:-1]:
      threading.Timer(0.1 * msg.time, self.port.callback, args=[msg]).start()

    period = 0.26
    wall_start_time = time.time()
    captured_seqs = []

    def fn(captured_seq):
      self.assertAlmostEqual(0, (time.time() - wall_start_time) % period,
                             delta=0.01)
      captured_seqs.append(captured_seq)

    name = captor.register_callback(fn, period=period)
    time.sleep(1.0)
    captor.cancel_callback(name)

    self.assertLen(captured_seqs, 3)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    end_time = captured_seqs[0].total_time
    self.assertAlmostEqual(wall_start_time + period, end_time, delta=0.01)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0, [Note(1, 64, 2, end_time)])
    self.assertProtoEquals(captured_seqs[0], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    end_time = captured_seqs[1].total_time
    self.assertAlmostEqual(wall_start_time + 2 * period, end_time, delta=0.01)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, end_time)])
    self.assertProtoEquals(captured_seqs[1], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    end_time = captured_seqs[2].total_time
    self.assertAlmostEqual(wall_start_time + 3 * period, end_time, delta=0.01)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, end_time)])
    self.assertProtoEquals(captured_seqs[2], expected_seq)

  def testStartCapture_Callback_Period_Overrun(self):
    start_time = 1.0
    captor = self.midi_hub.start_capture(
        120, start_time)

    for msg in self.capture_messages[:-1]:
      threading.Timer(0.1 * msg.time, self.port.callback, args=[msg]).start()

    period = 0.26
    wall_start_time = time.time()
    captured_seqs = []

    def fn(captured_seq):
      time.sleep(0.5)
      captured_seqs.append(captured_seq)

    name = captor.register_callback(fn, period=period)
    time.sleep(1.3)
    captor.cancel_callback(name)

    self.assertLen(captured_seqs, 2)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    end_time = captured_seqs[0].total_time
    self.assertAlmostEqual(wall_start_time + period, end_time, delta=0.01)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0, [Note(1, 64, 2, end_time)])
    self.assertProtoEquals(captured_seqs[0], expected_seq)

    expected_seq = music_pb2.NoteSequence()
    expected_seq.tempos.add(qpm=120)
    end_time = captured_seqs[1].total_time
    self.assertAlmostEqual(wall_start_time + 2 * period, end_time, delta=0.01)
    expected_seq.total_time = end_time
    testing_lib.add_track_to_sequence(
        expected_seq, 0,
        [Note(1, 64, 2, 5), Note(2, 64, 3, 4), Note(3, 64, 4, end_time)])
    self.assertProtoEquals(captured_seqs[1], expected_seq)

  def testPassThrough_Poly(self):
    self.midi_hub.passthrough = False
    self.send_capture_messages()
    self.assertTrue(self.port.message_queue.empty())
    self.midi_hub.passthrough = True
    self.send_capture_messages()

    passed_messages = []
    while not self.port.message_queue.empty():
      passed_messages.append(self.port.message_queue.get().bytes())
    self.assertListEqual(
        passed_messages, [m.bytes() for m in self.capture_messages])

  def testPassThrough_Mono(self):
    self.midi_hub = midi_hub.MidiHub([self.port], [self.port],
                                     midi_hub.TextureType.MONOPHONIC)
    self.midi_hub.passthrough = False
    self.send_capture_messages()
    self.assertTrue(self.port.message_queue.empty())
    self.midi_hub.passthrough = True
    self.send_capture_messages()

    passed_messages = []
    while not self.port.message_queue.empty():
      passed_messages.append(self.port.message_queue.get())
      passed_messages[-1].time = 0
    expected_messages = [
        mido.Message(type='note_on', note=0),
        mido.Message(type='control_change', control=1, value=1),
        mido.Message(type='note_off', note=0),
        mido.Message(type='note_on', note=1),
        mido.Message(type='note_off', note=1),
        mido.Message(type='note_on', note=2),
        mido.Message(type='note_off', note=2),
        mido.Message(type='note_on', note=3),
        mido.Message(type='control_change', control=1, value=1),
        mido.Message(type='note_off', note=3)]

    self.assertListEqual(passed_messages, expected_messages)

  def testWaitForEvent_Signal(self):
    for msg in self.capture_messages[3:-1]:
      threading.Timer(0.2 * msg.time, self.port.callback, args=[msg]).start()

    wait_start = time.time()

    self.midi_hub.wait_for_event(
        signal=midi_hub.MidiSignal(type='control_change', value=1))
    self.assertAlmostEqual(time.time() - wait_start, 1.2, delta=0.01)

  def testWaitForEvent_Time(self):
    for msg in self.capture_messages[3:-1]:
      threading.Timer(0.1 * msg.time, self.port.callback, args=[msg]).start()

    wait_start = time.time()

    self.midi_hub.wait_for_event(timeout=0.3)
    self.assertAlmostEqual(time.time() - wait_start, 0.3, delta=0.01)

  def testSendControlChange(self):
    self.midi_hub.send_control_change(0, 1)

    sent_messages = []
    while not self.port.message_queue.empty():
      sent_messages.append(self.port.message_queue.get())

    self.assertListEqual(
        sent_messages,
        [mido.Message(type='control_change', control=0, value=1,
                      time=sent_messages[0].time)])

if __name__ == '__main__':
  tf.test.main()
