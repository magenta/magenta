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

"""A module for interfacing with the MIDI environment."""

# TODO(adarob): Use flattened imports.

import abc
import collections
import queue
import re
import threading
import time

from magenta.common import concurrency
import mido
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf

_DEFAULT_METRONOME_TICK_DURATION = 0.05
_DEFAULT_METRONOME_PROGRAM = 117  # Melodic Tom
_DEFAULT_METRONOME_MESSAGES = [
    mido.Message(type='note_on', note=44, velocity=64),
    mido.Message(type='note_on', note=35, velocity=64),
    mido.Message(type='note_on', note=35, velocity=64),
    mido.Message(type='note_on', note=35, velocity=64),
]
_DEFAULT_METRONOME_CHANNEL = 1

# 0-indexed.
_DRUM_CHANNEL = 9

try:
  # The RtMidi backend is easier to install and has support for virtual ports.
  import rtmidi  # pylint: disable=unused-import,g-import-not-at-top
  mido.set_backend('mido.backends.rtmidi')
except ImportError:
  # Tries to use PortMidi backend by default.
  tf.logging.warn('Could not import RtMidi. Virtual ports are disabled.')


class MidiHubError(Exception):  # pylint:disable=g-bad-exception-name
  """Base class for exceptions in this module."""
  pass


def get_available_input_ports():
  """Returns a list of available input MIDI ports."""
  return mido.get_input_names()


def get_available_output_ports():
  """Returns a list of available output MIDI ports."""
  return mido.get_output_names()


class MidiSignal(object):
  """A class for representing a MIDI-based event signal.

  Provides a `__str__` method to return a regular expression pattern for
  matching against the string representation of a mido.Message with wildcards
  for unspecified values.

  Supports matching for message types 'note_on', 'note_off', and
  'control_change'. If a mido.Message is given as the `msg` argument, matches
  against the exact message, ignoring the time attribute. If a `msg` is
  not given, keyword arguments must be provided matching some non-empty subset
  of those listed as a value for at least one key in `_VALID_ARGS`.

  Examples:
    # A signal that matches any 'note_on' message.
    note_on_signal = MidiSignal(type='note_on')

    # A signal that matches any 'note_on' or 'note_off' message with a pitch
    # value of 4 and a velocity of 127.
    note_signal = MidiSignal(note=4, velocity=127)

    # A signal that matches a specific mido.Message exactly (ignoring time).
    msg = mido.Message(type='control_signal', control=1, value=127)
    control_1_127_signal = MidiSignal(msg=msg)

  Args:
    msg: A mido.Message that should be matched exactly (excluding the time
        attribute) or None if wildcards are to be used.
    **kwargs: Valid mido.Message arguments. Those that are not provided will be
        treated as wildcards.

  Raises:
    MidiHubError: If the message type is unsupported or the arguments are
        not in the valid set for the given or inferred type.
  """
  _NOTE_ARGS = set(['type', 'note', 'program_number', 'velocity'])
  _CONTROL_ARGS = set(['type', 'control', 'value'])
  _VALID_ARGS = {
      'note_on': _NOTE_ARGS,
      'note_off': _NOTE_ARGS,
      'control_change': _CONTROL_ARGS,
  }

  def __init__(self, msg=None, **kwargs):
    if msg is not None and kwargs:
      raise MidiHubError(
          'Either a mido.Message should be provided or arguments. Not both.')

    type_ = msg.type if msg is not None else kwargs.get('type')
    if 'type' in kwargs:
      del kwargs['type']

    if type_ is not None and type_ not in self._VALID_ARGS:
      raise MidiHubError(
          "The type of a MidiSignal must be either 'note_on', 'note_off', "
          "'control_change' or None for wildcard matching. Got '%s'." % type_)

    # The compatible mido.Message types.
    inferred_types = [type_] if type_ is not None else []
    # If msg is not provided, check that the given arguments are valid for some
    # message type.
    if msg is None:
      if type_ is not None:
        for arg_name in kwargs:
          if arg_name not in self._VALID_ARGS[type_]:
            raise MidiHubError(
                "Invalid argument for type '%s': %s" % (type_, arg_name))
      else:
        if kwargs:
          for name, args in self._VALID_ARGS.items():
            if set(kwargs) <= args:
              inferred_types.append(name)
        if not inferred_types:
          raise MidiHubError(
              'Could not infer a message type for set of given arguments: %s'
              % ', '.join(kwargs))
        # If there is only a single valid inferred type, use it.
        if len(inferred_types) == 1:
          type_ = inferred_types[0]

    self._msg = msg
    self._kwargs = kwargs
    self._type = type_
    self._inferred_types = inferred_types

  def to_message(self):
    """Returns a message using the signal's specifications, if possible."""
    if self._msg:
      return self._msg
    if not self._type:
      raise MidiHubError('Cannot build message if type is not inferrable.')
    return mido.Message(self._type, **self._kwargs)

  def __str__(self):
    """Returns a regex pattern for matching against a mido.Message string."""
    if self._msg is not None:
      regex_pattern = '^' + mido.messages.format_as_string(
          self._msg, include_time=False) + r' time=\d+.\d+$'
    else:
      # Generate regex pattern.
      parts = ['.*' if self._type is None else self._type]
      for name in mido.messages.SPEC_BY_TYPE[self._inferred_types[0]][
          'value_names']:
        if name in self._kwargs:
          parts.append('%s=%d' % (name, self._kwargs[name]))
        else:
          parts.append(r'%s=\d+' % name)
      regex_pattern = '^' + ' '.join(parts) + r' time=\d+.\d+$'
    return regex_pattern


class Metronome(threading.Thread):
  """A thread implementing a MIDI metronome.

  Args:
    outport: The Mido port for sending messages.
    qpm: The integer quarters per minute to signal on.
    start_time: The float wall time in seconds to treat as the first beat
        for alignment. If in the future, the first tick will not start until
        after this time.
    stop_time: The float wall time in seconds after which the metronome should
        stop, or None if it should continue until `stop` is called.
    program: The MIDI program number to use for metronome ticks.
    signals: An ordered collection of MidiSignals whose underlying messages are
        to be output on the metronome's tick, cyclically. A None value can be
        used in place of a MidiSignal to output nothing on a given tick.
    duration: The duration of the metronome's tick.
    channel: The MIDI channel to output on.
  """
  daemon = True

  def __init__(self,
               outport,
               qpm,
               start_time,
               stop_time=None,
               program=_DEFAULT_METRONOME_PROGRAM,
               signals=None,
               duration=_DEFAULT_METRONOME_TICK_DURATION,
               channel=None):
    self._outport = outport
    self.update(
        qpm, start_time, stop_time, program, signals, duration, channel)
    super(Metronome, self).__init__()

  def update(self,
             qpm,
             start_time,
             stop_time=None,
             program=_DEFAULT_METRONOME_PROGRAM,
             signals=None,
             duration=_DEFAULT_METRONOME_TICK_DURATION,
             channel=None):
    """Updates Metronome options."""
    # Locking is not required since variables are independent and assignment is
    # atomic.
    self._channel = _DEFAULT_METRONOME_CHANNEL if channel is None else channel

    # Set the program number for the channels.
    self._outport.send(
        mido.Message(
            type='program_change', program=program, channel=self._channel))
    self._period = 60. / qpm
    self._start_time = start_time
    self._stop_time = stop_time
    if signals is None:
      self._messages = _DEFAULT_METRONOME_MESSAGES
    else:
      self._messages = [s.to_message() if s else None for s in signals]
    self._duration = duration

  def run(self):
    """Sends message on the qpm interval until stop signal received."""
    sleeper = concurrency.Sleeper()
    while True:
      now = time.time()
      tick_number = max(0, int((now - self._start_time) // self._period) + 1)
      tick_time = tick_number * self._period + self._start_time

      if self._stop_time is not None and self._stop_time < tick_time:
        break

      sleeper.sleep_until(tick_time)

      metric_position = tick_number % len(self._messages)
      tick_message = self._messages[metric_position]

      if tick_message is None:
        continue

      tick_message.channel = self._channel
      self._outport.send(tick_message)

      if tick_message.type == 'note_on':
        sleeper.sleep(self._duration)
        end_tick_message = mido.Message(
            'note_off', note=tick_message.note, channel=self._channel)
        self._outport.send(end_tick_message)

  def stop(self, stop_time=0, block=True):
    """Signals for the metronome to stop.

    Args:
      stop_time: The float wall time in seconds after which the metronome should
          stop. By default, stops at next tick.
      block: If true, blocks until thread terminates.
    """
    self._stop_time = stop_time
    if block:
      self.join()


class MidiPlayer(threading.Thread):
  """A thread for playing back a NoteSequence proto via MIDI.

  The NoteSequence times must be based on the wall time. The playhead matches
  the wall clock. The playback sequence may be updated at any time if
  `allow_updates` is set to True.

  Args:
    outport: The Mido port for sending messages.
    sequence: The NoteSequence to play.
    start_time: The float time before which to strip events. Defaults to
        construction time. Events before this time will be sent immediately on
        start.
    allow_updates: If False, the thread will terminate after playback of
        `sequence` completes and calling `update_sequence` will result in an
        exception. Otherwise, the the thread will stay alive until `stop` is
        called, allowing for additional updates via `update_sequence`.
    channel: The MIDI channel to send playback events.
    offset: The float time in seconds to adjust the playback event times by.
  """

  def __init__(self, outport, sequence, start_time=time.time(),
               allow_updates=False, channel=0, offset=0.0):
    self._outport = outport
    self._channel = channel
    self._offset = offset

    # Set of notes (pitches) that are currently on.
    self._open_notes = set()
    # Lock for serialization.
    self._lock = threading.RLock()
    # A control variable to signal when the sequence has been updated.
    self._update_cv = threading.Condition(self._lock)
    # The queue of mido.Message objects to send, sorted by ascending time.
    self._message_queue = collections.deque()
    # An event that is set when `stop` has been called.
    self._stop_signal = threading.Event()

    # Initialize message queue.
    # We first have to allow "updates" to set the initial sequence.
    self._allow_updates = True
    self.update_sequence(sequence, start_time=start_time)
    # We now make whether we allow updates dependent on the argument.
    self._allow_updates = allow_updates

    super(MidiPlayer, self).__init__()

  @concurrency.serialized
  def update_sequence(self, sequence, start_time=None):
    """Updates sequence being played by the MidiPlayer.

    Adds events to close any notes that are no longer being closed by the
    new sequence using the times when they would have been closed by the
    previous sequence.

    Args:
      sequence: The NoteSequence to play back.
      start_time: The float time before which to strip events. Defaults to call
          time.
    Raises:
      MidiHubError: If called when _allow_updates is False.
    """
    if start_time is None:
      start_time = time.time()

    if not self._allow_updates:
      raise MidiHubError(
          'Attempted to update a MidiPlayer sequence with updates disabled.')

    new_message_list = []
    # The set of pitches that are already playing and will be closed without
    # first being reopened in in the new sequence.
    closed_notes = set()
    for note in sequence.notes:
      if note.start_time >= start_time:
        new_message_list.append(
            mido.Message(type='note_on', note=note.pitch,
                         velocity=note.velocity, time=note.start_time))
        new_message_list.append(
            mido.Message(type='note_off', note=note.pitch, time=note.end_time))
      elif note.end_time >= start_time and note.pitch in self._open_notes:
        new_message_list.append(
            mido.Message(type='note_off', note=note.pitch, time=note.end_time))
        closed_notes.add(note.pitch)

    # Close remaining open notes at the next event time to avoid abruptly ending
    # notes.
    notes_to_close = self._open_notes - closed_notes
    if notes_to_close:
      next_event_time = (
          min(msg.time for msg in new_message_list) if new_message_list else 0)
      for note in notes_to_close:
        new_message_list.append(
            mido.Message(type='note_off', note=note, time=next_event_time))

    for msg in new_message_list:
      msg.channel = self._channel
      msg.time += self._offset

    self._message_queue = collections.deque(
        sorted(new_message_list, key=lambda msg: (msg.time, msg.note)))
    self._update_cv.notify()

  @concurrency.serialized
  def run(self):
    """Plays messages in the queue until empty and _allow_updates is False."""
    # Assumes model where NoteSequence is time-stamped with wall time.
    # TODO(hanzorama): Argument to allow initial start not at sequence start?

    while self._message_queue and self._message_queue[0].time < time.time():
      self._message_queue.popleft()

    while True:
      while self._message_queue:
        delta = self._message_queue[0].time - time.time()
        if delta > 0:
          self._update_cv.wait(timeout=delta)
        else:
          msg = self._message_queue.popleft()
          if msg.type == 'note_on':
            self._open_notes.add(msg.note)
          elif msg.type == 'note_off':
            self._open_notes.discard(msg.note)
          self._outport.send(msg)

      # Either keep player alive and wait for sequence update, or return.
      if self._allow_updates:
        self._update_cv.wait()
      else:
        break

  def stop(self, block=True):
    """Signals for the playback to stop and ends all open notes.

    Args:
      block: If true, blocks until thread terminates.
    """
    with self._lock:
      if not self._stop_signal.is_set():
        self._stop_signal.set()
        self._allow_updates = False

        # Replace message queue with immediate end of open notes.
        self._message_queue.clear()
        for note in self._open_notes:
          self._message_queue.append(
              mido.Message(type='note_off', note=note, time=time.time()))
        self._update_cv.notify()
    if block:
      self.join()


class MidiCaptor(threading.Thread):
  """Base class for thread that captures MIDI into a NoteSequence proto.

  If neither `stop_time` nor `stop_signal` are provided as arguments, the
  capture will continue until the `stop` method is called.

  Args:
    qpm: The quarters per minute to use for the captured sequence.
    start_time: The float wall time in seconds when the capture begins. Events
        occuring before this time are ignored.
    stop_time: The float wall time in seconds when the capture is to be stopped
        or None.
    stop_signal: A MidiSignal to use as a signal to stop capture.
  """
  _metaclass__ = abc.ABCMeta

  # A message that is used to wake the consumer thread.
  _WAKE_MESSAGE = None

  def __init__(self, qpm, start_time=0, stop_time=None, stop_signal=None):
    # A lock for synchronization.
    self._lock = threading.RLock()
    self._receive_queue = queue.Queue()
    self._captured_sequence = music_pb2.NoteSequence()
    self._captured_sequence.tempos.add(qpm=qpm)
    self._start_time = start_time
    self._stop_time = stop_time
    self._stop_regex = re.compile(str(stop_signal))
    # A set of active MidiSignals being used by iterators.
    self._iter_signals = []
    # An event that is set when `stop` has been called.
    self._stop_signal = threading.Event()
    # Active callback threads keyed by unique thread name.
    self._callbacks = {}
    super(MidiCaptor, self).__init__()

  @property
  @concurrency.serialized
  def start_time(self):
    return self._start_time

  @start_time.setter
  @concurrency.serialized
  def start_time(self, value):
    """Updates the start time, removing any notes that started before it."""
    self._start_time = value
    i = 0
    for note in self._captured_sequence.notes:
      if note.start_time >= self._start_time:
        break
      i += 1
    del self._captured_sequence.notes[:i]

  @property
  @concurrency.serialized
  def _stop_time(self):
    return self._stop_time_unsafe

  @_stop_time.setter
  @concurrency.serialized
  def _stop_time(self, value):
    self._stop_time_unsafe = value

  def receive(self, msg):
    """Adds received mido.Message to the queue for capture.

    Args:
      msg: The incoming mido.Message object to add to the queue for capture. The
           time attribute is assumed to be pre-set with the wall time when the
           message was received.
    Raises:
      MidiHubError: When the received message has an empty time attribute.
    """
    if not msg.time:
      raise MidiHubError(
          'MidiCaptor received message with empty time attribute: %s' % msg)
    self._receive_queue.put(msg)

  @abc.abstractmethod
  def _capture_message(self, msg):
    """Handles a single incoming MIDI message during capture.

    Must be serialized in children.

    Args:
      msg: The incoming mido.Message object to capture. The time field is
           assumed to be pre-filled with the wall time when the message was
           received.
    """
    pass

  def _add_note(self, msg):
    """Adds and returns a new open note based on the MIDI message."""
    new_note = self._captured_sequence.notes.add()
    new_note.start_time = msg.time
    new_note.pitch = msg.note
    new_note.velocity = msg.velocity
    new_note.is_drum = (msg.channel == _DRUM_CHANNEL)
    return new_note

  def run(self):
    """Captures incoming messages until stop time or signal received."""
    while True:
      timeout = None
      stop_time = self._stop_time
      if stop_time is not None:
        timeout = stop_time - time.time()
        if timeout <= 0:
          break
      try:
        msg = self._receive_queue.get(block=True, timeout=timeout)
      except queue.Empty:
        continue

      if msg is MidiCaptor._WAKE_MESSAGE:
        continue

      if msg.time <= self._start_time:
        continue

      if self._stop_regex.match(str(msg)) is not None:
        break

      with self._lock:
        msg_str = str(msg)
        for regex, queue_ in self._iter_signals:
          if regex.match(msg_str) is not None:
            queue_.put(msg.copy())

      self._capture_message(msg)

    stop_time = self._stop_time
    end_time = stop_time if stop_time is not None else msg.time

    # Acquire lock to avoid race condition with `iterate`.
    with self._lock:
      # Set final captured sequence.
      self._captured_sequence = self.captured_sequence(end_time)
      # Wake up all generators.
      for regex, queue_ in self._iter_signals:
        queue_.put(MidiCaptor._WAKE_MESSAGE)

  def stop(self, stop_time=None, block=True):
    """Ends capture and truncates the captured sequence at `stop_time`.

    Args:
      stop_time: The float time in seconds to stop the capture, or None if it
         should be stopped now. May be in the past, in which case the captured
         sequence will be truncated appropriately.
      block: If True, blocks until the thread terminates.
    Raises:
      MidiHubError: When called multiple times with a `stop_time`.
    """
    with self._lock:
      if self._stop_signal.is_set():
        if stop_time is not None:
          raise MidiHubError(
              '`stop` must not be called multiple times with a `stop_time` on '
              'MidiCaptor.')
      else:
        self._stop_signal.set()
        self._stop_time = time.time() if stop_time is None else stop_time
        # Force the thread to wake since we've updated the stop time.
        self._receive_queue.put(MidiCaptor._WAKE_MESSAGE)
    if block:
      self.join()

  def captured_sequence(self, end_time=None):
    """Returns a copy of the current captured sequence.

    If called before the thread terminates, `end_time` is required and any open
    notes will have their end time set to it, any notes starting after it will
    be removed, and any notes ending after it will be truncated. `total_time`
    will also be set to `end_time`.

    Args:
      end_time: The float time in seconds to close any open notes and after
          which to close or truncate notes, if the thread is still alive.
          Otherwise, must be None.

    Returns:
      A copy of the current captured NoteSequence proto with open notes closed
      at and later notes removed or truncated to `end_time`.

    Raises:
      MidiHubError: When the thread is alive and `end_time` is None or the
         thread is terminated and `end_time` is not None.
    """
    # Make a copy of the sequence currently being captured.
    current_captured_sequence = music_pb2.NoteSequence()
    with self._lock:
      current_captured_sequence.CopyFrom(self._captured_sequence)

    if self.is_alive():
      if end_time is None:
        raise MidiHubError(
            '`end_time` must be provided when capture thread is still running.')
      for i, note in enumerate(current_captured_sequence.notes):
        if note.start_time >= end_time:
          del current_captured_sequence.notes[i:]
          break
        if not note.end_time or note.end_time > end_time:
          note.end_time = end_time
      current_captured_sequence.total_time = end_time
    elif end_time is not None:
      raise MidiHubError(
          '`end_time` must not be provided when capture is complete.')

    return current_captured_sequence

  def iterate(self, signal=None, period=None):
    """Yields the captured sequence at every signal message or time period.

    Exactly one of `signal` or `period` must be specified. Continues until the
    captor terminates, at which point the final captured sequence is yielded
    before returning.

    If consecutive calls to iterate are longer than the period, immediately
    yields and logs a warning.

    Args:
      signal: A MidiSignal to use as a signal to yield, or None.
      period: A float period in seconds, or None.

    Yields:
      The captured NoteSequence at event time.

    Raises:
      MidiHubError: If neither `signal` nor `period` or both are specified.
    """
    if (signal, period).count(None) != 1:
      raise MidiHubError(
          'Exactly one of `signal` or `period` must be provided to `iterate` '
          'call.')

    if signal is None:
      sleeper = concurrency.Sleeper()
      next_yield_time = time.time() + period
    else:
      regex = re.compile(str(signal))
      capture_queue = queue.Queue()
      with self._lock:
        self._iter_signals.append((regex, capture_queue))

    while self.is_alive():
      if signal is None:
        skipped_periods = (time.time() - next_yield_time) // period
        if skipped_periods > 0:
          tf.logging.warn(
              'Skipping %d %.3fs period(s) to catch up on iteration.',
              skipped_periods, period)
          next_yield_time += skipped_periods * period
        else:
          sleeper.sleep_until(next_yield_time)
        end_time = next_yield_time
        next_yield_time += period
      else:
        signal_msg = capture_queue.get()
        if signal_msg is MidiCaptor._WAKE_MESSAGE:
          # This is only recieved when the thread is in the process of
          # terminating. Wait until it is done before yielding the final
          # sequence.
          self.join()
          break
        end_time = signal_msg.time
      # Acquire lock so that `captured_sequence` will be called before thread
      # terminates, if it has not already done so.
      with self._lock:
        if not self.is_alive():
          break
        captured_sequence = self.captured_sequence(end_time)
      yield captured_sequence
    yield self.captured_sequence()

  def register_callback(self, fn, signal=None, period=None):
    """Calls `fn` at every signal message or time period.

    The callback function must take exactly one argument, which will be the
    current captured NoteSequence.

    Exactly one of `signal` or `period` must be specified. Continues until the
    captor thread terminates, at which point the callback is called with the
    final sequence, or `cancel_callback` is called.

    If callback execution is longer than a period, immediately calls upon
    completion and logs a warning.

    Args:
      fn: The callback function to call, passing in the captured sequence.
      signal: A MidiSignal to use as a signal to call `fn` on the current
          captured sequence, or None.
      period: A float period in seconds to specify how often to call `fn`, or
          None.

    Returns:
      The unqiue name of the callback thread to enable cancellation.

    Raises:
      MidiHubError: If neither `signal` nor `period` or both are specified.
    """

    class IteratorCallback(threading.Thread):
      """A thread for executing a callback on each iteration."""

      def __init__(self, iterator, fn):
        self._iterator = iterator
        self._fn = fn
        self._stop_signal = threading.Event()
        super(IteratorCallback, self).__init__()

      def run(self):
        """Calls the callback function for each iterator value."""
        for captured_sequence in self._iterator:
          if self._stop_signal.is_set():
            break
          self._fn(captured_sequence)

      def stop(self):
        """Stops the thread on next iteration, without blocking."""
        self._stop_signal.set()

    t = IteratorCallback(self.iterate(signal, period), fn)
    t.start()

    with self._lock:
      assert t.name not in self._callbacks
      self._callbacks[t.name] = t

    return t.name

  @concurrency.serialized
  def cancel_callback(self, name):
    """Cancels the callback with the given name.

    While the thread may continue to run until the next iteration, the callback
    function will not be executed.

    Args:
      name: The unique name of the callback thread to cancel.
    """
    self._callbacks[name].stop()
    del self._callbacks[name]


class MonophonicMidiCaptor(MidiCaptor):
  """A MidiCaptor for monophonic melodies."""

  def __init__(self, *args, **kwargs):
    self._open_note = None
    super(MonophonicMidiCaptor, self).__init__(*args, **kwargs)

  @concurrency.serialized
  def _capture_message(self, msg):
    """Handles a single incoming MIDI message during capture.

    If the message is a note_on event, ends the previous note (if applicable)
    and opens a new note in the capture sequence. Ignores repeated note_on
    events.

    If the message is a note_off event matching the current open note in the
    capture sequence

    Args:
      msg: The mido.Message MIDI message to handle.
    """
    if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
      if self._open_note is None or msg.note != self._open_note.pitch:
        # This is not the note we're looking for. Drop it.
        return

      self._open_note.end_time = msg.time
      self._open_note = None

    elif msg.type == 'note_on':
      if self._open_note:
        if self._open_note.pitch == msg.note:
          # This is just a repeat of the previous message.
          return
        # End the previous note.
        self._open_note.end_time = msg.time

      self._open_note = self._add_note(msg)


class PolyphonicMidiCaptor(MidiCaptor):
  """A MidiCaptor for polyphonic melodies."""

  def __init__(self, *args, **kwargs):
    # A dictionary of open NoteSequence.Note messages keyed by pitch.
    self._open_notes = dict()
    super(PolyphonicMidiCaptor, self).__init__(*args, **kwargs)

  @concurrency.serialized
  def _capture_message(self, msg):
    """Handles a single incoming MIDI message during capture.

    Args:
      msg: The mido.Message MIDI message to handle.
    """
    if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
      if msg.note not in self._open_notes:
        # This is not a note we're looking for. Drop it.
        return

      self._open_notes[msg.note].end_time = msg.time
      del self._open_notes[msg.note]

    elif msg.type == 'note_on':
      if msg.note in self._open_notes:
        # This is likely just a repeat of the previous message.
        return

      new_note = self._add_note(msg)
      self._open_notes[new_note.pitch] = new_note


class TextureType(object):
  """An Enum specifying the type of musical texture."""
  MONOPHONIC = 1
  POLYPHONIC = 2


class MidiHub(object):
  """A MIDI interface for capturing and playing NoteSequences.

  Ignores/filters `program_change` messages. Assumes all messages are on the
  same channel.

  Args:
    input_midi_port: The string MIDI port name or mido.ports.BaseInput object to
        use for input. If a name is given that is not an available port, a
        virtual port will be opened with that name.
    output_midi_port: The string MIDI port name mido.ports.BaseOutput object to
        use for output. If a name is given that is not an available port, a
        virtual port will be opened with that name.
    texture_type: A TextureType Enum specifying the musical texture to assume
        during capture, passthrough, and playback.
    passthrough: A boolean specifying whether or not to pass incoming messages
        through to the output, applying the appropriate texture rules.
    playback_channel: The MIDI channel to send playback events.
    playback_offset: The float time in seconds to adjust the playback event
        times by.
  """

  def __init__(self, input_midi_ports, output_midi_ports, texture_type,
               passthrough=True, playback_channel=0, playback_offset=0.0):
    self._texture_type = texture_type
    self._passthrough = passthrough
    self._playback_channel = playback_channel
    self._playback_offset = playback_offset
    # When `passthrough` is True, this is the set of open MIDI note pitches.
    self._open_notes = set()
    # This lock is used by the serialized decorator.
    self._lock = threading.RLock()
    # A dictionary mapping a compiled MidiSignal regex to a condition variable
    # that will be notified when a matching messsage is received.
    self._signals = {}
    # A dictionary mapping a compiled MidiSignal regex to a list of functions
    # that will be called with the triggering message in individual threads when
    # a matching message is received.
    self._callbacks = collections.defaultdict(list)
    # A dictionary mapping integer control numbers to most recently-received
    # integer value.
    self._control_values = {}
    # Threads actively being used to capture incoming messages.
    self._captors = []
    # Potentially active player threads.
    self._players = []
    self._metronome = None

    # Open MIDI ports.

    inports = []
    if input_midi_ports:
      for port in input_midi_ports:
        if isinstance(port, mido.ports.BaseInput):
          inport = port
        else:
          virtual = port not in get_available_input_ports()
          if virtual:
            tf.logging.info(
                "Opening '%s' as a virtual MIDI port for input.", port)
          inport = mido.open_input(port, virtual=virtual)
        # Start processing incoming messages.
        inport.callback = self._timestamp_and_handle_message
        inports.append(inport)
      # Keep references to input ports to prevent deletion.
      self._inports = inports
    else:
      tf.logging.warn('No input port specified. Capture disabled.')
      self._inports = None

    outports = []
    for port in output_midi_ports:
      if isinstance(port, mido.ports.BaseOutput):
        outports.append(port)
      else:
        virtual = port not in get_available_output_ports()
        if virtual:
          tf.logging.info(
              "Opening '%s' as a virtual MIDI port for output.", port)
        outports.append(mido.open_output(port, virtual=virtual))
    self._outport = mido.ports.MultiPort(outports)

  def __del__(self):
    """Stops all running threads and waits for them to terminate."""
    for captor in self._captors:
      captor.stop(block=False)
    for player in self._players:
      player.stop(block=False)
    self.stop_metronome()
    for captor in self._captors:
      captor.join()
    for player in self._players:
      player.join()

  @property
  @concurrency.serialized
  def passthrough(self):
    return self._passthrough

  @passthrough.setter
  @concurrency.serialized
  def passthrough(self, value):
    """Sets passthrough value, closing all open notes if being disabled."""
    if self._passthrough == value:
      return
    # Close all open notes.
    while self._open_notes:
      self._outport.send(mido.Message('note_off', note=self._open_notes.pop()))
    self._passthrough = value

  def _timestamp_and_handle_message(self, msg):
    """Stamps message with current time and passes it to the handler."""
    if msg.type == 'program_change':
      return
    if not msg.time:
      msg.time = time.time()
    self._handle_message(msg)

  @concurrency.serialized
  def _handle_message(self, msg):
    """Handles a single incoming MIDI message.

    -If the message is being used as a signal, notifies threads waiting on the
     appropriate condition variable.
    -Adds the message to any capture queues.
    -Passes the message through to the output port, if appropriate.

    Args:
      msg: The mido.Message MIDI message to handle.
    """
    # Notify any threads waiting for this message.
    msg_str = str(msg)
    for regex in list(self._signals):
      if regex.match(msg_str) is not None:
        self._signals[regex].notify_all()
        del self._signals[regex]

    # Call any callbacks waiting for this message.
    for regex in list(self._callbacks):
      if regex.match(msg_str) is not None:
        for fn in self._callbacks[regex]:
          threading.Thread(target=fn, args=(msg,)).start()

        del self._callbacks[regex]

    # Remove any captors that are no longer alive.
    self._captors[:] = [t for t in self._captors if t.is_alive()]
    # Add a different copy of the message to the receive queue of each live
    # capture thread.
    for t in self._captors:
      t.receive(msg.copy())

    # Update control values if this is a control change message.
    if msg.type == 'control_change':
      if self._control_values.get(msg.control, None) != msg.value:
        tf.logging.debug('Control change %d: %d', msg.control, msg.value)
      self._control_values[msg.control] = msg.value

    # Pass the message through to the output port, if appropriate.
    if not self._passthrough:
      pass
    elif self._texture_type == TextureType.POLYPHONIC:
      if msg.type == 'note_on' and msg.velocity > 0:
        self._open_notes.add(msg.note)
      elif (msg.type == 'note_off' or
            (msg.type == 'note_on' and msg.velocity == 0)):
        self._open_notes.discard(msg.note)
      self._outport.send(msg)
    elif self._texture_type == TextureType.MONOPHONIC:
      assert len(self._open_notes) <= 1
      if msg.type not in ['note_on', 'note_off']:
        self._outport.send(msg)
      elif ((msg.type == 'note_off' or
             msg.type == 'note_on' and msg.velocity == 0) and
            msg.note in self._open_notes):
        self._outport.send(msg)
        self._open_notes.remove(msg.note)
      elif msg.type == 'note_on' and msg.velocity > 0:
        if self._open_notes:
          self._outport.send(
              mido.Message('note_off', note=self._open_notes.pop()))
        self._outport.send(msg)
        self._open_notes.add(msg.note)

  def start_capture(self, qpm, start_time, stop_time=None, stop_signal=None):
    """Starts a MidiCaptor to compile incoming messages into a NoteSequence.

    If neither `stop_time` nor `stop_signal`, are provided, the caller must
    explicitly stop the returned capture thread. If both are specified, the one
    that occurs first will stop the capture.

    Args:
      qpm: The integer quarters per minute to use for the captured sequence.
      start_time: The float wall time in seconds to start the capture. May be in
        the past. Used for beat alignment.
      stop_time: The optional float wall time in seconds to stop the capture.
      stop_signal: The optional mido.Message to use as a signal to use to stop
         the capture.

    Returns:
      The MidiCaptor thread.
    """
    if self._texture_type == TextureType.MONOPHONIC:
      captor_class = MonophonicMidiCaptor
    else:
      captor_class = PolyphonicMidiCaptor
    captor = captor_class(qpm, start_time, stop_time, stop_signal)
    with self._lock:
      self._captors.append(captor)
    captor.start()
    return captor

  def capture_sequence(self, qpm, start_time, stop_time=None, stop_signal=None):
    """Compiles and returns incoming messages into a NoteSequence.

    Blocks until capture stops. At least one of `stop_time` or `stop_signal`
    must be specified. If both are specified, the one that occurs first will
    stop the capture.

    Args:
      qpm: The integer quarters per minute to use for the captured sequence.
      start_time: The float wall time in seconds to start the capture. May be in
        the past. Used for beat alignment.
      stop_time: The optional float wall time in seconds to stop the capture.
      stop_signal: The optional mido.Message to use as a signal to use to stop
         the capture.

    Returns:
      The captured NoteSequence proto.
    Raises:
      MidiHubError: When neither `stop_time` nor `stop_signal` are provided.
    """
    if stop_time is None and stop_signal is None:
      raise MidiHubError(
          'At least one of `stop_time` and `stop_signal` must be provided to '
          '`capture_sequence` call.')
    captor = self.start_capture(qpm, start_time, stop_time, stop_signal)
    captor.join()
    return captor.captured_sequence()

  @concurrency.serialized
  def wait_for_event(self, signal=None, timeout=None):
    """Blocks until a matching mido.Message arrives or the timeout occurs.

    Exactly one of `signal` or `timeout` must be specified. Using a timeout
    with a threading.Condition object causes additional delays when notified.

    Args:
      signal: A MidiSignal to use as a signal to stop waiting, or None.
      timeout: A float timeout in seconds, or None.

    Raises:
      MidiHubError: If neither `signal` nor `timeout` or both are specified.
    """
    if (signal, timeout).count(None) != 1:
      raise MidiHubError(
          'Exactly one of `signal` or `timeout` must be provided to '
          '`wait_for_event` call.')

    if signal is None:
      concurrency.Sleeper().sleep(timeout)
      return

    signal_pattern = str(signal)
    cond_var = None
    for regex, cond_var in self._signals:
      if regex.pattern == signal_pattern:
        break
    if cond_var is None:
      cond_var = threading.Condition(self._lock)
      self._signals[re.compile(signal_pattern)] = cond_var

    cond_var.wait()

  @concurrency.serialized
  def wake_signal_waiters(self, signal=None):
    """Wakes all threads waiting on a signal event.

    Args:
      signal: The MidiSignal to wake threads waiting on, or None to wake all.
    """
    for regex in list(self._signals):
      if signal is None or regex.pattern == str(signal):
        self._signals[regex].notify_all()
        del self._signals[regex]
    for captor in self._captors:
      captor.wake_signal_waiters(signal)

  @concurrency.serialized
  def start_metronome(self, qpm, start_time, signals=None, channel=None):
    """Starts or updates the metronome with the given arguments.

    Args:
      qpm: The quarter notes per minute to use.
      start_time: The wall time in seconds that the metronome is started on for
        synchronization and beat alignment. May be in the past.
      signals: An ordered collection of MidiSignals whose underlying messages
        are to be output on the metronome's tick, cyclically. A None value can
        be used in place of a MidiSignal to output nothing on a given tick.
      channel: The MIDI channel to output ticks on.
    """
    if self._metronome is not None and self._metronome.is_alive():
      self._metronome.update(
          qpm, start_time, signals=signals, channel=channel)
    else:
      self._metronome = Metronome(
          self._outport, qpm, start_time, signals=signals, channel=channel)
      self._metronome.start()

  @concurrency.serialized
  def stop_metronome(self, stop_time=0, block=True):
    """Stops the metronome at the given time if it is currently running.

    Args:
      stop_time: The float wall time in seconds after which the metronome should
          stop. By default, stops at next tick.
      block: If true, blocks until metronome is stopped.
    """
    if self._metronome is None:
      return
    self._metronome.stop(stop_time, block)
    self._metronome = None

  def start_playback(self, sequence, start_time=time.time(),
                     allow_updates=False):
    """Plays the notes in aNoteSequence via the MIDI output port.

    Args:
      sequence: The NoteSequence to play, with times based on the wall clock.
      start_time: The float time before which to strip events. Defaults to call
          time. Events before this time will be sent immediately on start.
      allow_updates: A boolean specifying whether or not the player should stay
          allow the sequence to be updated and stay alive until `stop` is
          called.
    Returns:
      The MidiPlayer thread handling playback to enable updating.
    """
    player = MidiPlayer(self._outport, sequence, start_time, allow_updates,
                        self._playback_channel, self._playback_offset)
    with self._lock:
      self._players.append(player)
    player.start()
    return player

  @concurrency.serialized
  def control_value(self, control_number):
    """Returns the most recently received value for the given control number.

    Args:
      control_number: The integer control number to return the value for, or
          None.

    Returns:
      The most recently recieved integer value for the given control number, or
      None if no values have been received for that control.
    """
    if control_number is None:
      return None
    return self._control_values.get(control_number)

  def send_control_change(self, control_number, value):
    """Sends the specified control change message on the output port."""
    self._outport.send(
        mido.Message(
            type='control_change',
            control=control_number,
            value=value))

  @concurrency.serialized
  def register_callback(self, fn, signal):
    """Calls `fn` at the next signal message.

    The callback function must take exactly one argument, which will be the
    message triggering the signal.

    Survives until signal is called or the MidiHub is destroyed.

    Args:
      fn: The callback function to call, passing in the triggering message.
      signal: A MidiSignal to use as a signal to call `fn` on the triggering
          message.
    """
    self._callbacks[re.compile(str(signal))].append(fn)
