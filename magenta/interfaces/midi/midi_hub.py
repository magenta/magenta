"""A module for interfacing with the MIDI environment."""

import abc
from collections import deque
import logging
import Queue
import re
import threading
import time

# internal imports
import mido

# TODO(adarob): Use flattened imports.
from magenta.common import concurrency
from magenta.protobuf import music_pb2

_DEFAULT_METRONOME_TICK_DURATION = 0.05
_DEFAULT_METRONOME_PITCH = 95
_DEFAULT_METRONOME_VELOCITY = 64
_METRONOME_CHANNEL = 0

# The RtMidi backend is easier to install and has support for virtual ports.
mido.set_backend('mido.backends.rtmidi')


class MidiHubException(Exception):
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
    MidiHubException: If the message type is unsupported or the arguments are
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
      raise MidiHubException(
          'Either a mido.Message should be provided or arguments. Not both.')

    type_ = msg.type if msg is not None else kwargs.get('type')
    if type_ is not None and type_ not in self._VALID_ARGS:
      raise MidiHubException(
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
            raise MidiHubException(
                "Invalid argument for type '%s': %s" % (type_, arg_name))
      else:
        if kwargs:
          for name, args in self._VALID_ARGS.iteritems():
            if set(kwargs) <= args:
              inferred_types.append(name)
        if not inferred_types:
          raise MidiHubException(
              'Could not infer a message type for set of given arguments: %s'
              % ', '.join(kwargs))
        # If there is only a single valid inferred type, use it.
        if len(inferred_types) == 1:
          type_ = inferred_types[0]

    if msg is not None:
      self._regex_pattern = '^' + mido.messages.format_as_string(
          msg, include_time=False) + r' time=\d+.\d+$'
    else:
      # Generate regex pattern.
      parts = ['.*' if type_ is None else type_]
      for name in mido.messages.get_spec(inferred_types[0]).arguments:
        if name in kwargs:
          parts.append('%s=%d' % (name, kwargs[name]))
        else:
          parts.append(r'%s=\d+' % name)
      self._regex_pattern = '^' + ' '.join(parts) + r' time=\d+.\d+$'

  def __str__(self):
    """Returns a regex pattern for matching against a mido.Message string."""
    return self._regex_pattern


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
    velocity: The velocity of the metronome's tick `note_on` message.
    pitch: The pitch of the metronome's tick `note_on` message.
    duration: The duration of the metronome's tick.
  """

  def __init__(self,
               outport,
               qpm,
               start_time,
               stop_time=None,
               velocity=_DEFAULT_METRONOME_VELOCITY,
               pitch=_DEFAULT_METRONOME_PITCH,
               duration=_DEFAULT_METRONOME_TICK_DURATION):
    self._outport = outport
    self._qpm = qpm
    self._start_time = start_time
    self._velocity = velocity
    self._pitch = pitch
    self._duration = duration
    # A signal for when to stop the metronome.
    self._stop_time = stop_time
    super(Metronome, self).__init__()

  def run(self):
    """Outputs metronome tone on the qpm interval until stop signal received."""
    period = 60. / self._qpm
    sleeper = concurrency.Sleeper()
    now = time.time()
    next_tick_time = max(
        self._start_time,
        now + period - ((now - self._start_time) % period))
    while self._stop_time is None or self._stop_time > next_tick_time:
      sleeper.sleep_until(next_tick_time)

      self._outport.send(
          mido.Message(
              type='note_on',
              note=self._pitch,
              channel=_METRONOME_CHANNEL,
              velocity=self._velocity))

      sleeper.sleep(self._duration)

      self._outport.send(
          mido.Message(
              type='note_off',
              note=self._pitch,
              channel=_METRONOME_CHANNEL))

      now = time.time()
      next_tick_time = now + period - ((now - self._start_time) % period)

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
    allow_updates: If False, the thread will terminate after playback of
        `sequence` completes and calling `update_sequence` will result in an
        exception. Otherwise, the the thread will stay alive until `stop` is
        called, allowing for additional updates via `update_sequence`.
  """

  def __init__(self, outport, sequence, allow_updates=False):
    self._outport = outport
    # Set of notes (pitches) that are currently on.
    self._open_notes = set()
    # Lock for serialization.
    self._lock = threading.RLock()
    # A control variable to signal when the sequence has been updated.
    self._update_cv = threading.Condition(self._lock)
    # The queue of mido.Message objects to send, sorted by ascending time.
    self._message_queue = deque()
    # An event that is set when `stop` has been called.
    self._stop_signal = threading.Event()

    # Initialize message queue.
    # We first have to allow "updates" to set the initial sequence.
    self._allow_updates = True
    self.update_sequence(sequence)
    # We now make whether we allow updates dependent on the argument.
    self._allow_updates = allow_updates
    super(MidiPlayer, self).__init__()

  @concurrency.serialized
  def update_sequence(self, sequence):
    """Updates sequence being played by the MidiPlayer.

    Adds events to close any notes that are no longer being closed by the
    new sequence using the times when they would have been closed by the
    previous sequence.

    Args:
      sequence: The NoteSequence to play back.
    Raises:
      MidiHubException: If called when _allow_updates is False.
    """
    if not self._allow_updates:
      raise MidiHubException(
          'Attempted to update a MidiPlayer sequence with updates disabled.')

    start_time = time.time()

    new_message_list = []
    # The set of pitches that are already playing but are not closed without
    # being reopened in the future in the new sequence.
    notes_to_close = set()
    for note in sequence.notes:
      if note.start_time >= start_time:
        new_message_list.append(
            mido.Message(type='note_on', note=note.pitch,
                         velocity=note.velocity, time=note.start_time))
      if note.end_time >= start_time:
        new_message_list.append(
            mido.Message(type='note_off', note=note.pitch, time=note.end_time))
        if note.start_time < start_time and note.pitch not in self._open_notes:
          notes_to_close.add(note.pitch)

    for msg in self._message_queue:
      if not notes_to_close:
        break
      if msg.note in notes_to_close:
        assert msg.type == 'note_off'
        new_message_list.append(msg)
        notes_to_close.remove(msg.note)

    self._message_queue = deque(sorted(new_message_list, key=lambda x: x.time))
    self._update_cv.notify()

  @concurrency.serialized
  def run(self):
    """Plays messages in the queue until empty and _allow_updates is False."""
    # Assumes model where NoteSequence is time-stampped with wall time.
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
    self._receive_queue = Queue.Queue()
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
      MidiHubException: When the received message has an empty time attribute.
    """
    if not msg.time:
      raise MidiHubException(
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
      except Queue.Empty:
        continue

      if msg is MidiCaptor._WAKE_MESSAGE:
        continue

      if msg.time <= self._start_time:
        continue

      if self._stop_regex.match(str(msg)) is not None:
        break

      with self._lock:
        msg_str = str(msg)
        for regex, queue in self._iter_signals:
          if regex.match(msg_str) is not None:
            queue.put(msg.copy())

      self._capture_message(msg)

    stop_time = self._stop_time
    end_time = stop_time if stop_time is not None else msg.time

    # Acquire lock to avoid race condition with `iterate`.
    with self._lock:
      # Set final captured sequence.
      self._captured_sequence = self.captured_sequence(end_time)
      # Wake up all generators.
      for regex, queue in self._iter_signals:
        queue.put(MidiCaptor._WAKE_MESSAGE)

  def stop(self, stop_time=None, block=True):
    """Ends capture and truncates the captured sequence at `stop_time`.

    Args:
      stop_time: The float time in seconds to stop the capture, or None if it
         should be stopped now. May be in the past, in which case the captured
         sequence will be truncated appropriately.
      block: If True, blocks until the thread terminates.
    Raises:
      MidiHubException: When called multiple times with a `stop_time`.
    """
    with self._lock:
      if self._stop_signal.is_set():
        if stop_time is not None:
          raise MidiHubException(
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
      MidiHubException: When the thread is alive and `end_time` is None or the
         thread is terminated and `end_time` is not None.
    """
    # Make a copy of the sequence currently being captured.
    current_captured_sequence = music_pb2.NoteSequence()
    with self._lock:
      current_captured_sequence.CopyFrom(self._captured_sequence)

    if self.is_alive():
      if end_time is None:
        raise MidiHubException(
            '`end_time` must be provided when capture thread is still running.')
      for i, note in enumerate(current_captured_sequence.notes):
        if note.start_time >= end_time:
          del current_captured_sequence.notes[i:]
          break
        if not note.end_time or note.end_time > end_time:
          note.end_time = end_time
      current_captured_sequence.total_time = end_time
    elif end_time is not None:
      raise MidiHubException(
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
      MidiHubException: If neither `signal` nor `period` or both are specified.
    """
    if (signal, period).count(None) != 1:
      raise MidiHubException(
          'Exactly one of `signal` or `period` must be provided to `iterate` '
          'call.')

    if signal is None:
      sleeper = concurrency.Sleeper()
      next_yield_time = time.time() + period
    else:
      regex = re.compile(str(signal))
      queue = Queue.Queue()
      with self._lock:
        self._iter_signals.append((regex, queue))

    while self.is_alive():
      if signal is None:
        skipped_periods = (time.time() - next_yield_time) // period
        if skipped_periods > 0:
          logging.warning(
              'Skipping %d %.3fs period(s) to catch up on iteration.',
              skipped_periods, period)
          next_yield_time += skipped_periods * period
        else:
          sleeper.sleep_until(next_yield_time)
        end_time = next_yield_time
        next_yield_time += period
      else:
        signal_msg = queue.get()
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

    The callback function must take exactly a single argument, which will be the
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
      MidiHubException: If neither `signal` nor `period` or both are specified.
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

      new_note = self._captured_sequence.notes.add()
      new_note.start_time = msg.time
      new_note.pitch = msg.note
      new_note.velocity = msg.velocity
      self._open_note = new_note


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

      new_note = self._captured_sequence.notes.add()
      new_note.start_time = msg.time
      new_note.pitch = msg.note
      new_note.velocity = msg.velocity
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
        through to the output, applyig the appropriate texture rules.
  """

  def __init__(self, input_midi_port, output_midi_port, texture_type,
               passthrough=True):
    self._texture_type = texture_type
    self._passthrough = passthrough
    # When `passthrough` is True, this is the set of open MIDI note pitches.
    self._open_notes = set()
    # This lock is used by the serialized decorator.
    self._lock = threading.RLock()
    # A dictionary mapping a string-formatted mido.Messages to a condition
    # variable that will be notified when a matching messsage is received,
    # ignoring the time field.
    self._signals = {}
    # Threads actively being used to capture incoming messages.
    self._captors = []
    # Potentially active player threads.
    self._players = []
    self._metronome = None

    # Open MIDI ports.
    self._inport = (
        input_midi_port if isinstance(input_midi_port, mido.ports.BaseInput)
        else mido.open_input(
            input_midi_port,
            virtual=input_midi_port not in get_available_input_ports()))
    self._outport = (
        output_midi_port if isinstance(output_midi_port, mido.ports.BaseOutput)
        else mido.open_output(
            output_midi_port,
            virtual=output_midi_port not in get_available_output_ports()))

    # Start processing incoming messages.
    self._inport.callback = self._timestamp_and_handle_message

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

    # Remove any captors that are no longer alive.
    self._captors[:] = [t for t in self._captors if t.is_alive()]
    # Add a different copy of the message to the receive queue of each live
    # capture thread.
    for t in self._captors:
      t.receive(msg.copy())

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
    captor_class = (MonophonicMidiCaptor if
                    self._texture_type == TextureType.MONOPHONIC else
                    PolyphonicMidiCaptor)
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
      MidiHubException: When neither `stop_time` nor `stop_signal` are provided.
    """
    if stop_time is None and stop_signal is None:
      raise MidiHubException(
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
      MidiHubException: If neither `signal` nor `timeout` or both are specified.
    """
    if (signal, timeout).count(None) != 1:
      raise MidiHubException(
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

  @concurrency.serialized
  def start_metronome(self, qpm, start_time):
    """Starts or re-starts the metronome with the given arguments.

    Args:
      qpm: The quarter notes per minute to use.
      start_time: The wall time in seconds that the metronome is started on for
        synchronization and beat alignment. May be in the past.
    """
    if self._metronome is not None:
      self.stop_metronome()
    self._metronome = Metronome(self._outport, qpm, start_time)
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

  def start_playback(self, sequence, allow_updates=False):
    """Plays the notes in aNoteSequence via the MIDI output port.

    Args:
      sequence: The NoteSequence to play, with times based on the wall clock.
      allow_updates: A boolean specifying whether or not the player should stay
          allow the sequence to be updated and stay alive until `stop` is
          called.
    Returns:
      The MidiPlayer thread handling playback to enable updating.
    """
    player = MidiPlayer(self._outport, sequence, allow_updates)
    with self._lock:
      self._players.append(player)
    player.start()
    return player
