"""A module for interfacing with the MIDI environment."""

import abc
from collections import deque
import functools
import Queue
import re
import threading
import time

# internal imports
import mido

from magenta.protobuf import music_pb2

_DEFAULT_METRONOME_TICK_DURATION = 0.05
_DEFAULT_METRONOME_PITCH = 95
_DEFAULT_METRONOME_VELOCITY = 64
_METRONOME_CHANNEL = 0


def serialized(func):
  """Decorator to provide mutual exclusion for method using _lock attribute."""

  @functools.wraps(func)
  def serialized_method(self, *args, **kwargs):
    lock = getattr(self, '_lock')
    with lock:
      return func(self, *args, **kwargs)

  return serialized_method


class MidiHubException(Exception):
  """Base class for exceptions in this module."""
  pass


class MidiSignal(object):
  """A class for representing a MIDI-based event signal.

  Provides a `__str__` method to return a regular expression pattern for
  matching against the string representation of a mido.Message with wildcards
  for unspecified values.

  Supports matching for message types 'note_on', 'note_off', and
  'control_change'. If a mido.Message is given as the `msg` argument, matches
  against the exact message, ignoring the time attribute. If a `msg` is
  not given, keyword arguments must be provided matching some subset of those
  listed in `_VALID_ARGS`.

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
    start_time: The float wall time in seconds to treat as the first beat
        for alignment.
    qpm: The integer quarters per minute to signal on.
    velocity: The velocity of the metronome's tick `note_on` message.
    pitch: The pitch of the metronome's tick `note_on` message.
    duration: The duration of the metronome's tick.
  """
  daemon = True

  def __init__(self,
               outport,
               start_time,
               qpm,
               velocity=_DEFAULT_METRONOME_VELOCITY,
               pitch=_DEFAULT_METRONOME_PITCH,
               duration=_DEFAULT_METRONOME_TICK_DURATION):
    self._outport = outport
    self._start_time = start_time
    self._qpm = qpm
    self._velocity = velocity
    self._pitch = pitch
    self._duration = duration
    # A signal for when to stop the metronome.
    self._stop_metronome = False
    super(Metronome, self).__init__()

  def run(self):
    """Outputs metronome tone on the qpm interval until stop signal received."""
    period = 60. / self._qpm
    sleep_offset = 0
    while not self._stop_metronome:
      now = time.time()
      next_tick_time = now + period - ((now - self._start_time) % period)
      delta = next_tick_time - time.time()
      if delta > 0:
        time.sleep(delta + sleep_offset)

      # The sleep function tends to return a little early or a little late.
      # Gradually modify an offset based on whether it returned early or late,
      # but prefer returning a little bit early.
      # If it returned early, spin until the correct time occurs.
      tick_late = time.time() - next_tick_time
      if tick_late > 0:
        sleep_offset -= .0005
      elif tick_late < -.001:
        sleep_offset += .0005

      if tick_late < 0:
        while time.time() < next_tick_time:
          pass

      self._outport.send(
          mido.Message(
              type='note_on',
              note=self._pitch,
              channel=_METRONOME_CHANNEL,
              velocity=self._velocity))
      time.sleep(self._duration)
      self._outport.send(
          mido.Message(
              type='note_off',
              note=self._pitch,
              channel=_METRONOME_CHANNEL))

  def stop(self):
    """Signals for the metronome to stop and joins thread."""
    self._stop_metronome = True
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
  daemon = True

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

    # Initialize message queue.
    self._allow_updates = True
    self.update_sequence(sequence)
    self._allow_updates = allow_updates
    super(MidiPlayer, self).__init__()

  @serialized
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

  @serialized
  def run(self):
    """Plays messages in the queue until empty and _allow_updates is False."""
    # Assumes model where NoteSequence is time-stampped with wall time.
    # TODO(hanzorama): Argument to allow initial start not at sequence start?

    while self._message_queue[0].time < time.time():
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

  def stop(self):
    """Signals for the playback to stop and ends all open notes.

    Blocks until completed.
    """
    with self._lock:
      self._allow_updates = False

      # Replace message queue with immediate end of open notes.
      self._message_queue.clear()
      for note in self._open_notes:
        self._message_queue.append(
            mido.Message(type='note_off', note=note, time=time.time()))
      self._update_cv.notify()
    self.join()


class MidiCaptor(threading.Thread):
  """Base class for thread that captures MIDI into a NoteSequence proto.

  If neither `stop_time` nor `stop_signal` are provided as arguments, the
  capture will continue until the `stop` method is called. If both are provided,
  the first to occur while trigger termination of the thread.

  Args:
    qpm: The quarters per minute to use for the captured sequence.
    start_time: The float wall time in seconds when the capture begins.
    stop_time: The float wall time in seconds when the capture is to be stopped
        or None.
    stop_signal: A MidiSignal to use as a signal to stop capture.

  """
  _metaclass__ = abc.ABCMeta
  daemon = True

  # A message that is used to wake the consumer thread.
  _WAKE_MESSAGE = None

  def __init__(self, qpm, start_time, stop_time=None, stop_signal=None):
    self._receive_queue = Queue.Queue()
    self._captured_sequence = music_pb2.NoteSequence()
    self._captured_sequence.tempos.add(qpm=qpm)
    self._start_time = start_time
    self._stop_time = stop_time
    self._stop_regex = re.compile(str(stop_signal))
    # A set of active MidiSignals being used by iterators.
    self._iter_signals = []
    # A lock for accessing `_captured_sequence`.
    self._lock = threading.RLock()
    super(MidiCaptor, self).__init__()

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
    self._receive_queue.put_nowait(msg)

  @abc.abstractmethod
  @serialized
  def _capture_message(self, msg):
    """Handles a single incoming MIDI message during capture.

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
      with self._lock:
        if self._stop_time is not None:
          timeout = self._stop_time - time.time()
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
        for regex, cond_var in self._iter_signals:
          if regex.match(msg_str) is not None:
            cond_var.notify()

      self._capture_message(msg)

    end_time = self._stop_time if self._stop_time is not None else msg.time
    with self._lock:
      # Set final captured sequence.
      self._captured_sequence = self.captured_sequence(end_time)
      # Wake up all generators.
      for regex, cond_var in self._iter_signals:
        cond_var.notify()

  def stop(self, stop_time=None):
    """Ends capture and truncates the captured sequence at `stop_time`.

    Blocks until complete.

    Args:
      stop_time: The float time in seconds to stop the capture, or None if it
         should be stopped now. May be in the past, in which case the captured
         sequence will be truncated appropriately.

    Raises:
      MidiHubException: When thread is not running.
    """
    if not self.is_alive:
      raise MidiHubException(
          'Attempted to stop MidiCaptor that is not running.')
    with self._lock:
      self._stop_time = time.time() if stop_time is None else stop_time
    # Force the thread to wake since we've updated the stop time.
    self._receive_queue.put_nowait(MidiCaptor._WAKE_MESSAGE)
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
    with self._lock:
      current_captured_sequence = music_pb2.NoteSequence()
      current_captured_sequence.CopyFrom(self._captured_sequence)

    if self.is_alive():
      if end_time is None:
        raise MidiHubException(
            '`end_time` must be provided when capture thread is still running.')
      for i, note in enumerate(current_captured_sequence.notes):
        if note.start_time > end_time:
          del current_captured_sequence.notes[i:]
          break
        if not note.end_time or note.end_time > end_time:
          note.end_time = end_time
      current_captured_sequence.total_time = end_time
    elif end_time is not None:
      raise MidiHubException(
          '`end_time` must not be provided when capture is complete.')

    return current_captured_sequence

  def iterate(self, signal=None, timeout=None):
    """Blocks until a matching mido.Message arrives or the timeout occurs.

    At least one of `signal` or `timeout` must be specified. If both are
    specified, returns when the first occurs.
    Continues until the thread terminates, at which point the final captured
    sequence is yielded before returning.

    Args:
      signal: A MidiSignal to use as a signal to stop waiting.
      timeout: A float timeout in seconds.
    Yields:
      The captured NoteSequence at event time.
    """
    # The serialized decorator is not compatible with generators.
    with self._lock:
      if signal is not None:
        regex = re.compile(str(signal))
        cond_var = threading.Condition(self._lock)
        self._iter_signals.append((regex, cond_var))
      while self.is_alive:
        if signal is None:
          time.sleep(timeout)
        else:
          cond_var.wait(timeout=timeout)
        if not self.is_alive:
          break
        yield self.captured_sequence(time.time())
      yield self.captured_sequence()


class MonophonicMidiCaptor(MidiCaptor):
  """A MidiCaptor for monophonic melodies."""

  def __init__(self, *args, **kwargs):
    self._open_note = None
    super(MonophonicMidiCaptor, self).__init__(*args, **kwargs)

  @serialized
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

  @serialized
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
    input_midi_port: The string MIDI port name to use for input.
    output_midi_port: The string MIDI port name to use for output.
    texture_type: A TextureType Enum specifying the musical texture to assume
        during capture, passthrough, and playback.
    passthrough: A boolean specifying whether or not to pass incoming messages
        through to the output, applyig the appropriate texture rules.
  """

  def __init__(self, input_midi_port, output_midi_port, texture_type,
               passthrough):
    self._inport = mido.open_input(input_midi_port)
    self._outport = mido.open_output(output_midi_port)
    self._texture_type = texture_type
    self._passthrough = passthrough
    # When `passthrough` is True, this is the set of open MIDI note pitches.
    self._open_notes = set()
    # This lock is used by the serialized decorator.
    self._lock = threading.RLock()
    # A dictionary mapping a string-formatted mido.Messages to a condition
    # variable that will be notified when a matching messsage is received,
    # ignoring the time field.
    self._signals = dict()
    # Threads actively being used to capture incoming messages.
    self._capture_threads = []

    # Start processing incoming messages.
    self._inport.callback = self._timestamp_and_handle_message

  @property
  @serialized
  def passthrough(self):
    return self._passthrough

  @passthrough.setter
  @serialized
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
    msg.time = time.time()
    self._handle_message(msg)

  @serialized
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
    for regex in self._signals:
      if regex.match(msg_str) is not None:
        self._signals[regex].notify_all()
        del self._signals[regex]

    # Remove any capture threads that are no longer alive.
    self._capture_threads[:] = [
        t for t in self._capture_threads if t.is_alive()
    ]
    # Add a different copy of the message to the receive queue of each live
    # capture thread.
    for t in self._capture_threads:
      t.receive(msg.copy())

    # Pass the message through to the output port, if appropriate.
    if not self._passthrough:
      pass
    elif self._texture_type == TextureType.POLYPHONIC:
      if msg.type == 'note_on':
        self._open_notes.add(msg.note)
      elif msg.type == 'note_off':
        self._open_notes.discard(msg.note)
      self._outport.send(msg)
    elif self._texture_type == TextureType.MONOPHONIC:
      assert len(self._open_notes) <= 1
      if msg.type not in ['note_on', 'note_off']:
        self._outport.send(msg)
      elif msg.type == 'note_off' and msg.note in self._open_notes:
        self._outport.send(msg)
        self._open_notes.remove(msg.note)
      elif msg.type == 'note_on':
        if self._open_notes:
          self._outport.send(
              mido.Message(
                  'note_off', note=self._open_notes.pop()))
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
      self._capture_threads.append(captor)
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
          'Either `stop_time` or `stop_signal` must be provided to '
          '`capture_sequence` call.')
    captor = self.start_capture(start_time, qpm, stop_time, stop_signal)
    captor.join()
    return captor.captured_sequence()

  @serialized
  def wait_for_event(self, signal=None, timeout=None):
    """Blocks until a matching mido.Message arrives or the timeout occurs.

    At least one of `signal` or `timeout` must be specified. If both are
    specified, returns when the first occurs.

    Args:
      signal: A MidiSignal to use as a signal to stop waiting.
      timeout: A float timeout in seconds.
    """
    if signal is None:
      time.sleep(timeout)
      return

    signal_pattern = str(signal)
    cond_var = None
    for regex, cond_var in self._signals:
      if regex.pattern == signal_pattern:
        break
    if cond_var is None:
      cond_var = threading.Condition(self._lock)
      self._signals[re.compile(signal_pattern)] = cond_var

    cond_var.wait(timeout=timeout)

  @serialized
  def start_metronome(self, start_time, qpm):
    """Starts or re-starts the metronome with the given arguments.

    Args:
      start_time: The wall time in seconds that the metronome is started on for
          synchronization and beat alignment. May be in the past.
      qpm: The quarter notes per minute to use.
    """
    if self._metronome is not None:
      self.stop_metronome()
    self._metronome = Metronome(self._outport, start_time, qpm)

  @serialized
  def stop_metronome(self):
    """Stops the metronome if it is currently running."""
    if self._metronome is None:
      return
    self._metronome.stop()
    self._metronome = None

  def start_playback(self, sequence, stay_alive=False):
    """Plays the notes in aNoteSequence via the MIDI output port.

    Args:
      sequence: The NoteSequence to play, with times based on the wall clock.
      stay_alive: A boolean specifying whether or not the player should stay
          alive waiting for updates after it plays the last note of the input
          sequence.
    Returns:
      The MidiPlayer thread handling playback to enable updating.
    """
    player = MidiPlayer(self._outport, sequence, stay_alive)
    player.start()
    return player
