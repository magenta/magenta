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
"""A MIDI interface to the sequence generators.

Captures monophonic input MIDI sequences and plays back responses from the
sequence generator.
"""

import ast
import functools
from sys import stdout
import threading
import time

# internal imports
import mido
import tensorflow as tf
import magenta

from magenta.models.attention_rnn import attention_rnn_generator
from magenta.models.basic_rnn import basic_rnn_generator
from magenta.models.lookback_rnn import lookback_rnn_generator
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool(
    'list',
    False,
    'Only list available MIDI ports.')
tf.app.flags.DEFINE_string(
    'input_port',
    None,
    'The name of the input MIDI port.')
tf.app.flags.DEFINE_string(
    'output_port',
    None,
    'The name of the output MIDI port.')
tf.app.flags.DEFINE_integer(
    'start_capture_control_number',
    1,
    'The control change number to use as a signal to start '
    'capturing. Defaults to modulation wheel.')
tf.app.flags.DEFINE_integer(
    'start_capture_control_value',
    127,
    'The control change value to use as a signal to start '
    'capturing. If None, any control change with '
    'start_capture_control_number will start capture.')
tf.app.flags.DEFINE_integer(
    'stop_capture_control_number',
    1,
    'The control change number to use as a signal to stop '
    'capturing and generate. Defaults to the modulation '
    'wheel.')
tf.app.flags.DEFINE_integer(
    'stop_capture_control_value',
    0,
    'The control change value to use as a signal to stop '
    'capturing and generate. If None, any control change with'
    'stop_capture_control_number will stop capture.')
# TODO(adarob): Make the qpm adjustable by a control change signal.
tf.app.flags.DEFINE_integer(
    'qpm',
    90,
    'The quarters per minute to use for the metronome and generated sequence.')
# TODO(adarob): Make the number of bars to generate adjustable.
tf.app.flags.DEFINE_integer(
    'num_bars_to_generate',
    5,
    'The number of bars to generate each time.')
tf.app.flags.DEFINE_integer(
    'metronome_channel',
    0,
    'The MIDI channel on which to send the metronome click.')
tf.app.flags.DEFINE_integer(
    'metronome_playback_velocity',
    0,
    'The velocity of the generated playback metronome '
    'expressed as an integer between 0 and 127.')
tf.app.flags.DEFINE_string(
    'bundle_file',
    None,
    'The location of the bundle file to use. If specified, generator_name, '
    'checkpoint, and hparams cannot be specified.')
tf.app.flags.DEFINE_string(
    'generator_name',
    None,
    'The name of the SequenceGenerator being used.')
tf.app.flags.DEFINE_string(
    'checkpoint',
    None,
    'The training directory with checkpoint files or the path to a single '
    'checkpoint file for the model being used.')
tf.app.flags.DEFINE_string(
    'hparams',
    '{}',
    'String representation of a Python dictionary containing hyperparameter to '
    'value mappings. This mapping is merged with the default hyperparameters.')

# A map from a string generator name to its factory class.
_GENERATOR_FACTORY_MAP = {
    'attention_rnn': attention_rnn_generator,
    'basic_rnn': basic_rnn_generator,
    'lookback_rnn': lookback_rnn_generator,
}

_METRONOME_TICK_DURATION = 0.05
_METRONOME_PITCH = 95
# TODO(hanzorama): Make velocity adjustable by a control change signal.
_METRONOME_CAPTURE_VELOCITY = 64


def serialized(func):
  """Decorator to provide mutual exclusion for method using _lock attribute."""
  @functools.wraps(func)
  def serialized_method(self, *args, **kwargs):
    lock = getattr(self, '_lock')
    with lock:
      return func(self, *args, **kwargs)
  return serialized_method


def stdout_write_and_flush(s):
  stdout.write(s)
  stdout.flush()


class GeneratorException(Exception):
  """An exception raised by the Generator class."""
  pass


class Generator(object):
  """A class wrapping a SequenceGenerator.

  Args:
    generator_name: The name of the generator to wrap. Must be present in
        _GENERATOR_FACTORY_MAP.
    num_bars_to_generate: The number of bars to generate on each call.
        Assumes 4/4 time.
    hparams: A Python dictionary containing hyperparameter to value mappings to
        be merged with the default hyperparameters.
    checkpoint: The training directory with checkpoint files or the path to a
        single checkpoint file for the model being used.
  Raises:
    GeneratorException: If an invalid generator name is given or no training
        directory is given.
  """

  def __init__(
      self,
      generator_name,
      num_bars_to_generate,
      hparams,
      checkpoint=None,
      bundle_file=None):
    self._num_bars_to_generate = num_bars_to_generate

    if not checkpoint and not bundle_file:
      raise GeneratorException(
          'No generator checkpoint or bundle location supplied.')
    if (checkpoint or generator_name or hparams) and bundle_file:
      raise GeneratorException(
          'Cannot specify both bundle file and checkpoint, generator_name, '
          'or hparams.')

    bundle = None
    if bundle_file:
      bundle = magenta.music.read_bundle_file(bundle_file)
      generator_name = bundle.generator_details.id

    if generator_name not in _GENERATOR_FACTORY_MAP:
      raise GeneratorException('Invalid generator name given: %s',
                               generator_name)

    generator = _GENERATOR_FACTORY_MAP[generator_name].create_generator(
        checkpoint=checkpoint, bundle=bundle, hparams=hparams)
    generator.initialize()

    self._generator = generator

  def generate_melody(self, input_sequence):
    """Calls the SequenceGenerator and returns the generated NoteSequence."""
    # TODO(fjord): Align generation time on a measure boundary.
    notes_by_end_time = sorted(input_sequence.notes, key=lambda n: n.end_time)
    last_end_time = notes_by_end_time[-1].end_time if notes_by_end_time else 0

    # Assume 4/4 time signature and a single tempo.
    qpm = input_sequence.tempos[0].qpm
    seconds_to_generate = (60.0 / qpm) * 4 * self._num_bars_to_generate

    generator_options = generator_pb2.GeneratorOptions()
    section = generator_options.generate_sections.add()
    # Start generating 1 quarter note after the sequence ends.
    section.start_time_seconds = last_end_time + (60.0 / qpm)
    section.end_time_seconds = section.start_time_seconds + seconds_to_generate

    return self._generator.generate(input_sequence, generator_options)


class Metronome(threading.Thread):
  """A thread implementing a MIDI metronome.

  Attributes:
    _outport: The Mido port for sending messages.
    _qpm: The integer quarters per minute to signal on.
    _stop_metronome: A boolean specifying whether the metronome should stop.
    _velocity: The velocity of the metronome's MIDI note_on message.
  Args:
    outport: The Mido port for sending messages.
    qpm: The integer quarters per minute to signal on.
    velocity: The velocity of the metronome's MIDI note_on message.
  """
  daemon = True

  def __init__(self, outport, qpm, clock_start_time, velocity):
    self._outport = outport
    self._qpm = qpm
    self._stop_metronome = False
    self._clock_start_time = clock_start_time
    self._velocity = velocity
    super(Metronome, self).__init__()

  def run(self):
    """Outputs metronome tone on the qpm interval until stop signal received."""
    period = 60. / self._qpm
    sleep_offset = 0
    while not self._stop_metronome:
      now = time.time()
      next_tick_time = now + period - ((now - self._clock_start_time) % period)
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

      self._outport.send(mido.Message(type='note_on', note=_METRONOME_PITCH,
                                      channel=FLAGS.metronome_channel,
                                      velocity=self._velocity))
      time.sleep(_METRONOME_TICK_DURATION)
      self._outport.send(mido.Message(type='note_off', note=_METRONOME_PITCH,
                                      channel=FLAGS.metronome_channel))

  def stop(self):
    """Signals for the metronome to stop and joins thread."""
    self._stop_metronome = True
    self.join()


class MonoMidiPlayer(threading.Thread):
  """A thread for playing back a monophonic, sorted NoteSequence via MIDI.

  Attributes:
    _outport: The Mido port for sending messages.
    _sequence: The monohponic, chronologically sorted NoteSequence to play.
    _stop_playback: A boolean specifying whether the playback should stop.
  Args:
    outport: The Mido port for sending messages.
    sequence: The monohponic, chronologically sorted NoteSequence to play.
    metronome_velocity: The velocity of the metronome's MIDI note_on message.
  Raises:
    ValueError: The NoteSequence contains multiple tempos.
  """
  daemon = True

  def __init__(self, outport, sequence, metronome_velocity):
    self._outport = outport
    self._sequence = sequence
    self._stop_playback = False
    if len(sequence.tempos) != 1:
      raise ValueError('The NoteSequence contains multiple tempos.')
    self._metronome = Metronome(self._outport, sequence.tempos[0].qpm,
                                time.time(), metronome_velocity)
    super(MonoMidiPlayer, self).__init__()

  def run(self):
    """Plays back the NoteSequence until it ends or stop signal is received.

    Raises:
      ValueError: The NoteSequence is not monophonic and chronologically sorted.
    """
    stdout_write_and_flush('Playing sequence...')
    self._metronome.start()
    # Wall start time.
    play_start = time.time()
    # Time relative to start of NoteSequence.
    playhead = 0
    for note in self._sequence.notes:
      if self._stop_playback:
        self._outport.panic()
        return

      stdout_write_and_flush('.')
      if note.start_time < playhead:
        raise ValueError(
            'The NoteSequence is not monophonic and chronologically sorted.')
      playhead = note.start_time
      delta = playhead - (time.time() - play_start)
      if delta > 0:
        time.sleep(delta)
      self._outport.send(
          mido.Message(
              'note_on', note=note.pitch, velocity=note.velocity))

      if self._stop_playback:
        self._outport.panic()
        return
      if note.end_time < playhead:
        raise ValueError(
            'The NoteSequence is not monophonic and chronologically sorted.')
      playhead = note.end_time
      delta = playhead - (time.time() - play_start)
      if delta > 0:
        time.sleep(delta)
      self._outport.send(mido.Message('note_off', note=note.pitch))
    self._metronome.stop()
    stdout_write_and_flush('Done\n')

  def stop(self):
    """Signals for the playback and metronome to stop and joins thread."""
    self._stop_playback = True
    self._metronome.stop()
    self.join()


class MonoMidiHub(object):
  """A MIDI interface for capturing and playing monophonic NoteSequences.

  Attributes:
    _inport: The Mido port for receiving messages.
    _outport: The Mido port for sending messages.
    _lock: An RLock used for thread-safety.
    _capture_sequence: The NoteSequence being built from MIDI messages currently
        being captured or having been captured in the previous session.
    _control_cvs: A dictionary mapping (<control change number>,) and
        (<control change number>, <control change value>) to a condition
        variable that will be notified when a matching control change messsage
        is received.
    _player: A thread for playing back NoteSequences via the MIDI output port.
  Args:
    input_midi_port: The string MIDI port name to use for input.
    output_midi_port: The string MIDI port name to use for output.
  """

  def __init__(self, input_midi_port, output_midi_port):
    self._inport = mido.open_input(input_midi_port)
    self._outport = mido.open_output(output_midi_port)
    # This lock is used by the serialized decorator.
    self._lock = threading.RLock()
    self._control_cvs = dict()
    self._player = None
    self._capture_start_time = None
    self._sequence_start_time = None

  def _timestamp_and_capture_message(self, msg):
    """Stamps message with current time and passes it to the capture handler."""
    msg.time = time.time()
    self._capture_message(msg)

  @serialized
  def _capture_message(self, msg):
    """Handles a single incoming MIDI message during capture. Used as callback.

    If the message is a control change, notifies threads waiting on the
    appropriate condition variable.

    If the message is a note_on event, ends the previous note (if applicable)
    and opens a new note in the capture sequence. Also forwards the message to
    the output MIDI port. Ignores repeated note_on events.

    If the message is a note_off event matching the current open note in the
    capture sequence, ends that note and forwards the message to the output MIDI
    port.

    Args:
      msg: The mido.Message MIDI message to handle.
    """
    if msg.type == 'control_change':
      control_tuples = [(msg.control,), (msg.control, msg.value)]

      for control_tuple in control_tuples:
        if control_tuple in self._control_cvs:
          self._control_cvs[control_tuple].notify_all()
      return

    last_note = (self.captured_sequence.notes[-1] if
                 self.captured_sequence.notes else None)
    if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
      if (last_note is None or last_note.pitch != msg.note or
          last_note.end_time > 0):
        # This is not the note we're looking for. Drop it.
        return

      last_note.end_time = msg.time - self._sequence_start_time
      self._outport.send(msg)
      stdout_write_and_flush('.')

    elif msg.type == 'note_on':
      if self._sequence_start_time is None:
        # This is the first note.
        # Find the sequence start time based on the start of the most recent
        # quarter note. This ensures that the sequence start time lines up with
        # a metronome tick.
        period = 60. / self.captured_sequence.tempos[0].qpm
        self._sequence_start_time = msg.time - (
            (msg.time - self._capture_start_time) % period)
      elif last_note.end_time == 0:
        if last_note.pitch == msg.note:
          # This is just a repeat of the previous message.
          return
        # End the previous note.
        last_note.end_time = msg.time - self._sequence_start_time
        self._outport.send(mido.Message('note_off', note=last_note.pitch))

      self._outport.send(msg)
      new_note = self.captured_sequence.notes.add()
      new_note.start_time = msg.time - self._sequence_start_time
      new_note.pitch = msg.note
      new_note.velocity = msg.velocity
      stdout_write_and_flush('.')

  @serialized
  def start_capture(self, qpm):
    """Starts a capture session.

    Initializes a new capture sequence, sets the capture callback on the input
    port, and starts the metronome.

    Args:
      qpm: The integer quarters per minute to use for the metronome and captured
          sequence.
    Raises:
      RuntimeError: Already in a capture session.
    """
    if self._inport.callback is not None:
      raise RuntimeError('Already in a capture session.')

    self.captured_sequence = music_pb2.NoteSequence()
    self.captured_sequence.tempos.add().qpm = qpm
    self._sequence_start_time = None
    self._capture_start_time = time.time()
    self._inport.callback = self._timestamp_and_capture_message
    self._metronome = Metronome(self._outport, qpm, self._capture_start_time,
                                _METRONOME_CAPTURE_VELOCITY)

    self._metronome.start()

  @serialized
  def stop_capture(self):
    """Stops the capture session and returns the captured sequence.

    Resets the capture callback on the input port, closes the final open note
    (if applicable), stops the metronome, and returns the captured sequence.

    Returns:
        The captured NoteSequence.
    Raises:
      RuntimeError: Not in a capture session.
    """
    if self._inport.callback is None:
      raise RuntimeError('Not in a capture session.')

    self._inport.callback = None

    self._metronome.stop()
    last_note = (self.captured_sequence.notes[-1] if
                 self.captured_sequence.notes else None)
    if last_note is not None and last_note.end_time == 0:
      last_note.end_time = time.time() - self._sequence_start_time
    stdout_write_and_flush('Done\n')
    return self.captured_sequence

  @serialized
  def wait_for_control_signal(self, control_number, control_value=None):
    """Blocks until a specific control signal arrives.

    Args:
      control_number: The integer control change number.
      control_value: The integer control change value or None if any is
         acceptable.

    """
    if self._inport.callback is None:
      # Not in a capture session.
      for msg in self._inport:
        if (msg.type == 'control_change' and msg.control == control_number and
            (control_value is None or msg.value == control_value)):
          return
    else:
      # In a capture session.
      control_tuple = ((control_number,) if control_value is None else
                       (control_number, control_value))
      if control_tuple not in self._control_cvs:
        self._control_cvs[control_tuple] = threading.Condition(self._lock)
      self._control_cvs[control_tuple].wait()

  def start_playback(self, sequence, metronome_velocity):
    """Plays the monophonic, sorted NoteSequence through the MIDI output port.

    Stops any previously playing sequences.

    Args:
      sequence: The monohponic, chronologically sorted NoteSequence to play.
      metronome_velocity: The velocity of the metronome's MIDI note_on message.
    """
    self.stop_playback()
    self._player = MonoMidiPlayer(self._outport, sequence, metronome_velocity)
    self._player.start()

  def stop_playback(self):
    """Stops any active sequence playback."""
    if self._player is not None and self._player.is_alive():
      self._player.stop()
      stdout_write_and_flush('Stopped\n')


def main(unused_argv):
  if FLAGS.list:
    print "Input ports: '" + "', '".join(mido.get_input_names()) + "'"
    print "Output ports: '" + "', '".join(mido.get_output_names()) + "'"
    return

  if FLAGS.input_port is None or FLAGS.output_port is None:
    print '--inport_port and --output_port must be specified.'
    return

  if (FLAGS.start_capture_control_number == FLAGS.stop_capture_control_number
      and
      (FLAGS.start_capture_control_value == FLAGS.stop_capture_control_value or
       FLAGS.start_capture_control_value is None or
       FLAGS.stop_capture_control_value is None)):
    print('If using the same number for --start_capture_control_number and '
          '--stop_capture_control_number, --start_capture_control_value and '
          '--stop_capture_control_value must both be defined and unique.')
    return

  if not 0 <= FLAGS.metronome_playback_velocity <= 127:
    print 'The metronome_playback_velocity must be an integer between 0 and 127'
    return

  generator = Generator(
      FLAGS.generator_name,
      FLAGS.num_bars_to_generate,
      ast.literal_eval(FLAGS.hparams if FLAGS.hparams else '{}'),
      FLAGS.checkpoint,
      FLAGS.bundle_file)
  hub = MonoMidiHub(FLAGS.input_port, FLAGS.output_port)

  stdout_write_and_flush('Waiting for start control signal...\n')
  while True:
    hub.wait_for_control_signal(FLAGS.start_capture_control_number,
                                FLAGS.start_capture_control_value)
    hub.stop_playback()
    hub.start_capture(FLAGS.qpm)
    stdout_write_and_flush('Capturing notes until stop control signal...')
    hub.wait_for_control_signal(FLAGS.stop_capture_control_number,
                                FLAGS.stop_capture_control_value)
    captured_sequence = hub.stop_capture()

    stdout_write_and_flush('Generating response...')
    generated_sequence = generator.generate_melody(captured_sequence)
    stdout_write_and_flush('Done\n')

    hub.start_playback(generated_sequence, FLAGS.metronome_playback_velocity)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
