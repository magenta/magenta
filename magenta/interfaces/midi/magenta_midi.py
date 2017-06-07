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
from functools import partial
import re
import threading
import time

# internal imports
import tensorflow as tf
import magenta

from magenta.interfaces.midi import midi_hub
from magenta.interfaces.midi import midi_interaction
from magenta.models.drums_rnn import drums_rnn_sequence_generator
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_sequence_generator
from magenta.models.polyphony_rnn import polyphony_sequence_generator

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool(
    'list_ports',
    False,
    'Only list available MIDI ports.')
tf.app.flags.DEFINE_string(
    'input_ports',
    'magenta_in',
    'Comma-separated list of names of the input MIDI ports.')
tf.app.flags.DEFINE_string(
    'output_ports',
    'magenta_out',
    'Comma-separated list of names of the output MIDI ports.')
tf.app.flags.DEFINE_bool(
    'passthrough',
    True,
    'Whether to pass input messages through to the output port.')
tf.app.flags.DEFINE_integer(
    'clock_control_number',
    None,
    'The control change number to use with value 127 as a signal for a tick of '
    'the external clock. If None, an internal clock is used that ticks once '
    'per bar based on the qpm.')
tf.app.flags.DEFINE_integer(
    'end_call_control_number',
    None,
    'The control change number to use with value 127 as a signal to end the '
    'call phrase on the next tick.')
tf.app.flags.DEFINE_integer(
    'panic_control_number',
    None,
    'The control change number to use with value 127 as a panic signal to '
    'close open notes and clear playback sequence.')
tf.app.flags.DEFINE_integer(
    'mutate_control_number',
    None,
    'The control change number to use with value 127 as a mutate signal to '
    'generate a new response using the current response sequence as a seed.')
tf.app.flags.DEFINE_integer(
    'min_listen_ticks_control_number',
    None,
    'The control change number to use for controlling minimum listen duration. '
    'The value for the control number will be used in clock ticks. Inputs less '
    'than this length will be ignored.')
tf.app.flags.DEFINE_integer(
    'max_listen_ticks_control_number',
    None,
    'The control change number to use for controlling maximum listen duration. '
    'The value for the control number will be used in clock ticks. After this '
    'number of ticks, a response will automatically be generated. A 0 value '
    'signifies infinite duration.')
tf.app.flags.DEFINE_integer(
    'response_ticks_control_number',
    None,
    'The control change number to use for controlling response duration. The '
    'value for the control number will be used in clock ticks. If not set, the '
    'response duration will match the call duration.')
tf.app.flags.DEFINE_integer(
    'temperature_control_number',
    None,
    'The control change number to use for controlling softmax temperature.'
    'The value of control changes with this number will be used to set the '
    'temperature in a linear range between 0.1 and 2.')
tf.app.flags.DEFINE_boolean(
    'allow_overlap',
    False,
    'Whether to allow the call to overlap with the response.')
tf.app.flags.DEFINE_boolean(
    'enable_metronome',
    True,
    'Whether to enable the metronome. Ignored if `clock_control_number` is '
    'provided.')
tf.app.flags.DEFINE_integer(
    'metronome_channel',
    1,
    'The 0-based MIDI channel to output the metronome on. Ignored if '
    '`enable_metronome` is False or `clock_control_number` is provided.')
tf.app.flags.DEFINE_integer(
    'qpm',
    120,
    'The quarters per minute to use for the metronome and generated sequence. '
    'Overriden by values of control change signals for `tempo_control_number`.')
tf.app.flags.DEFINE_integer(
    'tempo_control_number',
    None,
    'The control change number to use for controlling tempo. qpm will be set '
    'to 60 more than the value of the control change.')
tf.app.flags.DEFINE_integer(
    'loop_control_number',
    None,
    'The control number to use for determining whether to loop the response. '
    'A value of 127 turns looping on and any other value turns it off.')
tf.app.flags.DEFINE_string(
    'bundle_files',
    None,
    'A comma-separated list of the location of the bundle files to use.')
tf.app.flags.DEFINE_integer(
    'generator_select_control_number',
    None,
    'The control number to use for selecting between generators when multiple '
    'bundle files are specified. Required unless only a single bundle file is '
    'specified.')
tf.app.flags.DEFINE_integer(
    'state_control_number',
    None,
    'The control number to use for sending the state. A value of 0 represents '
    '`IDLE`, 1 is `LISTENING`, and 2 is `RESPONDING`.')
tf.app.flags.DEFINE_float(
    'playback_offset',
    0.0,
    'Time in seconds to adjust playback time by.')
tf.app.flags.DEFINE_integer(
    'playback_channel',
    0,
    'MIDI channel to send play events.')
tf.app.flags.DEFINE_boolean(
    'learn_controls',
    False,
    'Whether to allow programming of control flags on startup.')
tf.app.flags.DEFINE_string(
    'log', 'WARN',
    'The threshold for what messages will be logged. DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')

_CONTROL_FLAGS = [
    'clock_control_number',
    'end_call_control_number',
    'panic_control_number',
    'mutate_control_number',
    'min_listen_ticks_control_number',
    'max_listen_ticks_control_number',
    'response_ticks_control_number',
    'temperature_control_number',
    'tempo_control_number',
    'loop_control_number',
    'generator_select_control_number',
    'state_control_number']

# A map from a string generator name to its class.
_GENERATOR_MAP = melody_rnn_sequence_generator.get_generator_map()
_GENERATOR_MAP.update(drums_rnn_sequence_generator.get_generator_map())
_GENERATOR_MAP.update(pianoroll_rnn_nade_sequence_generator.get_generator_map())
_GENERATOR_MAP.update(polyphony_sequence_generator.get_generator_map())


class CCMapper(object):
  """A class for mapping control change numbers to specific controls.

  Args:
    cc_map: A dictionary containing mappings from signal names to control
        change numbers (or None). This dictionary will be updated by the class.
    midi_hub_: An initialized MidiHub to receive inputs from.
  """

  def __init__(self, cc_map, midi_hub_):
    self._cc_map = cc_map
    self._signals = cc_map.keys()
    self._midi_hub = midi_hub_
    self._update_event = threading.Event()

  def _print_instructions(self):
    """Prints instructions for mapping control changes."""
    print ('Enter the index of a signal to set the control change for, or `q` '
           'when done.')
    fmt = '{:>6}\t{:<20}\t{:>6}'
    print fmt.format('Index', 'Control', 'Current')
    for i, signal in enumerate(self._signals):
      print fmt.format(i + 1, signal, self._cc_map.get(signal))
    print

  def _update_signal(self, signal, msg):
    """Updates mapping for the signal to the message's control change.

    Args:
      signal: The name of the signal to update the control change for.
      msg: The mido.Message whose control change the signal should be set to.
    """
    if msg.control in self._cc_map.values():
      print 'Control number %d is already assigned. Ignoring.' % msg.control
    else:
      self._cc_map[signal] = msg.control
      print 'Assigned control number %d to `%s`.' % (msg.control, signal)
    self._update_event.set()

  def update_map(self):
    """Enters a loop that receives user input to set signal controls."""
    while True:
      print
      self._print_instructions()
      response = raw_input('Selection: ')
      if response == 'q':
        return
      try:
        signal = self._signals[int(response) - 1]
      except (ValueError, IndexError):
        print 'Invalid response:', response
        continue
      self._update_event.clear()
      self._midi_hub.register_callback(
          partial(self._update_signal, signal),
          midi_hub.MidiSignal(type='control_change'))
      print('Send a control signal using the control number you wish to '
            'associate with `%s`.' % signal)
      self._update_event.wait()


def _validate_flags():
  """Returns True if flag values are valid or prints error and returns False."""
  if FLAGS.list_ports:
    print "Input ports: '%s'" % (
        "', '".join(midi_hub.get_available_input_ports()))
    print "Ouput ports: '%s'" % (
        "', '".join(midi_hub.get_available_output_ports()))
    return False

  if FLAGS.bundle_files is None:
    print '--bundle_files must be specified.'
    return False

  if (len(FLAGS.bundle_files.split(',')) > 1 and
      FLAGS.generator_select_control_number is None):
    tf.logging.warning(
        'You have specified multiple bundle files (generators), without '
        'setting `--generator_select_control_number`. You will only be able to '
        'use the first generator (%s).',
        FLAGS.bundle_files[0])

  return True


def _load_generator_from_bundle_file(bundle_file):
  """Returns initialized generator from bundle file path or None if fails."""
  try:
    bundle = magenta.music.sequence_generator_bundle.read_bundle_file(
        bundle_file)
  except magenta.music.sequence_generator_bundle.GeneratorBundleParseException:
    print 'Failed to parse bundle file: %s' % FLAGS.bundle_file
    return None

  generator_id = bundle.generator_details.id
  if generator_id not in _GENERATOR_MAP:
    print "Unrecognized SequenceGenerator ID '%s' in bundle file: %s" % (
        generator_id, FLAGS.bundle_file)
    return None

  generator = _GENERATOR_MAP[generator_id](checkpoint=None, bundle=bundle)
  generator.initialize()
  print "Loaded '%s' generator bundle from file '%s'." % (
      bundle.generator_details.id, bundle_file)
  return generator


def _print_instructions():
  """Prints instructions for interaction based on the flag values."""
  print ''
  print 'Instructions:'
  print 'Start playing  when you want to begin the call phrase.'
  if FLAGS.end_call_control_number is not None:
    print ('When you want to end the call phrase, signal control number %d '
           'with value 127, or stop playing and wait one clock tick.'
           % FLAGS.end_call_control_number)
  else:
    print ('When you want to end the call phrase, stop playing and wait one '
           'clock tick.')
  print ('Once the response completes, the interface will wait for you to '
         'begin playing again to start a new call phrase.')
  print ''
  print 'To end the interaction, press CTRL-C.'


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  if not _validate_flags():
    return

  # Load generators.
  generators = []
  for bundle_file in FLAGS.bundle_files.split(','):
    generators.append(_load_generator_from_bundle_file(bundle_file))
    if generators[-1] is None:
      return

  # Initialize MidiHub.
  hub = midi_hub.MidiHub(FLAGS.input_ports.split(','),
                         FLAGS.output_ports.split(','),
                         midi_hub.TextureType.POLYPHONIC,
                         passthrough=FLAGS.passthrough,
                         playback_channel=FLAGS.playback_channel,
                         playback_offset=FLAGS.playback_offset)

  control_map = {re.sub('_control_number$', '', f): FLAGS.__getattr__(f)
                 for f in _CONTROL_FLAGS}
  if FLAGS.learn_controls:
    CCMapper(control_map, hub).update_map()

  if control_map['clock'] is None:
    # Set the tick duration to be a single bar, assuming a 4/4 time signature.
    clock_signal = None
    tick_duration = 4 * (60. / FLAGS.qpm)
  else:
    clock_signal = midi_hub.MidiSignal(
        control=control_map['clock'], value=127)
    tick_duration = None

  end_call_signal = (
      None if control_map['end_call'] is None else
      midi_hub.MidiSignal(control=control_map['end_call'], value=127))
  panic_signal = (
      None if control_map['panic'] is None else
      midi_hub.MidiSignal(control=control_map['panic'], value=127))
  mutate_signal = (
      None if control_map['mutate'] is None else
      midi_hub.MidiSignal(control=control_map['mutate'], value=127))
  metronome_channel = (
      FLAGS.metronome_channel if FLAGS.enable_metronome else None)
  interaction = midi_interaction.CallAndResponseMidiInteraction(
      hub,
      generators,
      FLAGS.qpm,
      FLAGS.generator_select_control_number,
      clock_signal=clock_signal,
      tick_duration=tick_duration,
      end_call_signal=end_call_signal,
      panic_signal=panic_signal,
      mutate_signal=mutate_signal,
      allow_overlap=FLAGS.allow_overlap,
      metronome_channel=metronome_channel,
      min_listen_ticks_control_number=control_map['min_listen_ticks'],
      max_listen_ticks_control_number=control_map['max_listen_ticks'],
      response_ticks_control_number=control_map['response_ticks'],
      tempo_control_number=control_map['tempo'],
      temperature_control_number=control_map['temperature'],
      loop_control_number=control_map['loop'],
      state_control_number=control_map['state'])

  _print_instructions()

  interaction.start()
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    interaction.stop()

  print 'Interaction stopped.'


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
