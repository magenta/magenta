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

import time

# internal imports
import tensorflow as tf
import magenta

from magenta.interfaces.midi import midi_hub
from magenta.interfaces.midi import midi_interaction
from magenta.models.melody_rnn import melody_rnn_sequence_generator

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool(
    'list_ports',
    False,
    'Only list available MIDI ports.')
tf.app.flags.DEFINE_string(
    'input_port',
    'magenta_in',
    'The name of the input MIDI port.')
tf.app.flags.DEFINE_string(
    'output_port',
    'magenta_out',
    'The name of the output MIDI port.')
tf.app.flags.DEFINE_integer(
    'phrase_bars',
    None,
    'The number of bars of duration to use for the call and response phrases. '
    'If none, `end_call_control_number` must be specified.')
tf.app.flags.DEFINE_integer(
    'start_call_control_number',
    None,
    'The control change number to use as a signal to start the call phrase. If '
    'None, call will start immediately after response.')
tf.app.flags.DEFINE_integer(
    'end_call_control_number',
    None,
    'The control change number to use as a signal to end the call phrase. If '
    'None, `phrase_bars` must be specified.')
# TODO(adarob): Make the qpm adjustable by a control change signal.
tf.app.flags.DEFINE_integer(
    'qpm',
    90,
    'The quarters per minute to use for the metronome and generated sequence.')
tf.app.flags.DEFINE_string(
    'bundle_file',
    None,
    'The location of the bundle file to use.')
tf.app.flags.DEFINE_bool(
    'passthrough',
    False,
    'Passthrough MIDI from input_port to output_port.')
tf.app.flags.DEFINE_bool(
    'play_metronome',
    False,
    'Play a metronome during capture phase.')

# A map from a string generator name to its class.
_GENERATOR_MAP = melody_rnn_sequence_generator.get_generator_map()


def main(unused_argv):
  if FLAGS.list_ports:
    print "Input ports: '%s'" % (
        "', '".join(midi_hub.get_available_input_ports()))
    print "Ouput ports: '%s'" % (
        "', '".join(midi_hub.get_available_output_ports()))
    return

  if FLAGS.bundle_file is None:
    print '--bundle_file must be specified.'
    return

  if (FLAGS.end_call_control_number, FLAGS.phrase_bars).count(None) != 1:
    print('Exactly one of --end_call_control_number or --phrase_bars should be '
          'specified.')
    return

  try:
    bundle = magenta.music.sequence_generator_bundle.read_bundle_file(
        FLAGS.bundle_file)
  except magenta.music.sequence_generator_bundle.GeneratorBundleParseException:
    print 'Failed to parse bundle file: %s' % FLAGS.bundle_file
    return

  generator_id = bundle.generator_details.id
  if generator_id not in _GENERATOR_MAP:
    print "Unrecognized SequenceGenerator ID '%s' in bundle file: %s" % (
        generator_id, FLAGS.bundle_file)
    return
  generator = _GENERATOR_MAP[generator_id](checkpoint=None, bundle=bundle)
  generator.initialize()
  print "Loaded '%s' generator bundle from file '%s'." % (
      bundle.generator_details.id, FLAGS.bundle_file)

  if FLAGS.input_port not in midi_hub.get_available_input_ports():
    print "Opening '%s' as a virtual MIDI port for input." % FLAGS.input_port
  if FLAGS.output_port not in midi_hub.get_available_output_ports():
    print "Opening '%s' as a virtual MIDI port for output." % FLAGS.output_port
  hub = midi_hub.MidiHub(FLAGS.input_port, FLAGS.output_port,
                         midi_hub.TextureType.MONOPHONIC,
                         passthrough=FLAGS.passthrough)

  start_call_signal = (
      None if FLAGS.start_call_control_number is None else
      midi_hub.MidiSignal(control=FLAGS.start_call_control_number, value=127))
  end_call_signal = (
      None if FLAGS.end_call_control_number is None else
      midi_hub.MidiSignal(control=FLAGS.end_call_control_number, value=127))
  interaction = FollowerMidiInteraction(
      hub,
      FLAGS.qpm,
      generator,
      phrase_bars=FLAGS.phrase_bars,
      start_call_signal=start_call_signal,
      end_call_signal=end_call_signal,
      play_metronome=FLAGS.play_metronome)

  print ''
  print 'Instructions:'
  if FLAGS.start_call_control_number is not None:
    print ('When you want to begin the call phrase, signal control number %d '
           'with value 0.' % FLAGS.start_call_control_number)
  print 'Play when you hear the metronome ticking.'
  if FLAGS.phrase_bars is not None:
    print ('After %d bars (4 beats), Magenta will play its response.' %
           FLAGS.phrase_bars)
  else:
    print ('When you want to end the call phrase, signal control number %d '
           'with value 0' % FLAGS.end_call_control_number)
    print ('At the end of the current bar (4 beats), Magenta will play its '
           'response.')
  if FLAGS.start_call_control_number is not None:
    print ('Once the response completes, the interface will wait for you to '
           'signal a new call phrase using control number %d.' %
           FLAGS.start_call_control_number)
  else:
    print ('Once the response completes, the metronome will tick and you can '
           'play again.')

  print ''
  print 'To end the interaction, press CTRL-C.'

  interaction.start()
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    interaction.stop()

  print 'Interaction stopped.'


def console_entry_point():
  tf.app.run(main)


#### --- DEFINE MIDI INTERACTION CLASS --- ####


class FollowerMidiInteraction(midi_interaction.MidiInteraction):
  """Implementation of a MidiInteraction which follows external timing.

  Alternates between receiving input from the MidiHub ("call") and playing
  generated sequences ("response"). During the call stage, the input is captured
  and used to generate the response, which is then played back during the
  response stage. The timing and length of the call and response stages are set
  by external MIDI signals.

  Args:
    _midi_hub: The MidiHub to use for MIDI I/O.
    qpm: The quarters per minute to use for this interaction.
    sequence_generator: The SequenceGenerator to use to generate the responses
        in this interaction.
    quarters_per_bar: The number of quarter notes in each bar/measure.
    phrase_bars: The optional number of bars in each phrase. `end_call_signal`
        must be provided if None.
    start_call_signal: The control change number to use as a signal to start the
       call phrase. If None, call will start immediately after response.
    end_call_signal: The optional midi_hub.MidiSignal to use as a signal to stop
        the call phrase at the end of the current bar. `phrase_bars` must be
        provided if None.
  """

  def __init__(self,
               _midi_hub,
               qpm,
               sequence_generator,
               quarters_per_bar=4,
               phrase_bars=None,
               start_call_signal=None,
               end_call_signal=None,
               play_metronome=True):
    super(FollowerMidiInteraction, self).__init__(_midi_hub, qpm)
    self._sequence_generator = sequence_generator
    self._quarters_per_bar = quarters_per_bar
    self._phrase_bars = phrase_bars
    self._start_call_signal = start_call_signal
    self._end_call_signal = end_call_signal
    self._play_metronome = play_metronome

  def run(self):
    """The main loop for a real-time call and response interaction."""

    # We measure time in units of quarter notes.
    quarter_duration = 60.0 / self._qpm

    while not self._stop_signal.is_set():
      if self._start_call_signal is not None:
        # Wait for start signal.
        self._midi_hub.wait_for_event(self._start_call_signal)
        start_time = time.time()
        # Check to see if a stop has been requested.
        if self._stop_signal.is_set():
          break

      # Call stage.

      # Start the metronome at the beginning of the call stage.
      if self._play_metronome:
        self._midi_hub.start_metronome(self._qpm, start_time)

      # Start a captor at the beginning of the call stage.
      captor = self._midi_hub.start_capture(self._qpm, start_time)

      if self._phrase_bars is not None:
        # The duration of the call stage in quarter notes.
        # call_quarters = self._phrase_bars * self._quarters_per_bar
      else:
        # Wait for end signal.
        self._midi_hub.wait_for_event(self._end_call_signal)
        # The duration of the call stage in quarter notes.
        # We end the call stage at the end of the next bar that is at least
        # `predicathead_quarters` in the future.
        # end_time = time.time()
        call_duration = ((time.time() - start_time)
                         // quarter_duration) * quarter_duration
        quantized_end_time = start_time + call_duration

      # Set the metronome to stop at the appropriate time
      if self._play_metronome:
        self._midi_hub.stop_metronome(quantized_end_time, block=False)

      # Stop the captor at the appropriate time.
      captor.stop(stop_time=quantized_end_time)
      captured_sequence = captor.captured_sequence()

      # Check to see if a stop has been requested during capture.
      if self._stop_signal.is_set():
        break

      # Generate sequence options.
      response_start_time = quantized_end_time
      response_end_time = quantized_end_time + call_duration

      generator_options = magenta.protobuf.generator_pb2.GeneratorOptions()
      generator_options.generate_sections.add(
          start_time=response_start_time,
          end_time=response_end_time)

      # Generate response.
      response_sequence = self._sequence_generator.generate(
          captured_sequence, generator_options)

      # Check to see if a stop has been requested during generation.
      if self._stop_signal.is_set():
        break

      # Response stage.
      # Start response playback.
      self._midi_hub.start_playback(response_sequence)

  def stop(self):
    if self._start_call_signal is not None:
      self._midi_hub.wake_signal_waiters(self._start_call_signal)
    if self._end_call_signal is not None:
      self._midi_hub.wake_signal_waiters(self._end_call_signal)
    super(FollowerMidiInteraction, self).stop()


if __name__ == '__main__':
  console_entry_point()
