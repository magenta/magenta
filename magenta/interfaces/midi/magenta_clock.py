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

from magenta.interfaces.midi import midi_hub

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'output_ports',
    'magenta_in',
    'Comma-separated list of names of output MIDI ports.')
tf.app.flags.DEFINE_integer(
    'qpm',
    120,
    'The quarters per minute to use for the clock.')
tf.app.flags.DEFINE_integer(
    'clock_control_number',
    42,
    'The control change number to use with value 127 as a signal for a tick of '
    'the external clock. If None, an internal clock is used that ticks once '
    'per bar based on the qpm.')
tf.app.flags.DEFINE_string(
    'output_channels',
    '0',
    'Comma-separated list of 0-based MIDI channel numbers to output to.')
tf.app.flags.DEFINE_string(
    'log', 'WARN',
    'The threshold for what messages will be logged. DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  # Initialize MidiHub.
  hub = midi_hub.MidiHub(
      None, FLAGS.output_ports.split(','), midi_hub.TextureType.MONOPHONIC)

  cc = FLAGS.clock_control_number
  metronome_signals = (
    [midi_hub.MidiSignal(control=cc, value=127)] +
    [midi_hub.MidiSignal(control=cc, value=0)] * 3)

  channels = [int(channel) for channel in FLAGS.output_channels.split(',')]

  hub.start_metronome(
    FLAGS.qpm, start_time=0, signals=metronome_signals, channels=channels)

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    hub.stop_metronome()

  print 'Clock stopped.'


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
