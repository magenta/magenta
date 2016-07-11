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
"""Generate melodies from a trained checkpoint of a melody RNN model."""

import ast
import logging
import os
import random
import sys
import time

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import midi_io

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run_dir', '/tmp/melody_rnn/run1',
                           'Path to the directory where the latest checkpoint '
                           'will be loaded from.')
tf.app.flags.DEFINE_string('hparams', '{}',
                           'String representation of Python dictionary '
                           'containing hyperparameter to value mapping. This '
                           'mapping is merged with the default '
                           'hyperparameters.')
tf.app.flags.DEFINE_string('primer_melody', '',
                           'A string representation of a Python list of '
                           'melodies_lib.Melody event values. For example: '
                           '"[60, -2, 60, -2, 67, -2, 67, -2]". If specified, '
                           'this melody will be used as the priming melody. '
                           'If a priming melody is not specified, melodies '
                           'will be generated from scratch.')
tf.app.flags.DEFINE_string('primer_midi', '',
                           'The path to a MIDI file containing a melody that '
                           'will be used as a priming melody. If a primer '
                           'melody is not specified, melodies will be '
                           'generated from scratch.')
tf.app.flags.DEFINE_string('output_dir', '/tmp/melody_rnn_generated',
                           'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer('num_steps', 32,
                            'The total number of steps the generated melodies '
                            'should be, priming melody length + generated '
                            'steps. Each step is a 16th of a bar.')
tf.app.flags.DEFINE_integer('num_outputs', 16,
                            'The number of melodies to generate. One MIDI '
                            'file will be created for each.')


def run_generate(graph, train_dir, output_dir, melody_encoder_decoder,
                 primer_melody, num_steps, bpm):
  """Generates melodies and saves them as MIDI files.

  Args:
    graph: A tf.Graph object containing the model.
    train_dir: The path to the directory where the latest checkpoint will be
        loaded from.
    output_dir: The path to the directory where MIDI files will be saved to.
    melody_encoder_decoder: A melodies_lib.MelodyEncoderDecoder object.
    primer_melody: A melodies_lib.Melody object that will be used as the
        priming melody. If the priming melody is empty, melodies will be
        generated from scratch.
    num_steps: The total number of steps the final melodies should be,
        priming melody + generated steps. Each step is a 16th of a bar.
    bpm: The tempo in beats per minute that the generated MIDI files will have.
  """
  inputs = graph.get_collection('inputs')[0]
  initial_state = graph.get_collection('initial_state')[0]
  final_state = graph.get_collection('final_state')[0]
  softmax = graph.get_collection('softmax')[0]
  batch_size = softmax.get_shape()[0].value

  transpose_amount = primer_melody.squash(
      melody_encoder_decoder.min_note, melody_encoder_decoder.max_note,
      melody_encoder_decoder.transpose_to_key)

  melodies = []
  for _ in xrange(batch_size):
    melody = melodies_lib.Melody()
    if len(primer_melody):
      melody.from_event_list(primer_melody.events)
    else:
      melody.events = [random.randint(melody_encoder_decoder.min_note,
                                      melody_encoder_decoder.max_note)]
    melodies.append(melody)

  with graph.as_default():
    saver = tf.train.Saver()
    with tf.Session() as sess:
      checkpoint_file = tf.train.latest_checkpoint(train_dir)
      logging.info('Checkpoint used: %s\nGenerating melodies...',
                   checkpoint_file)
      saver.restore(sess, checkpoint_file)

      _final_state = None
      for i in xrange(num_steps - len(primer_melody)):
        if i == 0:
          _inputs = melody_encoder_decoder.get_inputs_batch(melodies,
                                                            full_length=True)
          _initial_state = sess.run(initial_state)
        else:
          _inputs = melody_encoder_decoder.get_inputs_batch(melodies)
          _initial_state = _final_state

        feed_dict = {inputs: _inputs, initial_state: _initial_state}
        _final_state, _softmax = sess.run([final_state, softmax], feed_dict)
        melody_encoder_decoder.extend_melodies(melodies, _softmax)

  date_and_time = time.strftime("%Y-%m-%d_%H-%M-%S")
  digits = len(str(len(melodies)))
  for i, melody in enumerate(melodies):
    melody.transpose(-transpose_amount)
    sequence = melody.to_sequence(bpm=bpm)
    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(output_dir, midi_filename)
    midi_io.sequence_proto_to_midi_file(sequence, midi_path)

  logging.info('Wrote %d MIDI files to %s', len(melodies), output_dir)


def run(melody_encoder_decoder, build_graph):
  """Generates melodies and saves them as MIDI files.

  Args:
    melody_encoder_decoder: A melodies_lib.MelodyEncoderDecoder object specific
        to your model.
    build_graph: A function that when called, returns the tf.Graph object for
        your model. The function will be passed the parameters:
        (mode, hparams_string, input_size, num_classes, sequence_example_file).
        For an example usage, check out models/basic_rnn/basic_rnn_graph.py.
  """
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  hparams = ast.literal_eval(FLAGS.hparams if FLAGS.hparams else '{}')
  hparams['batch_size'] = FLAGS.num_outputs
  hparams_string = repr(hparams)

  graph = build_graph('generate',
                      hparams_string,
                      melody_encoder_decoder.input_size,
                      melody_encoder_decoder.num_classes)

  train_dir = os.path.join(FLAGS.run_dir, 'train')

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  primer_melody = melodies_lib.Melody()
  bpm = melodies_lib.DEFAULT_BEATS_PER_MINUTE
  if FLAGS.primer_melody:
    primer_melody.from_event_list(ast.literal_eval(FLAGS.primer_melody))
  elif FLAGS.primer_midi:
    primer_sequence = midi_io.midi_file_to_sequence_proto(FLAGS.primer_midi)
    if len(primer_sequence.tempos):
      bpm = primer_sequence.tempos[0].bpm
    extracted_melodies = melodies_lib.extract_melodies(
        primer_sequence, min_bars=0, min_unique_pitches=1)
    if extracted_melodies:
      primer_melody = extracted_melodies[0]
    else:
      logging.info('No melodies were extracted from the MIDI file %s. '
                   'Melodies will be generated from scratch.' %
                   FLAGS.primer_midi)

  run_generate(graph, train_dir, FLAGS.output_dir, melody_encoder_decoder,
               primer_melody, FLAGS.num_steps, bpm)


if __name__ == '__main__':
  tf.app.run()
