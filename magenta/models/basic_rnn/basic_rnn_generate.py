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
"""Samples melodies from the model trained by basic_rnn_train.py

Melodies are sampled from the model by sampling a note from the RNN's output
distribution at a given timestep and feeding the result to the model as input
at the next timestep. The model is primed with a starting sequence of notes.
Many possible continuations of the primer are sampled from the model in a
minibatch.
"""

import ast
import basic_rnn_ops
import logging
import os
import os.path
import numpy as np
import sys
import tensorflow as tf

from magenta.lib import encoders
from magenta.lib import melodies_lib
from magenta.lib import midi_io


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('experiment_run_dir', '/tmp/basic_rnn/run1',
                           'Directory passed to basic_rnn_train.py where '
                           'training output was saved. Latest checkpoint is '
                           'loaded.')
tf.app.flags.DEFINE_string('hparams', '',
                           'String representation of Python dictionary '
                           'containing hyperparameter to value mapping. Must '
                           'be the same hyperparameter string passed into the '
                           'basic_rnn_train.py job that produced the '
                           'experiment output.')
tf.app.flags.DEFINE_string('primer_midi', '',
                           'Path to MIDI file containing melody to start '
                           'generating from.')
tf.app.flags.DEFINE_string('output_dir', '/tmp/basic_rnn_generated',
                           'Where output MIDI files will be saved.')
tf.app.flags.DEFINE_integer('num_steps', 32,
                            'How many steps to generate.')
tf.app.flags.DEFINE_integer('num_outputs', 16,
                            'How many samples to generate. One MIDI file will '
                            'be created for each.')


def make_graph(hparams_string='{}'):
  """Construct the model and return the graph.

  Hyperparameters are given in the hparams flag as a string
  representation of a Python dictionary.
  For example: '{"batch_size":64,"rnn_layer_sizes":[100,100]}'

  Args:
    hparams_string: A string literal of a Python dictionary. Keys are
        hyperparameter names, and values replace default values.

  Returns:
    tf.Graph instance which contains the TF ops.
  """
  with tf.Graph().as_default() as graph:
    with tf.device(lambda op: ""):
      hparams = basic_rnn_ops.default_hparams()
      hparams = hparams.parse(hparams_string)
      logging.info('hparams = %s', hparams.values())

      with tf.variable_scope('rnn_model'):
        # Define the type of RNN cell to use.
        cell = basic_rnn_ops.make_cell(hparams)

        # Construct dynamic_rnn inference.

        # Make a batch.
        melody_sequence = tf.placeholder(tf.float32,
                                         [hparams.batch_size, None,
                                          hparams.one_hot_length])
        lengths = tf.placeholder(tf.int32, [hparams.batch_size])

        # Make inference graph. That is, inputs to logits.
        (logits,
         initial_state,
         final_state) = basic_rnn_ops.dynamic_rnn_inference(
            melody_sequence, lengths, cell, hparams,
            zero_initial_state=False, parallel_iterations=1,
            swap_memory=True)

        softmax = tf.nn.softmax(tf.reshape(logits, [hparams.batch_size, -1]))

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('softmax', softmax)
      tf.add_to_collection('initial_state', initial_state)
      tf.add_to_collection('final_state', final_state)
      tf.add_to_collection('melody_sequence', melody_sequence)
      tf.add_to_collection('lengths', lengths)

  return graph


def make_onehot(int_list, one_hot_length):
  """Convert each int to a one-hot vector.
  A one-hot vector is 0 everywhere except at the index equal to the
  encoded value.
  For example: 5 as a one-hot vector is [0, 0, 0, 0, 0, 1, 0, 0, 0, ...]
  """
  return [[1.0 if j == i else 0.0 for j in xrange(one_hot_length)]
          for i in int_list]


def sampler_loop(graph, checkpoint_dir, primer, num_gen_steps):
  """Generate many melodies simulatneously given a primer.

  Generate melodies by sampling from model output and feeding it back into
  the model as input at every step.

  Args:
    graph: A tf.Graph instance containing the graph to sample from.
    checkpoint_dir: Directory to look for most recent model checkpoint in.
    primer: A Melody object.
    num_gen_steps: How many time steps to generate.

  Returns:
    List of generated melodies, each as a Melody object.
  """
  logits = graph.get_collection('logits')[0]
  softmax = graph.get_collection('softmax')[0]
  initial_state = graph.get_collection('initial_state')[0]
  final_state = graph.get_collection('final_state')[0]
  melody_sequence = graph.get_collection('melody_sequence')[0]
  lengths = graph.get_collection('lengths')[0]

  with graph.as_default():
    saver = tf.train.Saver()

  session = tf.Session(graph=graph)

  logging.info('Checkpoint dir: %s', checkpoint_dir)
  checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

  saver.restore(session, checkpoint_file)

  batch_size = logits.get_shape()[0].value

  # Convert primer Melody to model inputs.
  sequence_example, transpose_amount = encoders.basic_one_hot_encoder(primer)
  primer_input = [list(i.float_list.value) for i in sequence_example.feature_lists.feature_list['inputs'].feature]

  # Run model over primer sequence.
  primer_input_batch = np.tile([primer_input], (batch_size, 1, 1))
  state, _ = session.run(
      [final_state, logits],
      feed_dict={initial_state: np.zeros(initial_state.get_shape().as_list()),
                 melody_sequence: primer_input_batch,
                 lengths: np.full(batch_size, len(primer),
                                  dtype=int)})

  # Sample from model repeatedly to generate melodies.
  generated_sequences = [list() for i in xrange(batch_size)]
  last_outputs = [melody_sequence] * batch_size
  singleton_lengths = np.full(batch_size, 1, dtype=int)
  for i in xrange(num_gen_steps):
    input_batch = np.transpose(
        [make_onehot(last_outputs, basic_rnn_ops.NUM_CLASSES)], (1, 0, 2))
    state, batch_logits, batch_softmax = session.run(
      [final_state, logits, softmax],
      feed_dict={initial_state: state,
                 melody_sequence: input_batch,
                 lengths: singleton_lengths})
    last_outputs = [
        np.random.choice(basic_rnn_ops.NUM_CLASSES, p=p_dist.flatten())
        for p_dist in batch_softmax]
    for generated_seq, next_output in zip(generated_sequences, last_outputs):
      generated_seq.append(next_output)

  def decoder(event_list):
    return [e - melodies_lib.NUM_SPECIAL_EVENTS
            if e < melodies_lib.NUM_SPECIAL_EVENTS
            else e + 48 - transpose_amount
            for e in event_list]

  primer_event_list = list(primer)
  generated_melodies = []
  for seq in generated_sequences:
    melody = melodies_lib.Melody(steps_per_bar=primer.steps_per_bar)
    melody.from_event_list(primer_event_list + decoder(seq))
    generated_melodies.append(melody)

  return generated_melodies


def main(_):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  primer_sequence = midi_io.midi_file_to_sequence_proto(FLAGS.primer_midi)
  bpm = primer_sequence.tempos[0].bpm if len(primer_sequence.tempos) else 120.0

  extracted_melodies = melodies_lib.extract_melodies(primer_sequence,
                                                     min_bars=1,
                                                     min_unique_pitches=1)

  if not extracted_melodies:
    logging.info('No melodies were extracted from MIDI file %s'
                 % FLAGS.primer_midi)
    return

  graph = make_graph(hparams_string=FLAGS.hparams)

  checkpoint_dir = os.path.join(FLAGS.experiment_run_dir, 'train')
  
  generated = []
  while len(generated) < FLAGS.num_outputs:
    generated.extend(sampler_loop(graph, checkpoint_dir,
                                  extracted_melodies[0],
                                  FLAGS.num_steps))

  for i in xrange(FLAGS.num_outputs):
    sequence = generated[i].to_sequence(bpm=bpm)
    midi_io.sequence_proto_to_midi_file(
        sequence,
        os.path.join(FLAGS.output_dir, 'basic_rnn_sample_%d.mid' % i))

  logging.info('Wrote %d MIDI files to %s', FLAGS.num_outputs, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
