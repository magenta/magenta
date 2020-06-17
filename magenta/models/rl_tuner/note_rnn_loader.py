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

"""Defines a class and operations for the MelodyRNN model.

Note RNN Loader allows a basic melody prediction LSTM RNN model to be loaded
from a checkpoint file, primed, and used to predict next notes.

This class can be used as the q_network and target_q_network for the RLTuner
class.

The graph structure of this model is similar to basic_rnn, but more flexible.
It allows you to either train it with data from a queue, or just 'call' it to
produce the next action.

It also provides the ability to add the model's graph to an existing graph as a
subcomponent, and then load variables from a checkpoint file into only that
piece of the overall graph.

These functions are necessary for use with the RL Tuner class.
"""

import os

from magenta.common import sequence_example_lib
from magenta.models.rl_tuner import rl_tuner_ops
from magenta.models.shared import events_rnn_graph
from magenta.pipelines import melody_pipelines
import note_seq
from note_seq import midi_io
from note_seq import sequences_lib
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim


class NoteRNNLoader(object):
  """Builds graph for a Note RNN and instantiates weights from a checkpoint.

  Loads weights from a previously saved checkpoint file corresponding to a pre-
  trained basic_rnn model. Has functions that allow it to be primed with a MIDI
  melody, and allow it to be called to produce its predictions for the next
  note in a sequence.

  Used as part of the RLTuner class.
  """

  def __init__(self, graph, scope, checkpoint_dir, checkpoint_file=None,
               midi_primer=None, training_file_list=None, hparams=None,
               note_rnn_type='default', checkpoint_scope='rnn_model'):
    """Initialize by building the graph and loading a previous checkpoint.

    Args:
      graph: A tensorflow graph where the MelodyRNN's graph will be added.
      scope: The tensorflow scope where this network will be saved.
      checkpoint_dir: Path to the directory where the checkpoint file is saved.
      checkpoint_file: Path to a checkpoint file to be used if none can be
        found in the checkpoint_dir
      midi_primer: Path to a single midi file that can be used to prime the
        model.
      training_file_list: List of paths to tfrecord files containing melody
        training data.
      hparams: A tf_lib.HParams object. Must match the hparams used to create
        the checkpoint file.
      note_rnn_type: If 'default', will use the basic LSTM described in the
        research paper. If 'basic_rnn', will assume the checkpoint is from a
        Magenta basic_rnn model.
      checkpoint_scope: The scope in lstm which the model was originally defined
        when it was first trained.
    """
    self.graph = graph
    self.session = None
    self.scope = scope
    self.batch_size = 1
    self.midi_primer = midi_primer
    self.checkpoint_scope = checkpoint_scope
    self.note_rnn_type = note_rnn_type
    self.training_file_list = training_file_list
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = checkpoint_file

    if hparams is not None:
      tf.logging.info('Using custom hparams')
      self.hparams = hparams
    else:
      tf.logging.info('Empty hparams string. Using defaults')
      self.hparams = rl_tuner_ops.default_hparams()

    self.build_graph()
    self.state_value = self.get_zero_state()

    if midi_primer is not None:
      self.load_primer()

    self.variable_names = rl_tuner_ops.get_variable_names(self.graph,
                                                          self.scope)

    self.transpose_amount = 0

  def get_zero_state(self):
    """Gets an initial state of zeros of the appropriate size.

    Required size is based on the model's internal RNN cell.

    Returns:
      A matrix of batch_size x cell size zeros.
    """
    return np.zeros((self.batch_size, self.cell.state_size))

  def restore_initialize_prime(self, session):
    """Saves the session, restores variables from checkpoint, primes model.

    Model is primed with its default midi file.

    Args:
      session: A tensorflow session.
    """
    self.session = session
    self.restore_vars_from_checkpoint(self.checkpoint_dir)
    self.prime_model()

  def initialize_and_restore(self, session):
    """Saves the session, restores variables from checkpoint.

    Args:
      session: A tensorflow session.
    """
    self.session = session
    self.restore_vars_from_checkpoint(self.checkpoint_dir)

  def initialize_new(self, session=None):
    """Saves the session, initializes all variables to random values.

    Args:
      session: A tensorflow session.
    """
    with self.graph.as_default():
      if session is None:
        self.session = tf.Session(graph=self.graph)
      else:
        self.session = session
      self.session.run(tf.initialize_all_variables())

  def get_variable_name_dict(self):
    """Constructs a dict mapping the checkpoint variables to those in new graph.

    Returns:
      A dict mapping variable names in the checkpoint to variables in the graph.
    """
    var_dict = dict()
    for var in self.variables():
      inner_name = rl_tuner_ops.get_inner_scope(var.name)
      inner_name = rl_tuner_ops.trim_variable_postfixes(inner_name)
      if '/Adam' in var.name:
        # TODO(lukaszkaiser): investigate the problem here and remove this hack.
        pass
      elif self.note_rnn_type == 'basic_rnn':
        var_dict[inner_name] = var
      else:
        var_dict[self.checkpoint_scope + '/' + inner_name] = var

    return var_dict

  def build_graph(self):
    """Constructs the portion of the graph that belongs to this model."""

    tf.logging.info('Initializing melody RNN graph for scope %s', self.scope)

    with self.graph.as_default():
      with tf.device(lambda op: ''):
        with tf.variable_scope(self.scope):
          # Make an LSTM cell with the number and size of layers specified in
          # hparams.
          if self.note_rnn_type == 'basic_rnn':
            self.cell = events_rnn_graph.make_rnn_cell(
                self.hparams.rnn_layer_sizes)
          else:
            self.cell = rl_tuner_ops.make_rnn_cell(self.hparams.rnn_layer_sizes)
          # Shape of melody_sequence is batch size, melody length, number of
          # output note actions.
          self.melody_sequence = tf.placeholder(tf.float32,
                                                [None, None,
                                                 self.hparams.one_hot_length],
                                                name='melody_sequence')
          self.lengths = tf.placeholder(tf.int32, [None], name='lengths')
          self.initial_state = tf.placeholder(tf.float32,
                                              [None, self.cell.state_size],
                                              name='initial_state')

          if self.training_file_list is not None:
            # Set up a tf queue to read melodies from the training data tfrecord
            (self.train_sequence,
             self.train_labels,
             self.train_lengths) = sequence_example_lib.get_padded_batch(
                 self.training_file_list, self.hparams.batch_size,
                 self.hparams.one_hot_length)

          # Closure function is used so that this part of the graph can be
          # re-run in multiple places, such as __call__.
          def run_network_on_melody(m_seq,
                                    lens,
                                    initial_state,
                                    swap_memory=True,
                                    parallel_iterations=1):
            """Internal function that defines the RNN network structure.

            Args:
              m_seq: A batch of melody sequences of one-hot notes.
              lens: Lengths of the melody_sequences.
              initial_state: Vector representing the initial state of the RNN.
              swap_memory: Uses more memory and is faster.
              parallel_iterations: Argument to tf.nn.dynamic_rnn.
            Returns:
              Output of network (either softmax or logits) and RNN state.
            """
            outputs, final_state = tf.nn.dynamic_rnn(
                self.cell,
                m_seq,
                sequence_length=lens,
                initial_state=initial_state,
                swap_memory=swap_memory,
                parallel_iterations=parallel_iterations)

            outputs_flat = tf.reshape(outputs,
                                      [-1, self.hparams.rnn_layer_sizes[-1]])
            if self.note_rnn_type == 'basic_rnn':
              linear_layer = tf_slim.layers.linear
            else:
              linear_layer = tf_slim.layers.legacy_linear
            logits_flat = linear_layer(
                outputs_flat, self.hparams.one_hot_length)
            return logits_flat, final_state

          (self.logits, self.state_tensor) = run_network_on_melody(
              self.melody_sequence, self.lengths, self.initial_state)
          self.softmax = tf.nn.softmax(self.logits)

          self.run_network_on_melody = run_network_on_melody

        if self.training_file_list is not None:
          # Does not recreate the model architecture but rather uses it to feed
          # data from the training queue through the model.
          with tf.variable_scope(self.scope, reuse=True):
            zero_state = self.cell.zero_state(
                batch_size=self.hparams.batch_size, dtype=tf.float32)

            (self.train_logits, self.train_state) = run_network_on_melody(
                self.train_sequence, self.train_lengths, zero_state)
            self.train_softmax = tf.nn.softmax(self.train_logits)

  def restore_vars_from_checkpoint(self, checkpoint_dir):
    """Loads model weights from a saved checkpoint.

    Args:
      checkpoint_dir: Directory which contains a saved checkpoint of the
        model.
    """
    tf.logging.info('Restoring variables from checkpoint')

    var_dict = self.get_variable_name_dict()
    with self.graph.as_default():
      saver = tf.train.Saver(var_list=var_dict)

    tf.logging.info('Checkpoint dir: %s', checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_file is None:
      tf.logging.warn("Can't find checkpoint file, using %s",
                      self.checkpoint_file)
      checkpoint_file = self.checkpoint_file
    tf.logging.info('Checkpoint file: %s', checkpoint_file)

    saver.restore(self.session, checkpoint_file)

  def load_primer(self):
    """Loads default MIDI primer file.

    Also assigns the steps per bar of this file to be the model's defaults.
    """

    if not os.path.exists(self.midi_primer):
      tf.logging.warn('ERROR! No such primer file exists! %s', self.midi_primer)
      return

    self.primer_sequence = midi_io.midi_file_to_sequence_proto(self.midi_primer)
    quantized_seq = sequences_lib.quantize_note_sequence(
        self.primer_sequence, steps_per_quarter=4)
    extracted_melodies, _ = melody_pipelines.extract_melodies(
        quantized_seq, min_bars=0, min_unique_pitches=1)
    self.primer = extracted_melodies[0]
    self.steps_per_bar = self.primer.steps_per_bar

  def prime_model(self):
    """Primes the model with its default midi primer."""
    with self.graph.as_default():
      tf.logging.debug('Priming the model with MIDI file %s', self.midi_primer)

      # Convert primer Melody to model inputs.
      encoder = note_seq.OneHotEventSequenceEncoderDecoder(
          note_seq.MelodyOneHotEncoding(
              min_note=rl_tuner_ops.MIN_NOTE, max_note=rl_tuner_ops.MAX_NOTE))

      primer_input, _ = encoder.encode(self.primer)

      # Run model over primer sequence.
      primer_input_batch = np.tile([primer_input], (self.batch_size, 1, 1))
      self.state_value, softmax = self.session.run(
          [self.state_tensor, self.softmax],
          feed_dict={self.initial_state: self.state_value,
                     self.melody_sequence: primer_input_batch,
                     self.lengths: np.full(self.batch_size,
                                           len(self.primer),
                                           dtype=int)})
      priming_output = softmax[-1, :]
      self.priming_note = self.get_note_from_softmax(priming_output)

  def get_note_from_softmax(self, softmax):
    """Extracts a one-hot encoding of the most probable note.

    Args:
      softmax: Softmax probabilities over possible next notes.
    Returns:
      One-hot encoding of most probable note.
    """

    note_idx = np.argmax(softmax)
    note_enc = rl_tuner_ops.make_onehot([note_idx], rl_tuner_ops.NUM_CLASSES)
    return np.reshape(note_enc, (rl_tuner_ops.NUM_CLASSES))

  def __call__(self):
    """Allows the network to be called, as in the following code snippet!

        q_network = MelodyRNN(...)
        q_network()

    The q_network() operation can then be placed into a larger graph as a tf op.

    Note that to get actual values from call, must do session.run and feed in
    melody_sequence, lengths, and initial_state in the feed dict.

    Returns:
      Either softmax probabilities over notes, or raw logit scores.
    """
    with self.graph.as_default():
      with tf.variable_scope(self.scope, reuse=True):
        logits, self.state_tensor = self.run_network_on_melody(
            self.melody_sequence, self.lengths, self.initial_state)
        return logits

  def run_training_batch(self):
    """Runs one batch of training data through the model.

    Uses a queue runner to pull one batch of data from the training files
    and run it through the model.

    Returns:
      A batch of softmax probabilities and model state vectors.
    """
    if self.training_file_list is None:
      tf.logging.warn('No training file path was provided, cannot run training'
                      'batch')
      return

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=self.session, coord=coord)

    softmax, state, lengths = self.session.run([self.train_softmax,
                                                self.train_state,
                                                self.train_lengths])

    coord.request_stop()

    return softmax, state, lengths

  def get_next_note_from_note(self, note):
    """Given a note, uses the model to predict the most probable next note.

    Args:
      note: A one-hot encoding of the note.
    Returns:
      Next note in the same format.
    """
    with self.graph.as_default():
      with tf.variable_scope(self.scope, reuse=True):
        singleton_lengths = np.full(self.batch_size, 1, dtype=int)

        input_batch = np.reshape(note,
                                 (self.batch_size, 1, rl_tuner_ops.NUM_CLASSES))

        softmax, self.state_value = self.session.run(
            [self.softmax, self.state_tensor],
            {self.melody_sequence: input_batch,
             self.initial_state: self.state_value,
             self.lengths: singleton_lengths})

        return self.get_note_from_softmax(softmax)

  def variables(self):
    """Gets names of all the variables in the graph belonging to this model.

    Returns:
      List of variable names.
    """
    with self.graph.as_default():
      return [v for v in tf.global_variables() if v.name.startswith(self.scope)]
