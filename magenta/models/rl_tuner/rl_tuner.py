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

"""Defines the main RL Tuner class.

RL Tuner is a Deep Q Network (DQN) with augmented reward to create melodies
by using reinforcement learning to fine-tune a trained Note RNN according
to some music theory rewards.

Also implements two alternatives to Q learning: Psi and G learning. The
algorithm can be switched using the 'algorithm' hyperparameter.

For more information, please consult the README.md file in this directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import os
from os import makedirs
from os.path import exists
import random
import urllib

# internal imports

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from six.moves import range          # pylint: disable=redefined-builtin
from six.moves import reload_module  # pylint: disable=redefined-builtin
from six.moves import urllib         # pylint: disable=redefined-builtin
import tensorflow as tf

from magenta.models.rl_tuner import note_rnn_loader
from magenta.models.rl_tuner import rl_tuner_eval_metrics
from magenta.models.rl_tuner import rl_tuner_ops
from magenta.music import melodies_lib as mlib
from magenta.music import midi_io

# Note values of special actions.
NOTE_OFF = 0
NO_EVENT = 1

# Training data sequences are limited to this length, so the padding queue pads
# to this length.
TRAIN_SEQUENCE_LENGTH = 192


def reload_files():
  """Used to reload the imported dependency files (needed for ipynb notebooks).
  """
  reload_module(note_rnn_loader)
  reload_module(rl_tuner_ops)
  reload_module(rl_tuner_eval_metrics)


class RLTuner(object):
  """Implements a recurrent DQN designed to produce melody sequences."""

  def __init__(self, output_dir,

               # Hyperparameters
               dqn_hparams=None,
               reward_mode='music_theory_all',
               reward_scaler=1.0,
               exploration_mode='egreedy',
               priming_mode='random_note',
               stochastic_observations=False,
               algorithm='q',

               # Trained Note RNN to load and tune
               note_rnn_checkpoint_dir=None,
               note_rnn_checkpoint_file=None,
               note_rnn_type='default',
               note_rnn_hparams=None,

               # Other music related settings.
               num_notes_in_melody=32,
               input_size=rl_tuner_ops.NUM_CLASSES,
               num_actions=rl_tuner_ops.NUM_CLASSES,
               midi_primer=None,

               # Logistics.
               save_name='rl_tuner.ckpt',
               output_every_nth=1000,
               training_file_list=None,
               summary_writer=None,
               initialize_immediately=True):
    """Initializes the MelodyQNetwork class.

    Args:
      output_dir: Where the model will save its compositions (midi files).
      dqn_hparams: A HParams object containing the hyperparameters of
        the DQN algorithm, including minibatch size, exploration probability,
        etc.
      reward_mode: Controls which reward function can be applied. There are
        several, including 'scale', which teaches the model to play a scale,
        and of course 'music_theory_all', which is a music-theory-based reward
        function composed of other functions.
      reward_scaler: Controls the emphasis placed on the music theory rewards.
        This value is the inverse of 'c' in the academic paper.
      exploration_mode: can be 'egreedy' which is an epsilon greedy policy, or
        it can be 'boltzmann', in which the model will sample from its output
        distribution to choose the next action.
      priming_mode: Each time the model begins a new composition, it is primed
        with either a random note ('random_note'), a random MIDI file from the
        training data ('random_midi'), or a particular MIDI file
        ('single_midi').
      stochastic_observations: If False, the note that the model chooses to
        play next (the argmax of its softmax probabilities) deterministically
        becomes the next note it will observe. If True, the next observation
        will be sampled from the model's softmax output.
      algorithm: can be 'default', 'psi', 'g' or 'pure_rl', for different
        learning algorithms
      note_rnn_checkpoint_dir: The directory from which the internal
        NoteRNNLoader will load its checkpointed LSTM.
      note_rnn_checkpoint_file: A checkpoint file to use in case one cannot be
        found in the note_rnn_checkpoint_dir.
      note_rnn_type: If 'default', will use the basic LSTM described in the
        research paper. If 'basic_rnn', will assume the checkpoint is from a
        Magenta basic_rnn model.
      note_rnn_hparams: A HParams object which defines the hyper parameters
        used to train the MelodyRNN model that will be loaded from a checkpoint.
      num_notes_in_melody: The length of a composition of the model
      input_size: the size of the one-hot vector encoding a note that is input
        to the model.
      num_actions: The size of the one-hot vector encoding a note that is
        output by the model.
      midi_primer: A midi file that can be used to prime the model if
        priming_mode is set to 'single_midi'.
      save_name: Name the model will use to save checkpoints.
      output_every_nth: How many training steps before the model will print
        an output saying the cumulative reward, and save a checkpoint.
      training_file_list: A list of paths to tfrecord files containing melody
        training data. This is necessary to use the 'random_midi' priming mode.
      summary_writer: A tf.summary.FileWriter used to log metrics.
      initialize_immediately: if True, the class will instantiate its component
        MelodyRNN networks and build the graph in the constructor.
    """
    # Make graph.
    self.graph = tf.Graph()

    with self.graph.as_default():
      # Memorize arguments.
      self.input_size = input_size
      self.num_actions = num_actions
      self.output_every_nth = output_every_nth
      self.output_dir = output_dir
      self.save_path = os.path.join(output_dir, save_name)
      self.reward_scaler = reward_scaler
      self.reward_mode = reward_mode
      self.exploration_mode = exploration_mode
      self.num_notes_in_melody = num_notes_in_melody
      self.stochastic_observations = stochastic_observations
      self.algorithm = algorithm
      self.priming_mode = priming_mode
      self.midi_primer = midi_primer
      self.training_file_list = training_file_list
      self.note_rnn_checkpoint_dir = note_rnn_checkpoint_dir
      self.note_rnn_checkpoint_file = note_rnn_checkpoint_file
      self.note_rnn_hparams = note_rnn_hparams
      self.note_rnn_type = note_rnn_type

      if priming_mode == 'single_midi' and midi_primer is None:
        tf.logging.fatal('A midi primer file is required when using'
                         'the single_midi priming mode.')

      if note_rnn_checkpoint_dir is None or not note_rnn_checkpoint_dir:
        print('Retrieving checkpoint of Note RNN from Magenta download server.')
        urllib.request.urlretrieve(
            'http://download.magenta.tensorflow.org/models/'
            'rl_tuner_note_rnn.ckpt', 'note_rnn.ckpt')
        self.note_rnn_checkpoint_dir = os.getcwd()
        self.note_rnn_checkpoint_file = os.path.join(os.getcwd(),
                                                     'note_rnn.ckpt')

      if self.note_rnn_hparams is None:
        if self.note_rnn_type == 'basic_rnn':
          self.note_rnn_hparams = rl_tuner_ops.basic_rnn_hparams()
        else:
          self.note_rnn_hparams = rl_tuner_ops.default_hparams()

      if self.algorithm == 'g' or self.algorithm == 'pure_rl':
        self.reward_mode = 'music_theory_only'

      if dqn_hparams is None:
        self.dqn_hparams = rl_tuner_ops.default_dqn_hparams()
      else:
        self.dqn_hparams = dqn_hparams
      self.discount_rate = tf.constant(self.dqn_hparams.discount_rate)
      self.target_network_update_rate = tf.constant(
          self.dqn_hparams.target_network_update_rate)

      self.optimizer = tf.train.AdamOptimizer()

      # DQN state.
      self.actions_executed_so_far = 0
      self.experience = deque(maxlen=self.dqn_hparams.max_experience)
      self.iteration = 0
      self.summary_writer = summary_writer
      self.num_times_store_called = 0
      self.num_times_train_called = 0

    # Stored reward metrics.
    self.reward_last_n = 0
    self.rewards_batched = []
    self.music_theory_reward_last_n = 0
    self.music_theory_rewards_batched = []
    self.note_rnn_reward_last_n = 0
    self.note_rnn_rewards_batched = []
    self.eval_avg_reward = []
    self.eval_avg_music_theory_reward = []
    self.eval_avg_note_rnn_reward = []
    self.target_val_list = []

    # Variables to keep track of characteristics of the current composition
    # TODO(natashajaques): Implement composition as a class to obtain data
    # encapsulation so that you can't accidentally change the leap direction.
    self.beat = 0
    self.composition = []
    self.composition_direction = 0
    self.leapt_from = None  # stores the note at which composition leapt
    self.steps_since_last_leap = 0

    if not exists(self.output_dir):
      makedirs(self.output_dir)

    if initialize_immediately:
      self.initialize_internal_models_graph_session()

  def initialize_internal_models_graph_session(self,
                                               restore_from_checkpoint=True):
    """Initializes internal RNN models, builds the graph, starts the session.

    Adds the graphs of the internal RNN models to this graph, adds the DQN ops
    to the graph, and starts a new Saver and session. By having a separate
    function for this rather than doing it in the constructor, it allows a model
    inheriting from this class to define its q_network differently.

    Args:
      restore_from_checkpoint: If True, the weights for the 'q_network',
        'target_q_network', and 'reward_rnn' will be loaded from a checkpoint.
        If false, these models will be initialized with random weights. Useful
        for checking what pure RL (with no influence from training data) sounds
        like.
    """
    with self.graph.as_default():
      # Add internal networks to the graph.
      tf.logging.info('Initializing q network')
      self.q_network = note_rnn_loader.NoteRNNLoader(
          self.graph, 'q_network',
          self.note_rnn_checkpoint_dir,
          midi_primer=self.midi_primer,
          training_file_list=self.training_file_list,
          checkpoint_file=self.note_rnn_checkpoint_file,
          hparams=self.note_rnn_hparams,
          note_rnn_type=self.note_rnn_type)

      tf.logging.info('Initializing target q network')
      self.target_q_network = note_rnn_loader.NoteRNNLoader(
          self.graph,
          'target_q_network',
          self.note_rnn_checkpoint_dir,
          midi_primer=self.midi_primer,
          training_file_list=self.training_file_list,
          checkpoint_file=self.note_rnn_checkpoint_file,
          hparams=self.note_rnn_hparams,
          note_rnn_type=self.note_rnn_type)

      tf.logging.info('Initializing reward network')
      self.reward_rnn = note_rnn_loader.NoteRNNLoader(
          self.graph, 'reward_rnn',
          self.note_rnn_checkpoint_dir,
          midi_primer=self.midi_primer,
          training_file_list=self.training_file_list,
          checkpoint_file=self.note_rnn_checkpoint_file,
          hparams=self.note_rnn_hparams,
          note_rnn_type=self.note_rnn_type)

      tf.logging.info('Q network cell: %s', self.q_network.cell)

      # Add rest of variables to graph.
      tf.logging.info('Adding RL graph variables')
      self.build_graph()

      # Prepare saver and session.
      self.saver = tf.train.Saver()
      self.session = tf.Session(graph=self.graph)
      self.session.run(tf.global_variables_initializer())

      # Initialize internal networks.
      if restore_from_checkpoint:
        self.q_network.initialize_and_restore(self.session)
        self.target_q_network.initialize_and_restore(self.session)
        self.reward_rnn.initialize_and_restore(self.session)

        # Double check that the model was initialized from checkpoint properly.
        reward_vars = self.reward_rnn.variables()
        q_vars = self.q_network.variables()

        reward1 = self.session.run(reward_vars[0])
        q1 = self.session.run(q_vars[0])

        if np.sum((q1 - reward1)**2) == 0.0:
          # TODO(natashamjaques): Remove print statement once tf.logging outputs
          # to Jupyter notebooks (once the following issue is resolved:
          # https://github.com/tensorflow/tensorflow/issues/3047)
          print('\nSuccessfully initialized internal nets from checkpoint!')
          tf.logging.info('\nSuccessfully initialized internal nets from '
                          'checkpoint!')
        else:
          tf.logging.fatal('Error! The model was not initialized from '
                           'checkpoint properly')
      else:
        self.q_network.initialize_new(self.session)
        self.target_q_network.initialize_new(self.session)
        self.reward_rnn.initialize_new(self.session)

    if self.priming_mode == 'random_midi':
      tf.logging.info('Getting priming melodies')
      self.get_priming_melodies()

  def get_priming_melodies(self):
    """Runs a batch of training data through MelodyRNN model.

    If the priming mode is 'random_midi', priming the q-network requires a
    random training melody. Therefore this function runs a batch of data from
    the training directory through the internal model, and the resulting
    internal states of the LSTM are stored in a list. The next note in each
    training melody is also stored in a corresponding list called
    'priming_notes'. Therefore, to prime the model with a random melody, it is
    only necessary to select a random index from 0 to batch_size-1 and use the
    hidden states and note at that index as input to the model.
    """
    (next_note_softmax,
     self.priming_states, lengths) = self.q_network.run_training_batch()

    # Get the next note that was predicted for each priming melody to be used
    # in priming.
    self.priming_notes = [0] * len(lengths)
    for i in range(len(lengths)):
      # Each melody has TRAIN_SEQUENCE_LENGTH outputs, but the last note is
      # actually stored at lengths[i]. The rest is padding.
      start_i = i * TRAIN_SEQUENCE_LENGTH
      end_i = start_i + lengths[i] - 1
      end_softmax = next_note_softmax[end_i, :]
      self.priming_notes[i] = np.argmax(end_softmax)

    tf.logging.info('Stored priming notes: %s', self.priming_notes)

  def prime_internal_model(self, model):
    """Prime an internal model such as the q_network based on priming mode.

    Args:
      model: The internal model that should be primed.

    Returns:
      The first observation to feed into the model.
    """
    model.state_value = model.get_zero_state()

    if self.priming_mode == 'random_midi':
      priming_idx = np.random.randint(0, len(self.priming_states))
      model.state_value = np.reshape(
          self.priming_states[priming_idx, :],
          (1, model.cell.state_size))
      priming_note = self.priming_notes[priming_idx]
      next_obs = np.array(
          rl_tuner_ops.make_onehot([priming_note], self.num_actions)).flatten()
      tf.logging.debug(
          'Feeding priming state for midi file %s and corresponding note %s',
          priming_idx, priming_note)
    elif self.priming_mode == 'single_midi':
      model.prime_model()
      next_obs = model.priming_note
    elif self.priming_mode == 'random_note':
      next_obs = self.get_random_note()
    else:
      tf.logging.warn('Error! Invalid priming mode. Priming with random note')
      next_obs = self.get_random_note()

    return next_obs

  def get_random_note(self):
    """Samle a note uniformly at random.

    Returns:
      random note
    """
    note_idx = np.random.randint(0, self.num_actions - 1)
    return np.array(rl_tuner_ops.make_onehot([note_idx],
                                             self.num_actions)).flatten()

  def reset_composition(self):
    """Starts the models internal composition over at beat 0, with no notes.

    Also resets statistics about whether the composition is in the middle of a
    melodic leap.
    """
    self.beat = 0
    self.composition = []
    self.composition_direction = 0
    self.leapt_from = None
    self.steps_since_last_leap = 0

  def build_graph(self):
    """Builds the reinforcement learning tensorflow graph."""

    tf.logging.info('Adding reward computation portion of the graph')
    with tf.name_scope('reward_computation'):
      self.reward_scores = tf.identity(self.reward_rnn(), name='reward_scores')

    tf.logging.info('Adding taking action portion of graph')
    with tf.name_scope('taking_action'):
      # Output of the q network gives the value of taking each action (playing
      # each note).
      self.action_scores = tf.identity(self.q_network(), name='action_scores')
      tf.summary.histogram(
          'action_scores', self.action_scores)

      # The action values for the G algorithm are computed differently.
      if self.algorithm == 'g':
        self.g_action_scores = self.action_scores + self.reward_scores

        # Compute predicted action, which is the argmax of the action scores.
        self.action_softmax = tf.nn.softmax(self.g_action_scores,
                                            name='action_softmax')
        self.predicted_actions = tf.one_hot(tf.argmax(self.g_action_scores,
                                                      dimension=1,
                                                      name='predicted_actions'),
                                            self.num_actions)
      else:
        # Compute predicted action, which is the argmax of the action scores.
        self.action_softmax = tf.nn.softmax(self.action_scores,
                                            name='action_softmax')
        self.predicted_actions = tf.one_hot(tf.argmax(self.action_scores,
                                                      dimension=1,
                                                      name='predicted_actions'),
                                            self.num_actions)

    tf.logging.info('Add estimating future rewards portion of graph')
    with tf.name_scope('estimating_future_rewards'):
      # The target q network is used to estimate the value of the best action at
      # the state resulting from the current action.
      self.next_action_scores = tf.stop_gradient(self.target_q_network())
      tf.summary.histogram(
          'target_action_scores', self.next_action_scores)

      # Rewards are observed from the environment and are fed in later.
      self.rewards = tf.placeholder(tf.float32, (None,), name='rewards')

      # Each algorithm is attempting to model future rewards with a different
      # function.
      if self.algorithm == 'psi':
        self.target_vals = tf.reduce_logsumexp(self.next_action_scores,
                                               reduction_indices=[1,])
      elif self.algorithm == 'g':
        self.g_normalizer = tf.reduce_logsumexp(self.reward_scores,
                                                reduction_indices=[1,])
        self.g_normalizer = tf.reshape(self.g_normalizer, [-1, 1])
        self.g_normalizer = tf.tile(self.g_normalizer, [1, self.num_actions])
        self.g_action_scores = tf.subtract(
            (self.next_action_scores + self.reward_scores), self.g_normalizer)
        self.target_vals = tf.reduce_logsumexp(self.g_action_scores,
                                               reduction_indices=[1,])
      else:
        # Use default based on Q learning.
        self.target_vals = tf.reduce_max(self.next_action_scores,
                                         reduction_indices=[1,])

      # Total rewards are the observed rewards plus discounted estimated future
      # rewards.
      self.future_rewards = self.rewards + self.discount_rate * self.target_vals

    tf.logging.info('Adding q value prediction portion of graph')
    with tf.name_scope('q_value_prediction'):
      # Action mask will be a one-hot encoding of the action the network
      # actually took.
      self.action_mask = tf.placeholder(tf.float32, (None, self.num_actions),
                                        name='action_mask')
      self.masked_action_scores = tf.reduce_sum(self.action_scores *
                                                self.action_mask,
                                                reduction_indices=[1,])

      temp_diff = self.masked_action_scores - self.future_rewards

      # Prediction error is the mean squared error between the reward the
      # network actually received for a given action, and what it expected to
      # receive.
      self.prediction_error = tf.reduce_mean(tf.square(temp_diff))

      # Compute gradients.
      self.params = tf.trainable_variables()
      self.gradients = self.optimizer.compute_gradients(self.prediction_error)

      # Clip gradients.
      for i, (grad, var) in enumerate(self.gradients):
        if grad is not None:
          self.gradients[i] = (tf.clip_by_norm(grad, 5), var)

      for grad, var in self.gradients:
        tf.summary.histogram(var.name, var)
        if grad is not None:
          tf.summary.histogram(var.name + '/gradients', grad)

      # Backprop.
      self.train_op = self.optimizer.apply_gradients(self.gradients)

    tf.logging.info('Adding target network update portion of graph')
    with tf.name_scope('target_network_update'):
      # Updates the target_q_network to be similar to the q_network based on
      # the target_network_update_rate.
      self.target_network_update = []
      for v_source, v_target in zip(self.q_network.variables(),
                                    self.target_q_network.variables()):
        # Equivalent to target = (1-alpha) * target + alpha * source
        update_op = v_target.assign_sub(self.target_network_update_rate *
                                        (v_target - v_source))
        self.target_network_update.append(update_op)
      self.target_network_update = tf.group(*self.target_network_update)

    tf.summary.scalar(
        'prediction_error', self.prediction_error)

    self.summarize = tf.summary.merge_all()
    self.no_op1 = tf.no_op()

  def train(self, num_steps=10000, exploration_period=5000, enable_random=True):
    """Main training function that allows model to act, collects reward, trains.

    Iterates a number of times, getting the model to act each time, saving the
    experience, and performing backprop.

    Args:
      num_steps: The number of training steps to execute.
      exploration_period: The number of steps over which the probability of
        exploring (taking a random action) is annealed from 1.0 to the model's
        random_action_probability.
      enable_random: If False, the model will not be able to act randomly /
        explore.
    """
    tf.logging.info('Evaluating initial model...')
    self.evaluate_model()

    self.actions_executed_so_far = 0

    if self.stochastic_observations:
      tf.logging.info('Using stochastic environment')

    sample_next_obs = False
    if self.exploration_mode == 'boltzmann' or self.stochastic_observations:
      sample_next_obs = True

    self.reset_composition()
    last_observation = self.prime_internal_models()

    for i in range(num_steps):
      # Experiencing observation, state, action, reward, new observation,
      # new state tuples, and storing them.
      state = np.array(self.q_network.state_value).flatten()

      action, new_observation, reward_scores = self.action(
          last_observation, exploration_period, enable_random=enable_random,
          sample_next_obs=sample_next_obs)

      new_state = np.array(self.q_network.state_value).flatten()
      new_reward_state = np.array(self.reward_rnn.state_value).flatten()

      reward = self.collect_reward(last_observation, new_observation,
                                   reward_scores)

      self.store(last_observation, state, action, reward, new_observation,
                 new_state, new_reward_state)

      # Used to keep track of how the reward is changing over time.
      self.reward_last_n += reward

      # Used to keep track of the current musical composition and beat for
      # the reward functions.
      self.composition.append(np.argmax(new_observation))
      self.beat += 1

      if i > 0 and i % self.output_every_nth == 0:
        tf.logging.info('Evaluating model...')
        self.evaluate_model()
        self.save_model(self.algorithm)

        if self.algorithm == 'g':
          self.rewards_batched.append(
              self.music_theory_reward_last_n + self.note_rnn_reward_last_n)
        else:
          self.rewards_batched.append(self.reward_last_n)
        self.music_theory_rewards_batched.append(
            self.music_theory_reward_last_n)
        self.note_rnn_rewards_batched.append(self.note_rnn_reward_last_n)

        # Save a checkpoint.
        save_step = len(self.rewards_batched)*self.output_every_nth
        self.saver.save(self.session, self.save_path, global_step=save_step)

        r = self.reward_last_n
        tf.logging.info('Training iteration %s', i)
        tf.logging.info('\tReward for last %s steps: %s',
                        self.output_every_nth, r)
        tf.logging.info('\t\tMusic theory reward: %s',
                        self.music_theory_reward_last_n)
        tf.logging.info('\t\tNote RNN reward: %s', self.note_rnn_reward_last_n)

        # TODO(natashamjaques): Remove print statement once tf.logging outputs
        # to Jupyter notebooks (once the following issue is resolved:
        # https://github.com/tensorflow/tensorflow/issues/3047)
        print('Training iteration', i)
        print('\tReward for last', self.output_every_nth, 'steps:', r)
        print('\t\tMusic theory reward:', self.music_theory_reward_last_n)
        print('\t\tNote RNN reward:', self.note_rnn_reward_last_n)

        if self.exploration_mode == 'egreedy':
          exploration_p = rl_tuner_ops.linear_annealing(
              self.actions_executed_so_far, exploration_period, 1.0,
              self.dqn_hparams.random_action_probability)
          tf.logging.info('\tExploration probability is %s', exploration_p)

        self.reward_last_n = 0
        self.music_theory_reward_last_n = 0
        self.note_rnn_reward_last_n = 0

      # Backprop.
      self.training_step()

      # Update current state as last state.
      last_observation = new_observation

      # Reset the state after each composition is complete.
      if self.beat % self.num_notes_in_melody == 0:
        tf.logging.debug('\nResetting composition!\n')
        self.reset_composition()
        last_observation = self.prime_internal_models()

  def action(self, observation, exploration_period=0, enable_random=True,
             sample_next_obs=False):
    """Given an observation, runs the q_network to choose the current action.

    Does not backprop.

    Args:
      observation: A one-hot encoding of a single observation (note).
      exploration_period: The total length of the period the network will
        spend exploring, as set in the train function.
      enable_random: If False, the network cannot act randomly.
      sample_next_obs: If True, the next observation will be sampled from
        the softmax probabilities produced by the model, and passed back
        along with the action. If False, only the action is passed back.

    Returns:
      The action chosen, the reward_scores returned by the reward_rnn, and the
      next observation. If sample_next_obs is False, the next observation is
      equal to the action.
    """
    assert len(observation.shape) == 1, 'Single observation only'

    self.actions_executed_so_far += 1

    if self.exploration_mode == 'egreedy':
      # Compute the exploration probability.
      exploration_p = rl_tuner_ops.linear_annealing(
          self.actions_executed_so_far, exploration_period, 1.0,
          self.dqn_hparams.random_action_probability)
    elif self.exploration_mode == 'boltzmann':
      enable_random = False
      sample_next_obs = True

    # Run the observation through the q_network.
    input_batch = np.reshape(observation,
                             (self.q_network.batch_size, 1, self.input_size))
    lengths = np.full(self.q_network.batch_size, 1, dtype=int)

    (action, action_softmax, self.q_network.state_value,
     reward_scores, self.reward_rnn.state_value) = self.session.run(
         [self.predicted_actions, self.action_softmax,
          self.q_network.state_tensor, self.reward_scores,
          self.reward_rnn.state_tensor],
         {self.q_network.melody_sequence: input_batch,
          self.q_network.initial_state: self.q_network.state_value,
          self.q_network.lengths: lengths,
          self.reward_rnn.melody_sequence: input_batch,
          self.reward_rnn.initial_state: self.reward_rnn.state_value,
          self.reward_rnn.lengths: lengths})

    reward_scores = np.reshape(reward_scores, (self.num_actions))
    action_softmax = np.reshape(action_softmax, (self.num_actions))
    action = np.reshape(action, (self.num_actions))

    if enable_random and random.random() < exploration_p:
      note = self.get_random_note()
      return note, note, reward_scores
    else:
      if not sample_next_obs:
        return action, action, reward_scores
      else:
        obs_note = rl_tuner_ops.sample_softmax(action_softmax)
        next_obs = np.array(
            rl_tuner_ops.make_onehot([obs_note], self.num_actions)).flatten()
        return action, next_obs, reward_scores

  def store(self, observation, state, action, reward, newobservation, newstate,
            new_reward_state):
    """Stores an experience in the model's experience replay buffer.

    One experience consists of an initial observation and internal LSTM state,
    which led to the execution of an action, the receipt of a reward, and
    finally a new observation and a new LSTM internal state.

    Args:
      observation: A one hot encoding of an observed note.
      state: The internal state of the q_network MelodyRNN LSTM model.
      action: A one hot encoding of action taken by network.
      reward: Reward received for taking the action.
      newobservation: The next observation that resulted from the action.
        Unless stochastic_observations is True, the action and new
        observation will be the same.
      newstate: The internal state of the q_network MelodyRNN that is
        observed after taking the action.
      new_reward_state: The internal state of the reward_rnn network that is
        observed after taking the action
    """
    if self.num_times_store_called % self.dqn_hparams.store_every_nth == 0:
      self.experience.append((observation, state, action, reward,
                              newobservation, newstate, new_reward_state))
    self.num_times_store_called += 1

  def training_step(self):
    """Backpropagate prediction error from a randomly sampled experience batch.

    A minibatch of experiences is randomly sampled from the model's experience
    replay buffer and used to update the weights of the q_network and
    target_q_network.
    """
    if self.num_times_train_called % self.dqn_hparams.train_every_nth == 0:
      if len(self.experience) < self.dqn_hparams.minibatch_size:
        return

      # Sample experience.
      samples = random.sample(range(len(self.experience)),
                              self.dqn_hparams.minibatch_size)
      samples = [self.experience[i] for i in samples]

      # Batch states.
      states = np.empty((len(samples), self.q_network.cell.state_size))
      new_states = np.empty((len(samples),
                             self.target_q_network.cell.state_size))
      reward_new_states = np.empty((len(samples),
                                    self.reward_rnn.cell.state_size))
      observations = np.empty((len(samples), self.input_size))
      new_observations = np.empty((len(samples), self.input_size))
      action_mask = np.zeros((len(samples), self.num_actions))
      rewards = np.empty((len(samples),))
      lengths = np.full(len(samples), 1, dtype=int)

      for i, (o, s, a, r, new_o, new_s, reward_s) in enumerate(samples):
        observations[i, :] = o
        new_observations[i, :] = new_o
        states[i, :] = s
        new_states[i, :] = new_s
        action_mask[i, :] = a
        rewards[i] = r
        reward_new_states[i, :] = reward_s

      observations = np.reshape(observations,
                                (len(samples), 1, self.input_size))
      new_observations = np.reshape(new_observations,
                                    (len(samples), 1, self.input_size))

      calc_summaries = self.iteration % 100 == 0
      calc_summaries = calc_summaries and self.summary_writer is not None

      if self.algorithm == 'g':
        _, _, target_vals, summary_str = self.session.run([
            self.prediction_error,
            self.train_op,
            self.target_vals,
            self.summarize if calc_summaries else self.no_op1,
        ], {
            self.reward_rnn.melody_sequence: new_observations,
            self.reward_rnn.initial_state: reward_new_states,
            self.reward_rnn.lengths: lengths,
            self.q_network.melody_sequence: observations,
            self.q_network.initial_state: states,
            self.q_network.lengths: lengths,
            self.target_q_network.melody_sequence: new_observations,
            self.target_q_network.initial_state: new_states,
            self.target_q_network.lengths: lengths,
            self.action_mask: action_mask,
            self.rewards: rewards,
        })
      else:
        _, _, target_vals, summary_str = self.session.run([
            self.prediction_error,
            self.train_op,
            self.target_vals,
            self.summarize if calc_summaries else self.no_op1,
        ], {
            self.q_network.melody_sequence: observations,
            self.q_network.initial_state: states,
            self.q_network.lengths: lengths,
            self.target_q_network.melody_sequence: new_observations,
            self.target_q_network.initial_state: new_states,
            self.target_q_network.lengths: lengths,
            self.action_mask: action_mask,
            self.rewards: rewards,
        })

      total_logs = (self.iteration * self.dqn_hparams.train_every_nth)
      if total_logs % self.output_every_nth == 0:
        self.target_val_list.append(np.mean(target_vals))

      self.session.run(self.target_network_update)

      if calc_summaries:
        self.summary_writer.add_summary(summary_str, self.iteration)

      self.iteration += 1

    self.num_times_train_called += 1

  def evaluate_model(self, num_trials=100, sample_next_obs=True):
    """Used to evaluate the rewards the model receives without exploring.

    Generates num_trials compositions and computes the note_rnn and music
    theory rewards. Uses no exploration so rewards directly relate to the
    model's policy. Stores result in internal variables.

    Args:
      num_trials: The number of compositions to use for evaluation.
      sample_next_obs: If True, the next note the model plays will be
        sampled from its output distribution. If False, the model will
        deterministically choose the note with maximum value.
    """

    note_rnn_rewards = [0] * num_trials
    music_theory_rewards = [0] * num_trials
    total_rewards = [0] * num_trials

    for t in range(num_trials):

      last_observation = self.prime_internal_models()
      self.reset_composition()

      for _ in range(self.num_notes_in_melody):
        _, new_observation, reward_scores = self.action(
            last_observation,
            0,
            enable_random=False,
            sample_next_obs=sample_next_obs)

        note_rnn_reward = self.reward_from_reward_rnn_scores(new_observation,
                                                             reward_scores)
        music_theory_reward = self.reward_music_theory(new_observation)
        adjusted_mt_reward = self.reward_scaler * music_theory_reward
        total_reward = note_rnn_reward + adjusted_mt_reward

        note_rnn_rewards[t] = note_rnn_reward
        music_theory_rewards[t] = music_theory_reward * self.reward_scaler
        total_rewards[t] = total_reward

        self.composition.append(np.argmax(new_observation))
        self.beat += 1
        last_observation = new_observation

    self.eval_avg_reward.append(np.mean(total_rewards))
    self.eval_avg_note_rnn_reward.append(np.mean(note_rnn_rewards))
    self.eval_avg_music_theory_reward.append(np.mean(music_theory_rewards))

  def collect_reward(self, obs, action, reward_scores):
    """Calls whatever reward function is indicated in the reward_mode field.

    New reward functions can be written and called from here. Note that the
    reward functions can make use of the musical composition that has been
    played so far, which is stored in self.composition. Some reward functions
    are made up of many smaller functions, such as those related to music
    theory.

    Args:
      obs: A one-hot encoding of the observed note.
      action: A one-hot encoding of the chosen action.
      reward_scores: The value for each note output by the reward_rnn.
    Returns:
      Float reward value.
    """
    # Gets and saves log p(a|s) as output by reward_rnn.
    note_rnn_reward = self.reward_from_reward_rnn_scores(action, reward_scores)
    self.note_rnn_reward_last_n += note_rnn_reward

    if self.reward_mode == 'scale':
      # Makes the model play a scale (defaults to c major).
      reward = self.reward_scale(obs, action)
    elif self.reward_mode == 'key':
      # Makes the model play within a key.
      reward = self.reward_key_distribute_prob(action)
    elif self.reward_mode == 'key_and_tonic':
      # Makes the model play within a key, while starting and ending on the
      # tonic note.
      reward = self.reward_key(action)
      reward += self.reward_tonic(action)
    elif self.reward_mode == 'non_repeating':
      # The model can play any composition it wants, but receives a large
      # negative reward for playing the same note repeatedly.
      reward = self.reward_non_repeating(action)
    elif self.reward_mode == 'music_theory_random':
      # The model receives reward for playing in key, playing tonic notes,
      # and not playing repeated notes. However the rewards it receives are
      # uniformly distributed over all notes that do not violate these rules.
      reward = self.reward_key(action)
      reward += self.reward_tonic(action)
      reward += self.reward_penalize_repeating(action)
    elif self.reward_mode == 'music_theory_basic':
      # As above, the model receives reward for playing in key, tonic notes
      # at the appropriate times, and not playing repeated notes. However, the
      # rewards it receives are based on the note probabilities learned from
      # data in the original model.
      reward = self.reward_key(action)
      reward += self.reward_tonic(action)
      reward += self.reward_penalize_repeating(action)

      return reward * self.reward_scaler + note_rnn_reward
    elif self.reward_mode == 'music_theory_basic_plus_variety':
      # Uses the same reward function as above, but adds a penalty for
      # compositions with a high autocorrelation (aka those that don't have
      # sufficient variety).
      reward = self.reward_key(action)
      reward += self.reward_tonic(action)
      reward += self.reward_penalize_repeating(action)
      reward += self.reward_penalize_autocorrelation(action)

      return reward * self.reward_scaler + note_rnn_reward
    elif self.reward_mode == 'preferred_intervals':
      reward = self.reward_preferred_intervals(action)
    elif self.reward_mode == 'music_theory_all':
      tf.logging.debug('Note RNN reward: %s', note_rnn_reward)

      reward = self.reward_music_theory(action)

      tf.logging.debug('Total music theory reward: %s',
                       self.reward_scaler * reward)
      tf.logging.debug('Total note rnn reward: %s', note_rnn_reward)

      self.music_theory_reward_last_n += reward * self.reward_scaler
      return reward * self.reward_scaler + note_rnn_reward
    elif self.reward_mode == 'music_theory_only':
      reward = self.reward_music_theory(action)
    else:
      tf.logging.fatal('ERROR! Not a valid reward mode. Cannot compute reward')

    self.music_theory_reward_last_n += reward * self.reward_scaler
    return reward * self.reward_scaler

  def reward_from_reward_rnn_scores(self, action, reward_scores):
    """Rewards based on probabilities learned from data by trained RNN.

    Computes the reward_network's learned softmax probabilities. When used as
    rewards, allows the model to maintain information it learned from data.

    Args:
      action: A one-hot encoding of the chosen action.
      reward_scores: The value for each note output by the reward_rnn.
    Returns:
      Float reward value.
    """
    action_note = np.argmax(action)
    normalization_constant = logsumexp(reward_scores)
    return reward_scores[action_note] - normalization_constant

  def get_reward_rnn_scores(self, observation, state):
    """Get note scores from the reward_rnn to use as a reward based on data.

    Runs the reward_rnn on an observation and initial state. Useful for
    maintaining the probabilities of the original LSTM model while training with
    reinforcement learning.

    Args:
      observation: One-hot encoding of the observed note.
      state: Vector representing the internal state of the target_q_network
        LSTM.

    Returns:
      Action scores produced by reward_rnn.
    """
    state = np.atleast_2d(state)

    input_batch = np.reshape(observation, (self.reward_rnn.batch_size, 1,
                                           self.num_actions))
    lengths = np.full(self.reward_rnn.batch_size, 1, dtype=int)

    rewards, = self.session.run(
        self.reward_scores,
        {self.reward_rnn.melody_sequence: input_batch,
         self.reward_rnn.initial_state: state,
         self.reward_rnn.lengths: lengths})
    return rewards

  def reward_music_theory(self, action):
    """Computes cumulative reward for all music theory functions.

    Args:
      action: A one-hot encoding of the chosen action.
    Returns:
      Float reward value.
    """
    reward = self.reward_key(action)
    tf.logging.debug('Key: %s', reward)
    prev_reward = reward

    reward += self.reward_tonic(action)
    if reward != prev_reward:
      tf.logging.debug('Tonic: %s', reward)
    prev_reward = reward

    reward += self.reward_penalize_repeating(action)
    if reward != prev_reward:
      tf.logging.debug('Penalize repeating: %s', reward)
    prev_reward = reward

    reward += self.reward_penalize_autocorrelation(action)
    if reward != prev_reward:
      tf.logging.debug('Penalize autocorr: %s', reward)
    prev_reward = reward

    reward += self.reward_motif(action)
    if reward != prev_reward:
      tf.logging.debug('Reward motif: %s', reward)
    prev_reward = reward

    reward += self.reward_repeated_motif(action)
    if reward != prev_reward:
      tf.logging.debug('Reward repeated motif: %s', reward)
    prev_reward = reward

    # New rewards based on Gauldin's book, "A Practical Approach to Eighteenth
    # Century Counterpoint"
    reward += self.reward_preferred_intervals(action)
    if reward != prev_reward:
      tf.logging.debug('Reward preferred_intervals: %s', reward)
    prev_reward = reward

    reward += self.reward_leap_up_back(action)
    if reward != prev_reward:
      tf.logging.debug('Reward leap up back: %s', reward)
    prev_reward = reward

    reward += self.reward_high_low_unique(action)
    if reward != prev_reward:
      tf.logging.debug('Reward high low unique: %s', reward)

    return reward

  def random_reward_shift_to_mean(self, reward):
    """Modifies reward by a small random values s to pull it towards the mean.

    If reward is above the mean, s is subtracted; if reward is below the mean,
    s is added. The random value is in the range 0-0.2. This function is helpful
    to ensure that the model does not become too certain about playing a
    particular note.

    Args:
      reward: A reward value that has already been computed by another reward
        function.
    Returns:
      Original float reward value modified by scaler.
    """
    s = np.random.randint(0, 2) * .1
    if reward > .5:
      reward -= s
    else:
      reward += s
    return reward

  def reward_scale(self, obs, action, scale=None):
    """Reward function that trains the model to play a scale.

    Gives rewards for increasing notes, notes within the desired scale, and two
    consecutive notes from the scale.

    Args:
      obs: A one-hot encoding of the observed note.
      action: A one-hot encoding of the chosen action.
      scale: The scale the model should learn. Defaults to C Major if not
        provided.
    Returns:
      Float reward value.
    """

    if scale is None:
      scale = rl_tuner_ops.C_MAJOR_SCALE

    obs = np.argmax(obs)
    action = np.argmax(action)
    reward = 0
    if action == 1:
      reward += .1
    if action > obs and action < obs + 3:
      reward += .05

    if action in scale:
      reward += .01
      if obs in scale:
        action_pos = scale.index(action)
        obs_pos = scale.index(obs)
        if obs_pos == len(scale) - 1 and action_pos == 0:
          reward += .8
        elif action_pos == obs_pos + 1:
          reward += .8

    return reward

  def reward_key_distribute_prob(self, action, key=None):
    """Reward function that rewards the model for playing within a given key.

    Any note within the key is given equal reward, which can cause the model to
    learn random sounding compositions.

    Args:
      action: One-hot encoding of the chosen action.
      key: The numeric values of notes belonging to this key. Defaults to C
        Major if not provided.
    Returns:
      Float reward value.
    """
    if key is None:
      key = rl_tuner_ops.C_MAJOR_KEY

    reward = 0

    action_note = np.argmax(action)
    if action_note in key:
      num_notes_in_key = len(key)
      extra_prob = 1.0 / num_notes_in_key

      reward = extra_prob

    return reward

  def reward_key(self, action, penalty_amount=-1.0, key=None):
    """Applies a penalty for playing notes not in a specific key.

    Args:
      action: One-hot encoding of the chosen action.
      penalty_amount: The amount the model will be penalized if it plays
        a note outside the key.
      key: The numeric values of notes belonging to this key. Defaults to
        C-major if not provided.
    Returns:
      Float reward value.
    """
    if key is None:
      key = rl_tuner_ops.C_MAJOR_KEY

    reward = 0

    action_note = np.argmax(action)
    if action_note not in key:
      reward = penalty_amount

    return reward

  def reward_tonic(self, action, tonic_note=rl_tuner_ops.C_MAJOR_TONIC,
                   reward_amount=3.0):
    """Rewards for playing the tonic note at the right times.

    Rewards for playing the tonic as the first note of the first bar, and the
    first note of the final bar.

    Args:
      action: One-hot encoding of the chosen action.
      tonic_note: The tonic/1st note of the desired key.
      reward_amount: The amount the model will be awarded if it plays the
        tonic note at the right time.
    Returns:
      Float reward value.
    """
    action_note = np.argmax(action)
    first_note_of_final_bar = self.num_notes_in_melody - 4

    if self.beat == 0 or self.beat == first_note_of_final_bar:
      if action_note == tonic_note:
        return reward_amount
    elif self.beat == first_note_of_final_bar + 1:
      if action_note == NO_EVENT:
        return reward_amount
    elif self.beat > first_note_of_final_bar + 1:
      if action_note == NO_EVENT or action_note == NOTE_OFF:
        return reward_amount
    return 0.0

  def reward_non_repeating(self, action):
    """Rewards the model for not playing the same note over and over.

    Penalizes the model for playing the same note repeatedly, although more
    repeititions are allowed if it occasionally holds the note or rests in
    between. Reward is uniform when there is no penalty.

    Args:
      action: One-hot encoding of the chosen action.
    Returns:
      Float reward value.
    """
    penalty = self.reward_penalize_repeating(action)
    if penalty >= 0:
      return .1

  def detect_repeating_notes(self, action_note):
    """Detects whether the note played is repeating previous notes excessively.

    Args:
      action_note: An integer representing the note just played.
    Returns:
      True if the note just played is excessively repeated, False otherwise.
    """
    num_repeated = 0
    contains_held_notes = False
    contains_breaks = False

    # Note that the current action yas not yet been added to the composition
    for i in range(len(self.composition)-1, -1, -1):
      if self.composition[i] == action_note:
        num_repeated += 1
      elif self.composition[i] == NOTE_OFF:
        contains_breaks = True
      elif self.composition[i] == NO_EVENT:
        contains_held_notes = True
      else:
        break

    if action_note == NOTE_OFF and num_repeated > 1:
      return True
    elif not contains_held_notes and not contains_breaks:
      if num_repeated > 4:
        return True
    elif contains_held_notes or contains_breaks:
      if num_repeated > 6:
        return True
    else:
      if num_repeated > 8:
        return True

    return False

  def reward_penalize_repeating(self,
                                action,
                                penalty_amount=-100.0):
    """Sets the previous reward to 0 if the same is played repeatedly.

    Allows more repeated notes if there are held notes or rests in between. If
    no penalty is applied will return the previous reward.

    Args:
      action: One-hot encoding of the chosen action.
      penalty_amount: The amount the model will be penalized if it plays
        repeating notes.
    Returns:
      Previous reward or 'penalty_amount'.
    """
    action_note = np.argmax(action)
    is_repeating = self.detect_repeating_notes(action_note)
    if is_repeating:
      return penalty_amount
    else:
      return 0.0

  def reward_penalize_autocorrelation(self,
                                      action,
                                      penalty_weight=3.0):
    """Reduces the previous reward if the composition is highly autocorrelated.

    Penalizes the model for creating a composition that is highly correlated
    with itself at lags of 1, 2, and 3 beats previous. This is meant to
    encourage variety in compositions.

    Args:
      action: One-hot encoding of the chosen action.
      penalty_weight: The default weight which will be multiplied by the sum
        of the autocorrelation coefficients, and subtracted from prev_reward.
    Returns:
      Float reward value.
    """
    composition = self.composition + [np.argmax(action)]
    lags = [1, 2, 3]
    sum_penalty = 0
    for lag in lags:
      coeff = rl_tuner_ops.autocorrelate(composition, lag=lag)
      if not np.isnan(coeff):
        if np.abs(coeff) > 0.15:
          sum_penalty += np.abs(coeff) * penalty_weight
    return -sum_penalty

  def detect_last_motif(self, composition=None, bar_length=8):
    """Detects if a motif was just played and if so, returns it.

    A motif should contain at least three distinct notes that are not note_on
    or note_off, and occur within the course of one bar.

    Args:
      composition: The composition in which the function will look for a
        recent motif. Defaults to the model's composition.
      bar_length: The number of notes in one bar.
    Returns:
      None if there is no motif, otherwise the motif in the same format as the
      composition.
    """
    if composition is None:
      composition = self.composition

    if len(composition) < bar_length:
      return None, 0

    last_bar = composition[-bar_length:]

    actual_notes = [a for a in last_bar if a != NO_EVENT and a != NOTE_OFF]
    num_unique_notes = len(set(actual_notes))
    if num_unique_notes >= 3:
      return last_bar, num_unique_notes
    else:
      return None, num_unique_notes

  def reward_motif(self, action, reward_amount=3.0):
    """Rewards the model for playing any motif.

    Motif must have at least three distinct notes in the course of one bar.
    There is a bonus for playing more complex motifs; that is, ones that involve
    a greater number of notes.

    Args:
      action: One-hot encoding of the chosen action.
      reward_amount: The amount that will be returned if the last note belongs
        to a motif.
    Returns:
      Float reward value.
    """

    composition = self.composition + [np.argmax(action)]
    motif, num_notes_in_motif = self.detect_last_motif(composition=composition)
    if motif is not None:
      motif_complexity_bonus = max((num_notes_in_motif - 3)*.3, 0)
      return reward_amount + motif_complexity_bonus
    else:
      return 0.0

  def detect_repeated_motif(self, action, bar_length=8):
    """Detects whether the last motif played repeats an earlier motif played.

    Args:
      action: One-hot encoding of the chosen action.
      bar_length: The number of beats in one bar. This determines how many beats
        the model has in which to play the motif.
    Returns:
      True if the note just played belongs to a motif that is repeated. False
      otherwise.
    """
    composition = self.composition + [np.argmax(action)]
    if len(composition) < bar_length:
      return False, None

    motif, _ = self.detect_last_motif(
        composition=composition, bar_length=bar_length)
    if motif is None:
      return False, None

    prev_composition = self.composition[:-(bar_length-1)]

    # Check if the motif is in the previous composition.
    for i in range(len(prev_composition) - len(motif) + 1):
      for j in range(len(motif)):
        if prev_composition[i + j] != motif[j]:
          break
      else:
        return True, motif
    return False, None

  def reward_repeated_motif(self,
                            action,
                            bar_length=8,
                            reward_amount=4.0):
    """Adds a big bonus to previous reward if the model plays a repeated motif.

    Checks if the model has just played a motif that repeats an ealier motif in
    the composition.

    There is also a bonus for repeating more complex motifs.

    Args:
      action: One-hot encoding of the chosen action.
      bar_length: The number of notes in one bar.
      reward_amount: The amount that will be added to the reward if the last
        note belongs to a repeated motif.
    Returns:
      Float reward value.
    """
    is_repeated, motif = self.detect_repeated_motif(action, bar_length)
    if is_repeated:
      actual_notes = [a for a in motif if a != NO_EVENT and a != NOTE_OFF]
      num_notes_in_motif = len(set(actual_notes))
      motif_complexity_bonus = max(num_notes_in_motif - 3, 0)
      return reward_amount + motif_complexity_bonus
    else:
      return 0.0

  def detect_sequential_interval(self, action, key=None):
    """Finds the melodic interval between the action and the last note played.

    Uses constants to represent special intervals like rests.

    Args:
      action: One-hot encoding of the chosen action
      key: The numeric values of notes belonging to this key. Defaults to
        C-major if not provided.
    Returns:
      An integer value representing the interval, or a constant value for
      special intervals.
    """
    if not self.composition:
      return 0, None, None

    prev_note = self.composition[-1]
    action_note = np.argmax(action)

    c_major = False
    if key is None:
      key = rl_tuner_ops.C_MAJOR_KEY
      c_notes = [2, 14, 26]
      g_notes = [9, 21, 33]
      e_notes = [6, 18, 30]
      c_major = True
      tonic_notes = [2, 14, 26]
      fifth_notes = [9, 21, 33]

    # get rid of non-notes in prev_note
    prev_note_index = len(self.composition) - 1
    while (prev_note == NO_EVENT or
           prev_note == NOTE_OFF) and prev_note_index >= 0:
      prev_note = self.composition[prev_note_index]
      prev_note_index -= 1
    if prev_note == NOTE_OFF or prev_note == NO_EVENT:
      tf.logging.debug('Action_note: %s, prev_note: %s', action_note, prev_note)
      return 0, action_note, prev_note

    tf.logging.debug('Action_note: %s, prev_note: %s', action_note, prev_note)

    # get rid of non-notes in action_note
    if action_note == NO_EVENT:
      if prev_note in tonic_notes or prev_note in fifth_notes:
        return (rl_tuner_ops.HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH,
                action_note, prev_note)
      else:
        return rl_tuner_ops.HOLD_INTERVAL, action_note, prev_note
    elif action_note == NOTE_OFF:
      if prev_note in tonic_notes or prev_note in fifth_notes:
        return (rl_tuner_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH,
                action_note, prev_note)
      else:
        return rl_tuner_ops.REST_INTERVAL, action_note, prev_note

    interval = abs(action_note - prev_note)

    if c_major and interval == rl_tuner_ops.FIFTH and (
        prev_note in c_notes or prev_note in g_notes):
      return rl_tuner_ops.IN_KEY_FIFTH, action_note, prev_note
    if c_major and interval == rl_tuner_ops.THIRD and (
        prev_note in c_notes or prev_note in e_notes):
      return rl_tuner_ops.IN_KEY_THIRD, action_note, prev_note

    return interval, action_note, prev_note

  def reward_preferred_intervals(self, action, scaler=5.0, key=None):
    """Dispenses reward based on the melodic interval just played.

    Args:
      action: One-hot encoding of the chosen action.
      scaler: This value will be multiplied by all rewards in this function.
      key: The numeric values of notes belonging to this key. Defaults to
        C-major if not provided.
    Returns:
      Float reward value.
    """
    interval, _, _ = self.detect_sequential_interval(action, key)
    tf.logging.debug('Interval:', interval)

    if interval == 0:  # either no interval or involving uninteresting rests
      tf.logging.debug('No interval or uninteresting.')
      return 0.0

    reward = 0.0

    # rests can be good
    if interval == rl_tuner_ops.REST_INTERVAL:
      reward = 0.05
      tf.logging.debug('Rest interval.')
    if interval == rl_tuner_ops.HOLD_INTERVAL:
      reward = 0.075
    if interval == rl_tuner_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH:
      reward = 0.15
      tf.logging.debug('Rest interval after 1st or 5th.')
    if interval == rl_tuner_ops.HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH:
      reward = 0.3

    # large leaps and awkward intervals bad
    if interval == rl_tuner_ops.SEVENTH:
      reward = -0.3
      tf.logging.debug('7th')
    if interval > rl_tuner_ops.OCTAVE:
      reward = -1.0
      tf.logging.debug('More than octave.')

    # common major intervals are good
    if interval == rl_tuner_ops.IN_KEY_FIFTH:
      reward = 0.1
      tf.logging.debug('In key 5th')
    if interval == rl_tuner_ops.IN_KEY_THIRD:
      reward = 0.15
      tf.logging.debug('In key 3rd')

    # smaller steps are generally preferred
    if interval == rl_tuner_ops.THIRD:
      reward = 0.09
      tf.logging.debug('3rd')
    if interval == rl_tuner_ops.SECOND:
      reward = 0.08
      tf.logging.debug('2nd')
    if interval == rl_tuner_ops.FOURTH:
      reward = 0.07
      tf.logging.debug('4th')

    # larger leaps not as good, especially if not in key
    if interval == rl_tuner_ops.SIXTH:
      reward = 0.05
      tf.logging.debug('6th')
    if interval == rl_tuner_ops.FIFTH:
      reward = 0.02
      tf.logging.debug('5th')

    tf.logging.debug('Interval reward', reward * scaler)
    return reward * scaler

  def detect_high_unique(self, composition):
    """Checks a composition to see if the highest note within it is repeated.

    Args:
      composition: A list of integers representing the notes in the piece.
    Returns:
      True if the lowest note was unique, False otherwise.
    """
    max_note = max(composition)
    if list(composition).count(max_note) == 1:
      return True
    else:
      return False

  def detect_low_unique(self, composition):
    """Checks a composition to see if the lowest note within it is repeated.

    Args:
      composition: A list of integers representing the notes in the piece.
    Returns:
      True if the lowest note was unique, False otherwise.
    """
    no_special_events = [x for x in composition
                         if x != NO_EVENT and x != NOTE_OFF]
    if no_special_events:
      min_note = min(no_special_events)
      if list(composition).count(min_note) == 1:
        return True
    return False

  def reward_high_low_unique(self, action, reward_amount=3.0):
    """Evaluates if highest and lowest notes in composition occurred once.

    Args:
      action: One-hot encoding of the chosen action.
      reward_amount: Amount of reward that will be given for the highest note
        being unique, and again for the lowest note being unique.
    Returns:
      Float reward value.
    """
    if len(self.composition) + 1 != self.num_notes_in_melody:
      return 0.0

    composition = np.array(self.composition)
    composition = np.append(composition, np.argmax(action))

    reward = 0.0

    if self.detect_high_unique(composition):
      reward += reward_amount

    if self.detect_low_unique(composition):
      reward += reward_amount

    return reward

  def detect_leap_up_back(self, action, steps_between_leaps=6):
    """Detects when the composition takes a musical leap, and if it is resolved.

    When the composition jumps up or down by an interval of a fifth or more,
    it is a 'leap'. The model then remembers that is has a 'leap direction'. The
    function detects if it then takes another leap in the same direction, if it
    leaps back, or if it gradually resolves the leap.

    Args:
      action: One-hot encoding of the chosen action.
      steps_between_leaps: Leaping back immediately does not constitute a
        satisfactory resolution of a leap. Therefore the composition must wait
        'steps_between_leaps' beats before leaping back.
    Returns:
      0 if there is no leap, 'LEAP_RESOLVED' if an existing leap has been
      resolved, 'LEAP_DOUBLED' if 2 leaps in the same direction were made.
    """
    if not self.composition:
      return 0

    outcome = 0

    interval, action_note, prev_note = self.detect_sequential_interval(action)

    if action_note == NOTE_OFF or action_note == NO_EVENT:
      self.steps_since_last_leap += 1
      tf.logging.debug('Rest, adding to steps since last leap. It is'
                       'now: %s', self.steps_since_last_leap)
      return 0

    # detect if leap
    if interval >= rl_tuner_ops.FIFTH or interval == rl_tuner_ops.IN_KEY_FIFTH:
      if action_note > prev_note:
        leap_direction = rl_tuner_ops.ASCENDING
        tf.logging.debug('Detected an ascending leap')
      else:
        leap_direction = rl_tuner_ops.DESCENDING
        tf.logging.debug('Detected a descending leap')

      # there was already an unresolved leap
      if self.composition_direction != 0:
        if self.composition_direction != leap_direction:
          tf.logging.debug('Detected a resolved leap')
          tf.logging.debug('Num steps since last leap: %s',
                           self.steps_since_last_leap)
          if self.steps_since_last_leap > steps_between_leaps:
            outcome = rl_tuner_ops.LEAP_RESOLVED
            tf.logging.debug('Sufficient steps before leap resolved, '
                             'awarding bonus')
          self.composition_direction = 0
          self.leapt_from = None
        else:
          tf.logging.debug('Detected a double leap')
          outcome = rl_tuner_ops.LEAP_DOUBLED

      # the composition had no previous leaps
      else:
        tf.logging.debug('There was no previous leap direction')
        self.composition_direction = leap_direction
        self.leapt_from = prev_note

      self.steps_since_last_leap = 0

    # there is no leap
    else:
      self.steps_since_last_leap += 1
      tf.logging.debug('No leap, adding to steps since last leap. '
                       'It is now: %s', self.steps_since_last_leap)

      # If there was a leap before, check if composition has gradually returned
      # This could be changed by requiring you to only go a 5th back in the
      # opposite direction of the leap.
      if (self.composition_direction == rl_tuner_ops.ASCENDING and
          action_note <= self.leapt_from) or (
              self.composition_direction == rl_tuner_ops.DESCENDING and
              action_note >= self.leapt_from):
        tf.logging.debug('detected a gradually resolved leap')
        outcome = rl_tuner_ops.LEAP_RESOLVED
        self.composition_direction = 0
        self.leapt_from = None

    return outcome

  def reward_leap_up_back(self, action, resolving_leap_bonus=5.0,
                          leaping_twice_punishment=-5.0):
    """Applies punishment and reward based on the principle leap up leap back.

    Large interval jumps (more than a fifth) should be followed by moving back
    in the same direction.

    Args:
      action: One-hot encoding of the chosen action.
      resolving_leap_bonus: Amount of reward dispensed for resolving a previous
        leap.
      leaping_twice_punishment: Amount of reward received for leaping twice in
        the same direction.
    Returns:
      Float reward value.
    """

    leap_outcome = self.detect_leap_up_back(action)
    if leap_outcome == rl_tuner_ops.LEAP_RESOLVED:
      tf.logging.debug('Leap resolved, awarding %s', resolving_leap_bonus)
      return resolving_leap_bonus
    elif leap_outcome == rl_tuner_ops.LEAP_DOUBLED:
      tf.logging.debug('Leap doubled, awarding %s', leaping_twice_punishment)
      return leaping_twice_punishment
    else:
      return 0.0

  def reward_interval_diversity(self):
    # TODO(natashajaques): music theory book also suggests having a mix of steps
    # that are both incremental and larger. Want to write a function that
    # rewards this. Could have some kind of interval_stats stored by
    # reward_preferred_intervals function.
    pass

  def generate_music_sequence(self, title='rltuner_sample',
                              visualize_probs=False, prob_image_name=None,
                              length=None, most_probable=False):
    """Generates a music sequence with the current model, and saves it to MIDI.

    The resulting MIDI file is saved to the model's output_dir directory. The
    sequence is generated by sampling from the output probabilities at each
    timestep, and feeding the resulting note back in as input to the model.

    Args:
      title: The name that will be used to save the output MIDI file.
      visualize_probs: If True, the function will plot the softmax
        probabilities of the model for each note that occur throughout the
        sequence. Useful for debugging.
      prob_image_name: The name of a file in which to save the softmax
        probability image. If None, the image will simply be displayed.
      length: The length of the sequence to be generated. Defaults to the
        num_notes_in_melody parameter of the model.
      most_probable: If True, instead of sampling each note in the sequence,
        the model will always choose the argmax, most probable note.
    """

    if length is None:
      length = self.num_notes_in_melody

    self.reset_composition()
    next_obs = self.prime_internal_models()
    tf.logging.info('Priming with note %s', np.argmax(next_obs))

    lengths = np.full(self.q_network.batch_size, 1, dtype=int)

    if visualize_probs:
      prob_image = np.zeros((self.input_size, length))

    generated_seq = [0] * length
    for i in range(length):
      input_batch = np.reshape(next_obs, (self.q_network.batch_size, 1,
                                          self.num_actions))
      if self.algorithm == 'g':
        (softmax, self.q_network.state_value,
         self.reward_rnn.state_value) = self.session.run(
             [self.action_softmax, self.q_network.state_tensor,
              self.reward_rnn.state_tensor],
             {self.q_network.melody_sequence: input_batch,
              self.q_network.initial_state: self.q_network.state_value,
              self.q_network.lengths: lengths,
              self.reward_rnn.melody_sequence: input_batch,
              self.reward_rnn.initial_state: self.reward_rnn.state_value,
              self.reward_rnn.lengths: lengths})
      else:
        softmax, self.q_network.state_value = self.session.run(
            [self.action_softmax, self.q_network.state_tensor],
            {self.q_network.melody_sequence: input_batch,
             self.q_network.initial_state: self.q_network.state_value,
             self.q_network.lengths: lengths})
      softmax = np.reshape(softmax, (self.num_actions))

      if visualize_probs:
        prob_image[:, i] = softmax  # np.log(1.0 + softmax)

      if most_probable:
        sample = np.argmax(softmax)
      else:
        sample = rl_tuner_ops.sample_softmax(softmax)
      generated_seq[i] = sample
      next_obs = np.array(rl_tuner_ops.make_onehot([sample],
                                                   self.num_actions)).flatten()

    tf.logging.info('Generated sequence: %s', generated_seq)
    # TODO(natashamjaques): Remove print statement once tf.logging outputs
    # to Jupyter notebooks (once the following issue is resolved:
    # https://github.com/tensorflow/tensorflow/issues/3047)
    print('Generated sequence:', generated_seq)

    melody = mlib.Melody(rl_tuner_ops.decoder(generated_seq,
                                              self.q_network.transpose_amount))

    sequence = melody.to_sequence(qpm=rl_tuner_ops.DEFAULT_QPM)
    filename = rl_tuner_ops.get_next_file_name(self.output_dir, title, 'mid')
    midi_io.sequence_proto_to_midi_file(sequence, filename)

    tf.logging.info('Wrote a melody to %s', self.output_dir)

    if visualize_probs:
      tf.logging.info('Visualizing note selection probabilities:')
      plt.figure()
      plt.imshow(prob_image, interpolation='none', cmap='Reds')
      plt.ylabel('Note probability')
      plt.xlabel('Time (beat)')
      plt.gca().invert_yaxis()
      if prob_image_name is not None:
        plt.savefig(self.output_dir + '/' + prob_image_name)
      else:
        plt.show()

  def evaluate_music_theory_metrics(self, num_compositions=10000, key=None,
                                    tonic_note=rl_tuner_ops.C_MAJOR_TONIC):
    """Computes statistics about music theory rule adherence.

    Args:
      num_compositions: How many compositions should be randomly generated
        for computing the statistics.
      key: The numeric values of notes belonging to this key. Defaults to C
        Major if not provided.
      tonic_note: The tonic/1st note of the desired key.

    Returns:
      A dictionary containing the statistics.
    """
    stat_dict = rl_tuner_eval_metrics.compute_composition_stats(
        self,
        num_compositions=num_compositions,
        composition_length=self.num_notes_in_melody,
        key=key,
        tonic_note=tonic_note)

    return stat_dict

  def save_model(self, name, directory=None):
    """Saves a checkpoint of the model and a .npz file with stored rewards.

    Args:
      name: String name to use for the checkpoint and rewards files.
      directory: Path to directory where the data will be saved. Defaults to
        self.output_dir if None is provided.
    """
    if directory is None:
      directory = self.output_dir

    save_loc = os.path.join(directory, name)
    self.saver.save(self.session, save_loc,
                    global_step=len(self.rewards_batched)*self.output_every_nth)

    self.save_stored_rewards(name)

  def save_stored_rewards(self, file_name):
    """Saves the models stored rewards over time in a .npz file.

    Args:
      file_name: Name of the file that will be saved.
    """
    training_epochs = len(self.rewards_batched) * self.output_every_nth
    filename = os.path.join(self.output_dir,
                            file_name + '-' + str(training_epochs))
    np.savez(filename,
             train_rewards=self.rewards_batched,
             train_music_theory_rewards=self.music_theory_rewards_batched,
             train_note_rnn_rewards=self.note_rnn_rewards_batched,
             eval_rewards=self.eval_avg_reward,
             eval_music_theory_rewards=self.eval_avg_music_theory_reward,
             eval_note_rnn_rewards=self.eval_avg_note_rnn_reward,
             target_val_list=self.target_val_list)

  def save_model_and_figs(self, name, directory=None):
    """Saves the model checkpoint, .npz file, and reward plots.

    Args:
      name: Name of the model that will be used on the images,
        checkpoint, and .npz files.
      directory: Path to directory where files will be saved.
        If None defaults to self.output_dir.
    """

    self.save_model(name, directory=directory)
    self.plot_rewards(image_name='TrainRewards-' + name + '.eps',
                      directory=directory)
    self.plot_evaluation(image_name='EvaluationRewards-' + name + '.eps',
                         directory=directory)
    self.plot_target_vals(image_name='TargetVals-' + name + '.eps',
                          directory=directory)

  def plot_rewards(self, image_name=None, directory=None):
    """Plots the cumulative rewards received as the model was trained.

    If image_name is None, should be used in jupyter notebook. If
    called outside of jupyter, execution of the program will halt and
    a pop-up with the graph will appear. Execution will not continue
    until the pop-up is closed.

    Args:
      image_name: Name to use when saving the plot to a file. If not
        provided, image will be shown immediately.
      directory: Path to directory where figure should be saved. If
        None, defaults to self.output_dir.
    """
    if directory is None:
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.rewards_batched))]
    plt.figure()
    plt.plot(x, self.rewards_batched)
    plt.plot(x, self.music_theory_rewards_batched)
    plt.plot(x, self.note_rnn_rewards_batched)
    plt.xlabel('Training epoch')
    plt.ylabel('Cumulative reward for last ' + str(reward_batch) + ' steps')
    plt.legend(['Total', 'Music theory', 'Note RNN'], loc='best')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def plot_evaluation(self, image_name=None, directory=None, start_at_epoch=0):
    """Plots the rewards received as the model was evaluated during training.

    If image_name is None, should be used in jupyter notebook. If
    called outside of jupyter, execution of the program will halt and
    a pop-up with the graph will appear. Execution will not continue
    until the pop-up is closed.

    Args:
      image_name: Name to use when saving the plot to a file. If not
        provided, image will be shown immediately.
      directory: Path to directory where figure should be saved. If
        None, defaults to self.output_dir.
      start_at_epoch: Training epoch where the plot should begin.
    """
    if directory is None:
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.eval_avg_reward))]
    start_index = start_at_epoch / self.output_every_nth
    plt.figure()
    plt.plot(x[start_index:], self.eval_avg_reward[start_index:])
    plt.plot(x[start_index:], self.eval_avg_music_theory_reward[start_index:])
    plt.plot(x[start_index:], self.eval_avg_note_rnn_reward[start_index:])
    plt.xlabel('Training epoch')
    plt.ylabel('Average reward')
    plt.legend(['Total', 'Music theory', 'Note RNN'], loc='best')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def plot_target_vals(self, image_name=None, directory=None):
    """Plots the target values used to train the model over time.

    If image_name is None, should be used in jupyter notebook. If
    called outside of jupyter, execution of the program will halt and
    a pop-up with the graph will appear. Execution will not continue
    until the pop-up is closed.

    Args:
      image_name: Name to use when saving the plot to a file. If not
        provided, image will be shown immediately.
      directory: Path to directory where figure should be saved. If
        None, defaults to self.output_dir.
    """
    if directory is None:
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.target_val_list))]

    plt.figure()
    plt.plot(x, self.target_val_list)
    plt.xlabel('Training epoch')
    plt.ylabel('Target value')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def prime_internal_models(self):
    """Primes both internal models based on self.priming_mode.

    Returns:
      A one-hot encoding of the note output by the q_network to be used as
      the initial observation.
    """
    self.prime_internal_model(self.target_q_network)
    self.prime_internal_model(self.reward_rnn)
    next_obs = self.prime_internal_model(self.q_network)
    return next_obs

  def restore_from_directory(self, directory=None, checkpoint_name=None,
                             reward_file_name=None):
    """Restores this model from a saved checkpoint.

    Args:
      directory: Path to directory where checkpoint is located. If
        None, defaults to self.output_dir.
      checkpoint_name: The name of the checkpoint within the
        directory.
      reward_file_name: The name of the .npz file where the stored
        rewards are saved. If None, will not attempt to load stored
        rewards.
    """
    if directory is None:
      directory = self.output_dir

    if checkpoint_name is not None:
      checkpoint_file = os.path.join(directory, checkpoint_name)
    else:
      tf.logging.info('Directory %s.', directory)
      checkpoint_file = tf.train.latest_checkpoint(directory)

    if checkpoint_file is None:
      tf.logging.fatal('Error! Cannot locate checkpoint in the directory')
      return
    # TODO(natashamjaques): Remove print statement once tf.logging outputs
    # to Jupyter notebooks (once the following issue is resolved:
    # https://github.com/tensorflow/tensorflow/issues/3047)
    print('Attempting to restore from checkpoint', checkpoint_file)
    tf.logging.info('Attempting to restore from checkpoint %s', checkpoint_file)

    self.saver.restore(self.session, checkpoint_file)

    if reward_file_name is not None:
      npz_file_name = os.path.join(directory, reward_file_name)
      # TODO(natashamjaques): Remove print statement once tf.logging outputs
      # to Jupyter notebooks (once the following issue is resolved:
      # https://github.com/tensorflow/tensorflow/issues/3047)
      print('Attempting to load saved reward values from file', npz_file_name)
      tf.logging.info('Attempting to load saved reward values from file %s',
                      npz_file_name)
      npz_file = np.load(npz_file_name)

      self.rewards_batched = npz_file['train_rewards']
      self.music_theory_rewards_batched = npz_file['train_music_theory_rewards']
      self.note_rnn_rewards_batched = npz_file['train_note_rnn_rewards']
      self.eval_avg_reward = npz_file['eval_rewards']
      self.eval_avg_music_theory_reward = npz_file['eval_music_theory_rewards']
      self.eval_avg_note_rnn_reward = npz_file['eval_note_rnn_rewards']
      self.target_val_list = npz_file['target_val_list']
