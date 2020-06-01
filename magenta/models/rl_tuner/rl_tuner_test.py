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

"""Tests for RLTuner and by proxy NoteRNNLoader.

To run this code:
$ python magenta/models/rl_tuner/rl_tuner_test.py
"""

import os
import os.path
import tempfile

from magenta.models.rl_tuner import note_rnn_loader
from magenta.models.rl_tuner import rl_tuner
import matplotlib
import matplotlib.pyplot as plt  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# Need to use 'Agg' option for plotting and saving files from command line.
# Can't use 'Agg' in RL Tuner because it breaks plotting in notebooks.
# pylint: disable=g-import-not-at-top,wrong-import-position
matplotlib.use('Agg')

# pylint: enable=g-import-not-at-top,wrong-import-position


class RLTunerTest(tf.test.TestCase):

  def setUp(self):
    self.output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    self.checkpoint_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    graph = tf.Graph()
    self.session = tf.Session(graph=graph)
    note_rnn = note_rnn_loader.NoteRNNLoader(
        graph, scope='test', checkpoint_dir=None)
    note_rnn.initialize_new(self.session)
    with graph.as_default():
      saver = tf.train.Saver(var_list=note_rnn.get_variable_name_dict())
      saver.save(
          self.session,
          os.path.join(self.checkpoint_dir, 'model.ckpt'))

  def tearDown(self):
    self.session.close()

  def testInitializationAndPriming(self):
    rlt = rl_tuner.RLTuner(
        self.output_dir, note_rnn_checkpoint_dir=self.checkpoint_dir)

    initial_note = rlt.prime_internal_models()
    self.assertTrue(initial_note is not None)

  def testInitialGeneration(self):
    rlt = rl_tuner.RLTuner(
        self.output_dir, note_rnn_checkpoint_dir=self.checkpoint_dir)

    plot_name = 'test_initial_plot.png'
    rlt.generate_music_sequence(visualize_probs=True,
                                prob_image_name=plot_name)
    output_path = os.path.join(self.output_dir, plot_name)
    self.assertTrue(os.path.exists(output_path))

  def testAction(self):
    rlt = rl_tuner.RLTuner(
        self.output_dir, note_rnn_checkpoint_dir=self.checkpoint_dir)

    initial_note = rlt.prime_internal_models()

    action = rlt.action(initial_note, 100, enable_random=False)
    self.assertTrue(action is not None)

  def testRewardNetwork(self):
    rlt = rl_tuner.RLTuner(
        self.output_dir, note_rnn_checkpoint_dir=self.checkpoint_dir)

    zero_state = rlt.q_network.get_zero_state()
    priming_note = rlt.get_random_note()

    reward_scores = rlt.get_reward_rnn_scores(priming_note, zero_state)
    self.assertTrue(reward_scores is not None)

  def testTraining(self):
    rlt = rl_tuner.RLTuner(
        self.output_dir, note_rnn_checkpoint_dir=self.checkpoint_dir,
        output_every_nth=30)
    rlt.train(num_steps=31, exploration_period=3)

    checkpoint_dir = os.path.dirname(rlt.save_path)
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f))]
    checkpoint_step_30 = [
        f for f in checkpoint_files
        if os.path.basename(rlt.save_path) + '-30' in f]

    self.assertTrue(len(checkpoint_step_30))

    self.assertTrue(len(rlt.rewards_batched) >= 1)
    self.assertTrue(len(rlt.eval_avg_reward) >= 1)

  def testCompositionStats(self):
    rlt = rl_tuner.RLTuner(
        self.output_dir, note_rnn_checkpoint_dir=self.checkpoint_dir,
        output_every_nth=30)
    stat_dict = rlt.evaluate_music_theory_metrics(num_compositions=10)

    self.assertTrue(stat_dict['num_repeated_notes'] >= 0)
    self.assertTrue(len(stat_dict['autocorrelation1']) > 1)

if __name__ == '__main__':
  tf.test.main()
