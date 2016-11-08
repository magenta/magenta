"""Tests for RLTuner and by proxy NoteRNNLoader.

To run this code:
$ bazel test rl_tuner:rl_tuner_test
"""

import os

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

import rl_tuner
import rl_tuner_ops


class RLTunerTest(tf.test.TestCase):

  def setUp(self):
    self.output_dir = '/tmp/rl_tuner_test'
    self.hparams = rl_tuner_ops.small_model_hparams()

  def testInitializationAndPriming(self):
    rlt = rl_tuner.RLTuner(self.output_dir, custom_hparams=self.hparams)

    initial_note = rlt.prime_internal_models()
    self.assertTrue(initial_note is not None)

  def testInitialGeneration(self):
    rlt = rl_tuner.RLTuner(self.output_dir, custom_hparams=self.hparams)

    plot_name = 'test_initial_plot.png'
    rlt.generate_music_sequence(visualize_probs=True,
                               prob_image_name=plot_name)
    output_path = os.path.join(self.output_dir, plot_name)
    self.assertTrue(os.path.exists(output_path))

  def testAction(self):
    rlt = rl_tuner.RLTuner(self.output_dir, custom_hparams=self.hparams)

    initial_note = rlt.prime_internal_models()

    action = rlt.action(initial_note, 100, enable_random=False)
    self.assertTrue(action is not None)

  def testRewardNetwork(self):
    rlt = rl_tuner.RLTuner(self.output_dir, custom_hparams=self.hparams)

    zero_state = rlt.q_network.get_zero_state()
    priming_note = rlt.get_random_note()

    reward_scores = rlt.get_reward_rnn_scores(priming_note, zero_state)
    self.assertTrue(reward_scores is not None)

  def testTraining(self):
    rlt = rl_tuner.RLTuner(self.output_dir, custom_hparams=self.hparams,
                           output_every_nth=30)
    rlt.train(num_steps=31, exploration_period=3)

    self.assertTrue(os.path.exists(rlt.save_path + '-30'))
    self.assertTrue(len(rlt.rewards_batched) >= 1)
    self.assertTrue(len(rlt.eval_avg_reward) >= 1)

  def testCompositionStats(self):
    rlt = rl_tuner.RLTuner(self.output_dir, custom_hparams=self.hparams,
                           output_every_nth=30)
    stat_dict = rlt.evaluate_music_theory_metrics(num_compositions=10)

    self.assertTrue(stat_dict['num_repeated_notes'] > 1)
    self.assertTrue(len(stat_dict['autocorrelation1']) > 1)

if __name__ == '__main__':
  tf.test.main()
