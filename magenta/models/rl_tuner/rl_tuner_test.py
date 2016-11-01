"""Tests for RLTuner and by proxy NoteRNNLoader.

To run this code:
$ bazel test rl_tuner:rl_tuner_test -- \
--test_tempdir 'path' --test_srcdir 'path'
"""

import os

import tensorflow as tf

import rl_tuner
import rl_tuner_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_tempdir', '',
                           'Directory where temporary test files are stored.')
tf.app.flags.DEFINE_string('test_srcdir', '',
                           'Directory containing source code to test.')

class RLTunerTest(tf.test.TestCase):

  def setUp(self):
    self.output_dir = os.path.join(FLAGS.test_tmpdir, 'rl_tuner_test')
    base_dir = FLAGS.test_srcdir
    self.checkpoint_dir = os.path.join(base_dir, 'testdata/')
    self.checkpoint_file = os.path.join(self.checkpoint_dir,'model.ckpt-1994')
    self.midi_primer = base_dir + 'testdata/primer.mid'
    self.hparams = rl_tuner_ops.small_model_hparams()

  def testDataAvailable(self):
    self.assertTrue(os.path.exists(self.midi_primer))
    self.assertTrue(os.path.exists(self.checkpoint_dir))

  def testInitializationAndPriming(self):
    rlt = rl_tuner.RLTuner(self.output_dir, self.checkpoint_dir, 
                          self.midi_primer, custom_hparams=self.hparams,
                          backup_checkpoint_file=self.checkpoint_file)

    initial_note = rlt.prime_internal_models()
    self.assertTrue(initial_note is not None)

  def testInitialGeneration(self):
    rlt = rl_tuner.RLTuner(self.output_dir, self.checkpoint_dir, 
                          self.midi_primer, custom_hparams=self.hparams,
                          backup_checkpoint_file=self.checkpoint_file)

    plot_name = 'test_initial_plot.png'
    rlt.generate_music_sequence(visualize_probs=True,
                               prob_image_name=plot_name)
    output_path = os.path.join(self.output_dir, plot_name)
    self.assertTrue(os.path.exists(output_path))

  def testAction(self):
    rlt = rl_tuner.RLTuner(self.output_dir, self.checkpoint_dir, 
                          self.midi_primer, custom_hparams=self.hparams,
                          backup_checkpoint_file=self.checkpoint_file)

    initial_note = rlt.prime_internal_models()

    action = rlt.action(initial_note, 100, enable_random=False)
    self.assertTrue(action is not None)

  def testRewardNetwork(self):
    rlt = rl_tuner.RLTuner(self.output_dir, self.checkpoint_dir, 
                          self.midi_primer, custom_hparams=self.hparams,
                          backup_checkpoint_file=self.checkpoint_file)

    zero_state = rlt.q_network.get_zero_state()
    priming_note = rlt.get_random_note()

    reward_scores = rlt.get_reward_rnn_scores(priming_note, zero_state)
    self.assertTrue(reward_scores is not None)

  def testTraining(self):
    rlt = rl_tuner.RLTuner(self.output_dir, self.checkpoint_dir, 
                          self.midi_primer, custom_hparams=self.hparams,
                          backup_checkpoint_file=self.checkpoint_file,
                          output_every_nth=30)
    rlt.train(num_steps=31, exploration_period=3)

    self.assertTrue(len(rlt.composition) == 31)
    self.assertTrue(os.path.exists(self.save_path + '-30'))
    self.assertTrue(len(rlt.rewards_batched) >= 1)

  def testCompositionStats(self):
    rlt = rl_tuner.RLTuner(self.output_dir, self.checkpoint_dir, 
                          self.midi_primer, custom_hparams=self.hparams,
                          backup_checkpoint_file=self.checkpoint_file,
                          output_every_nth=30)
    stat_dict = rlt.compute_composition_stats(num_compositions=10)

    self.assertTrue(stat_dict['num_repeated_notes'] > 1)
    self.assertTrue(len(stat_dict['autocorrelation1']) > 1)

if __name__ == '__main__':
  tf.test.main()
