"""Tests for MelodyQNetwork and by proxy MelodyRNN.

To run this code:
$ bazel test rl_rnn:melody_q_test
"""

import os


from ....rl_rnn import melody_q
from ....rl_rnn import rl_rnn_ops


class MelodyQTest(magic test thingy):

  def setUp(self):
    self.output_dir = os.path.join(FLAGS.test_tmpdir, 'melodyq_test')
    self.log_dir = os.path.join(FLAGS.test_tmpdir, 'melodyq_test/checkpoint')
    base_dir = os.path.join(
        FLAGS.test_srcdir,
        '.../rl_rnn/')
    self.checkpoint_dir = base_dir + 'testdata/'
    self.checkpoint_file = self.checkpoint_dir + 'train/model.ckpt-1994'
    self.midi_primer = base_dir + 'testdata/primer.mid'
    self.hparams = rl_rnn_ops.small_model_hparams()

  def testDataAvailable(self):
    self.assertTrue(os.path.exists(self.midi_primer))
    self.assertTrue(os.path.exists(self.checkpoint_dir))

  def testInitializationAndPriming(self):
    mq = melody_q.MelodyQNetwork(self.output_dir, self.log_dir,
                                 self.checkpoint_dir, self.midi_primer,
                                 custom_hparams=self.hparams,
                                 backup_checkpoint_file=self.checkpoint_file)

    initial_note = mq.prime_q_model()
    self.assertTrue(initial_note is not None)

  def testInitialGeneration(self):
    mq = melody_q.MelodyQNetwork(self.output_dir, self.log_dir,
                                 self.checkpoint_dir, self.midi_primer,
                                 custom_hparams=self.hparams,
                                 backup_checkpoint_file=self.checkpoint_file)

    plot_name = 'test_initial_plot.png'
    mq.generate_music_sequence(visualize_probs=True,
                               prob_image_name=plot_name)
    output_path = os.path.join(self.output_dir, plot_name)
    self.assertTrue(os.path.exists(output_path))

  def testAction(self):
    mq = melody_q.MelodyQNetwork(self.output_dir, self.log_dir,
                                 self.checkpoint_dir, self.midi_primer,
                                 custom_hparams=self.hparams,
                                 backup_checkpoint_file=self.checkpoint_file)

    initial_note = mq.prime_q_model()

    action = mq.action(initial_note, 100, enable_random=False)
    self.assertTrue(action is not None)

  def testRewardNetwork(self):
    mq = melody_q.MelodyQNetwork(self.output_dir, self.log_dir,
                                 self.checkpoint_dir, self.midi_primer,
                                 custom_hparams=self.hparams,
                                 backup_checkpoint_file=self.checkpoint_file)

    zero_state = mq.q_network.get_zero_state()
    priming_note = mq.get_random_note()

    reward_scores = mq.get_reward_rnn_scores(priming_note, zero_state)
    self.assertTrue(reward_scores is not None)

  def testTraining(self):
    mq = melody_q.MelodyQNetwork(self.output_dir, self.log_dir,
                                 self.checkpoint_dir, self.midi_primer,
                                 custom_hparams=self.hparams,
                                 output_every_nth=30,
                                 backup_checkpoint_file=self.checkpoint_file)
    mq.train(num_steps=31, exploration_period=3)

    self.assertTrue(len(mq.composition) == 31)
    self.assertTrue(os.path.exists(self.log_dir + '-30'))
    self.assertTrue(len(mq.rewards_batched) >= 1)

  def testCompositionStats(self):
    mq = melody_q.MelodyQNetwork(
        self.output_dir,
        self.log_dir,
        self.checkpoint_dir,
        self.midi_primer,
        custom_hparams=self.hparams,
        output_every_nth=30,
        backup_checkpoint_file=self.checkpoint_file)
    stat_dict = mq.compute_composition_stats(num_compositions=10)

    self.assertTrue(stat_dict['num_repeated_notes'] > 1)
    self.assertTrue(len(stat_dict['autocorrelation1']) > 1)

if __name__ == '__main__':
  magic test thingy.main()
