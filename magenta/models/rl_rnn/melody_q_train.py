r"""Code to train a MelodyQ model.

To run this code on your local machine:
$ bazel run rl_rnn:melody_q_train -- \
--melody_checkpoint_dir 'path' --midi_primer 'primer.mid' \
--training_data_path 'path.tfrecord'
"""

import matplotlib.pyplot as plt
import tensorflow as tf


import melody_q
import rl_rnn_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', '/tmp/melodyq',
                           'Directory where the model will save its'
                           'compositions (midi files)')
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/melodyq/melody_q_model',
                           'Directory where the model will save checkpoints')
tf.app.flags.DEFINE_string('melody_checkpoint_dir', '',
                           'Path to directory holding checkpoints for basic rnn'
                           'melody prediction models. These will be loaded into'
                           'the MelodyRNN class object. The directory should'
                           'contain a train subdirectory')
tf.app.flags.DEFINE_string('midi_primer', '',
                           'A midi file that can be used to prime the model')
tf.app.flags.DEFINE_string('before_image', 'before_rl.png',
                           'Name for the file that will store an image of the'
                           'models note probabilities as it composes before RL'
                           'is applied')
tf.app.flags.DEFINE_string('after_image', 'after_rl.png',
                           'Name for the file that will store an image of the'
                           'models note probabilities as it composes after RL'
                           'is applied')
tf.app.flags.DEFINE_integer('training_steps', 4000000,
                            'The number of steps used to train the model')
tf.app.flags.DEFINE_integer('exploration_steps', 2000000,
                            'The number of steps over which the models'
                            'probability of taking a random action (exploring)'
                            'will be annealed from 1.0 to its normal'
                            'exploration probability. Typically about half the'
                            'training_steps')
tf.app.flags.DEFINE_string('training_data_path', '',
                           'Directory where the model will get melody training'
                           'examples')


def main(_):
  hparams = rl_rnn_ops.small_model_hparams()

  mq_net = melody_q.MelodyQNetwork(FLAGS.output_dir, FLAGS.checkpoint_dir,
                                   FLAGS.melody_checkpoint_dir,
                                   FLAGS.midi_primer, custom_hparams=hparams,
                                   training_data_path=FLAGS.training_data_path)

  logging.info('Generating an initial music sequence')
  logging.info('Saving images and melodies to: %s', mq_net.output_dir)
  mq_net.generate_music_sequence(visualize_probs=True,
                                 prob_image_name=FLAGS.before_image)

  logging.info('\nTraining...')
  mq_net.train(num_steps=FLAGS.training_steps,
               exploration_period=FLAGS.exploration_steps)

  logging.info('\nFinished training. Saving output figures and composition.')
  plt.figure()
  plt.plot(mq_net.rewards_batched)
  plt.savefig(mq_net.output_dir + 'rewards_over_time.png')

  mq_net.generate_music_sequence(visualize_probs=True,
                                 prob_image_name=FLAGS.after_image)


if __name__ == '__main__':
  flags.MarkFlagAsRequired('melody_checkpoint_dir')
  flags.MarkFlagAsRequired('midi_primer')
  app.run()
