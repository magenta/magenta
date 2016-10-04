r"""Code to train a MelodyQ model.

To run this code on your local machine:
$ bazel run magenta/models/rl_rnn:melody_q_train -- \
--melody_checkpoint_dir 'path' --midi_primer 'primer.mid' \
--training_data_path 'path.tfrecord'
"""

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf

from magenta.lib import tf_lib

import melody_q
import rl_rnn_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', '/home/natasha/Dropbox (MIT)/Google/Project/rl_rnn_output',
                           'Directory where the model will save its'
                           'compositions (midi files)')
tf.app.flags.DEFINE_string('checkpoint_name', '/home/natasha/Dropbox (MIT)/Google/Project/rl_rnn_output/rl_rnn_model.ckpt',
                           'Directory where the model will save checkpoints')
tf.app.flags.DEFINE_string('melody_checkpoint_dir', '/home/natasha/Developer/magenta_my_fork/magenta/magenta/models/rl_rnn/testdata',
                           'Path to directory holding checkpoints for basic rnn'
                           'melody prediction models. These will be loaded into'
                           'the MelodyRNN class object. The directory should'
                           'contain a train subdirectory')
tf.app.flags.DEFINE_string('model_save_dir', '/home/natasha/Dropbox (MIT)/Google/Project/checkpoints',
                           'Directory where a checkpoint of the fully trained'
                           'model will be saved.')
tf.app.flags.DEFINE_string('midi_primer', '/home/natasha/Developer/magenta_my_fork/magenta/magenta/models/rl_rnn/testdata/primer.mid',
                           'A midi file that can be used to prime the model')
tf.app.flags.DEFINE_string('before_image', 'before_rl.png',
                           'Name for the file that will store an image of the'
                           'models note probabilities as it composes before RL'
                           'is applied')
tf.app.flags.DEFINE_string('after_image', 'psi.png',
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
tf.app.flags.DEFINE_integer('num_notes_in_melody', 32,
                            'The number of notes in each composition')
tf.app.flags.DEFINE_float('reward_scaler', 2.5,
                          'The weight placed on music theory rewards')
tf.app.flags.DEFINE_string('training_data_path', '',
                           'Directory where the model will get melody training'
                           'examples')
tf.app.flags.DEFINE_string('algorithm', 'default',
                           'The name of the algorithm to use for training the'
                           'model. Can be default, psi, or g')


def main(_):
  hparams = rl_rnn_ops.small_model_hparams()

  dqn_hparams = tf_lib.HParams(random_action_probability=0.1,
                               store_every_nth=1,
                               train_every_nth=5,
                               minibatch_size=32,
                               discount_rate=0.5,
                               max_experience=100000,
                               target_network_update_rate=0.01)

  mq_net = melody_q.MelodyQNetwork(FLAGS.output_dir, FLAGS.checkpoint_name,
                                   FLAGS.melody_checkpoint_dir, FLAGS.midi_primer, 
                                   dqn_hparams=dqn_hparams, 
                                   reward_scaler=FLAGS.reward_scaler,
                                   output_every_nth=100000, 
                                   backup_checkpoint_file=FLAGS.melody_checkpoint_dir + '/model.ckpt-1994',
                                   custom_hparams=hparams, num_notes_in_melody=FLAGS.num_notes_in_melody,
                                   stochastic_observations=False,
                                   algorithm=FLAGS.algorithm)

  tf.logging.info('Generating an initial music sequence')
  tf.logging.info('Saving images and melodies to: %s', mq_net.output_dir)
  mq_net.generate_music_sequence(visualize_probs=True,
                                 prob_image_name=FLAGS.before_image)

  tf.logging.info('\nTraining...')
  mq_net.train(num_steps=FLAGS.training_steps,
               exploration_period=FLAGS.exploration_steps)

  tf.logging.info('\nFinished training. Saving output figures and composition.')
  matplotlib.pyplot.figure()
  matplotlib.pyplot.plot(mq_net.rewards_batched)
  matplotlib.pyplot.savefig(mq_net.output_dir + FLAGS.algorithm +'-rewards_over_time.png')

  mq_net.generate_music_sequence(visualize_probs=True,
                                 prob_image_name=FLAGS.after_image)

  mq_net.save_model(FLAGS.model_save_dir, FLAGS.algorithm)


if __name__ == '__main__':
  #flags.MarkFlagAsRequired('melody_checkpoint_dir')
  #flags.MarkFlagAsRequired('midi_primer')
  tf.app.run()
