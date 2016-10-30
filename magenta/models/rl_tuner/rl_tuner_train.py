r"""Code to train a MelodyQ model.

To run this code on your local machine:
$ bazel run magenta/models/rl_tuner:rl_tuner_train -- \
--note_rnn_checkpoint_dir 'path' --midi_primer 'primer.mid' \
--training_data_path 'path.tfrecord'
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import tensorflow as tf

from magenta.common import tf_lib

import rl_tuner
import rl_tuner_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', '/home/natasha/Dropbox (MIT)/Google/RL-RNN-Project/rl_rnn_output/',
                           'Directory where the model will save its'
                           'compositions (midi files)')
tf.app.flags.DEFINE_string('checkpoint_name', '/home/natasha/Dropbox (MIT)/Google/RL-RNN-Projec/rl_rnn_output/rl_tuner_model.ckpt',
                           'Directory where the model will save checkpoints')
tf.app.flags.DEFINE_string('note_rnn_checkpoint_dir', '/home/natasha/Developer/magenta_my_fork/magenta/magenta/models/rl_tuner/testdata',
                           'Path to directory holding checkpoints for note rnn'
                           'melody prediction models. These will be loaded into'
                           'the NoteRNNLoader class object. The directory should'
                           'contain a train subdirectory')
tf.app.flags.DEFINE_string('note_rnn_checkpoint_name', 'model.ckpt-1994',
                           'Filename of a checkpoint within the '
                           'note_rnn_checkpoint_dir directory.')
tf.app.flags.DEFINE_string('model_save_dir', '/home/natasha/Dropbox (MIT)/Google/RL-RNN-Project/checkpoints',
                           'Directory where a checkpoint of the fully trained'
                           'model will be saved.')
tf.app.flags.DEFINE_string('midi_primer', './testdata/primer.mid',
                           'A midi file that can be used to prime the model')
tf.app.flags.DEFINE_integer('training_steps', 1000000,
                            'The number of steps used to train the model')
tf.app.flags.DEFINE_integer('exploration_steps', 500000,
                            'The number of steps over which the models'
                            'probability of taking a random action (exploring)'
                            'will be annealed from 1.0 to its normal'
                            'exploration probability. Typically about half the'
                            'training_steps')
tf.app.flags.DEFINE_string('exploration_mode', 'boltzmann',
                           'Can be either egreedy for epsilon-greedy or '
                           'boltzmann, which will sample from the models'
                           'output distribution to select the next action')
tf.app.flags.DEFINE_integer('output_every_nth', 50000,
                            'The number of steps before the model will evaluate'
                            'itself and store a checkpoint')
tf.app.flags.DEFINE_integer('num_notes_in_melody', 32,
                            'The number of notes in each composition')
tf.app.flags.DEFINE_float('reward_scaler', 0.1,
                          'The weight placed on music theory rewards')
tf.app.flags.DEFINE_string('training_data_path', '',
                           'Directory where the model will get melody training'
                           'examples')
tf.app.flags.DEFINE_string('algorithm', 'default',
                           'The name of the algorithm to use for training the'
                           'model. Can be default, psi, or g')


def main(_):
  hparams = rl_tuner_ops.small_model_hparams()

  dqn_hparams = tf_lib.HParams(random_action_probability=0.1,
                               store_every_nth=1,
                               train_every_nth=5,
                               minibatch_size=32,
                               discount_rate=0.5,
                               max_experience=100000,
                               target_network_update_rate=0.01)

  output_dir = os.path.join(FLAGS.output_dir, FLAGS.algorithm)
  output_ckpt = FLAGS.algorithm + '.ckpt'
  backup_checkpoint_file = os.path.join(FLAGS.note_rnn_checkpoint_dir, 
                                        FLAGS.note_rnn_checkpoint_name)

  rlt = rl_tuner.RLTuner(output_dir, FLAGS.note_rnn_checkpoint_dir, 
                         FLAGS.midi_primer, 
                         dqn_hparams=dqn_hparams, 
                         reward_scaler=FLAGS.reward_scaler,
                         save_name = output_ckpt,
                         output_every_nth=FLAGS.output_every_nth, 
                         backup_checkpoint_file=backup_checkpoint_file,
                         custom_hparams=hparams, 
                         num_notes_in_melody=FLAGS.num_notes_in_melody,
                         exploration_mode=FLAGS.exploration_mode,
                         algorithm=FLAGS.algorithm)

  tf.logging.info('Saving images and melodies to: %s', rlt.output_dir)

  tf.logging.info('\nTraining...')
  rlt.train(num_steps=FLAGS.training_steps,
               exploration_period=FLAGS.exploration_steps)

  tf.logging.info('\nFinished training. Saving output figures and composition.')
  rlt.plot_rewards(image_name='Rewards-' + FLAGS.algorithm + '.eps')

  rlt.generate_music_sequence(visualize_probs=True, title=FLAGS.algorithm,
                                 prob_image_name=FLAGS.algorithm + '.png')

  rlt.save_model_and_figs(FLAGS.algorithm)


if __name__ == '__main__':
  #flags.MarkFlagAsRequired('note_rnn_checkpoint_dir')
  #flags.MarkFlagAsRequired('midi_primer')
  tf.app.run()
