from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import poly_rnn_lib
import poly_rnn_graph
import functools

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('note_sequence_input', None,
                           'Polyphonic tfrecord NoteSequence file.')
tf.app.flags.DEFINE_string('run_dir', '/tmp/poly_rnn/logdir/run1',
                           'Path to the directory where checkpoints and '
                           'summary events will be saved during training and '
                           'evaluation. Separate subdirectories for training '
                           'events and eval events will be created within '
                           '`run_dir`. Multiple runs can be stored within the '
                           'parent directory of `run_dir`. Point TensorBoard '
                           'to the parent directory of `run_dir` to see all '
                           'your runs.')


def _loop(graph, itr, sess, inits=None, do_updates=True):
    i_h1 = np.zeros((graph.batch_size, graph.rnn_dim)).astype("float32")
    duration_mb, note_mb = next(itr)
    X_note_mb = note_mb[:-1]
    y_note_mb = note_mb[1:]
    X_duration_mb = duration_mb[:-1]
    y_duration_mb = duration_mb[1:]
    feed = {graph.note_inpt: X_note_mb,
            graph.note_target: y_note_mb,
            graph.duration_inpt: X_duration_mb,
            graph.duration_target: y_duration_mb,
            graph.init_h1: i_h1}
    if do_updates:
        outs = [graph.cost, graph.final_h1, graph.updates]
        train_loss, h1_l, _ = sess.run(outs, feed)
    else:
        outs = [graph.cost, graph.final_h1]
        train_loss, h1_l = sess.run(outs, feed)
    return train_loss, h1_l


def main(unused_argv):
  run_dir = os.path.expanduser(FLAGS.run_dir)
  train_dir = os.path.join(run_dir, 'train')
  if not os.path.exists(train_dir):
    tf.gfile.MakeDirs(train_dir)
  tf.logging.info('Train dir: %s', train_dir)

  graph = poly_rnn_graph.Graph(FLAGS.note_sequence_input)
  poly_rnn_lib.run_loop(
      functools.partial(_loop, graph),
      train_dir,
      graph.train_itr,
      graph.valid_itr,
      n_epochs=graph.num_epochs,
      checkpoint_delay=40,
      checkpoint_every_n_epochs=5,
      skip_minimums=True)

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
