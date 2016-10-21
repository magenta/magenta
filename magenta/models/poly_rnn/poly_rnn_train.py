from __future__ import print_function
import tensorflow as tf
import numpy as np
from tfkdllib import run_loop
from tfkdllib import tfrecord_duration_and_pitch_iterator
import poly_rnn_graph
import functools

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None, 'Polyphonic tfrecord file')


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
  graph = poly_rnn_graph.Graph(FLAGS.input)
  run_loop(functools.partial(_loop, graph), graph.train_itr, graph.valid_itr,
           n_epochs=graph.num_epochs,
           checkpoint_delay=40,
           checkpoint_every_n_epochs=5)

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
