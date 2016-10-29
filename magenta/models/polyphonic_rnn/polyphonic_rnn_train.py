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
"""Train polyphonic RNN."""

import functools
import os

# internal imports

import numpy as np
import tensorflow as tf

from magenta.models.polyphonic_rnn import polyphonic_rnn_graph
from magenta.models.polyphonic_rnn import polyphonic_rnn_lib

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'note_sequence_input', None, 'Polyphonic tfrecord NoteSequence file.')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/tmp/polyphonic_rnn/checkpoints',
    'Path to the directory where checkpoints and summary events will be saved '
    'during training')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')


def _loop(graph, itr, sess, inits=None, do_updates=True):
  i_h1 = np.zeros((graph.batch_size, graph.rnn_dim)).astype('float32')
  duration_mb, note_mb = next(itr)
  x_note_mb = note_mb[:-1]
  y_note_mb = note_mb[1:]
  x_duration_mb = duration_mb[:-1]
  y_duration_mb = duration_mb[1:]
  feed = {graph.note_inpt: x_note_mb,
          graph.note_target: y_note_mb,
          graph.duration_inpt: x_duration_mb,
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
  tf.logging.set_verbosity(FLAGS.log)

  checkpoint_dir = os.path.expanduser(FLAGS.checkpoint_dir)
  if not os.path.exists(checkpoint_dir):
    tf.gfile.MakeDirs(checkpoint_dir)
  tf.logging.info('Checkpoint dir: %s', checkpoint_dir)

  graph = polyphonic_rnn_graph.Graph(FLAGS.note_sequence_input)
  polyphonic_rnn_lib.run_loop(
      functools.partial(_loop, graph),
      checkpoint_dir,
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
