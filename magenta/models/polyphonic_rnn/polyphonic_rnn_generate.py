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
"""Generate polyphonic sequences from a trained checkpoint."""

import os
import time

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
    'Path to the directory where the latest checkpoint will be loaded from.')
tf.app.flags.DEFINE_string(
    'output_dir', '/tmp/polyphonic_rnn/generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')


def sample(model_ckpt, runtime, note_sequence_input, sample_path, sample_len,
           temperature):
  graph = polyphonic_rnn_graph.Graph(note_sequence_input)
  graph.valid_itr.reset()
  duration_mb, note_mb = graph.valid_itr.next()
  polyphonic_rnn_lib.duration_and_pitch_to_midi(
      sample_path + '/gt_%i.mid' % runtime, duration_mb[:, 0], note_mb[:, 0])

  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, model_ckpt)
    i_h1 = np.zeros((graph.batch_size, graph.rnn_dim)).astype('float32')

    prime = 8
    note_mb = note_mb[:prime]
    duration_mb = duration_mb[:prime]
    for n in range(duration_mb.shape[1]):
      polyphonic_rnn_lib.duration_and_pitch_to_midi(
          sample_path + '/pre%i_%i.mid' % (n, runtime),
          duration_mb[:, n], note_mb[:, n], prime)

    note_inputs = note_mb
    duration_inputs = duration_mb

    shp = note_inputs.shape
    full_notes = np.zeros((sample_len, shp[1], shp[2]), dtype='float32')
    full_notes[:len(note_inputs)] = note_inputs[:]
    shp = duration_inputs.shape
    full_durations = np.zeros((sample_len, shp[1], shp[2]), dtype='float32')
    full_durations[:len(duration_inputs)] = duration_inputs[:]

    random_state = np.random.RandomState(1999)
    for j in range(sample_len - 1):
      # even predictions are note, odd are duration
      for ni in range(2 * graph.n_notes):
        feed = {graph.note_inpt: full_notes[j][None, :, :],
                graph.note_target: full_notes[j + 1][None, :, :],
                graph.duration_inpt: full_durations[j][None, :, :],
                graph.duration_target: full_durations[j + 1][None, :, :],
                graph.init_h1: i_h1}
        outs = []
        outs += graph.note_preds
        outs += graph.duration_preds
        outs += [graph.final_h1]
        r = sess.run(outs, feed)
        h_l = r[-1:]
        h1_l = h_l[-1]
        this_preds = r[:-1]
        this_probs = [
            polyphonic_rnn_lib.numpy_softmax(p, temperature=temperature)
            for p in this_preds]
        this_samples = [polyphonic_rnn_lib.numpy_sample_softmax(p, random_state)
                        for p in this_probs]
        if j < (len(note_inputs) - 1):
          # bypass sampling for now - still in prime seq
          continue
        # For debugging:
        # note_probs = this_probs[:graph.n_notes]
        # duration_probs = this_probs[graph.n_notes:]
        si = ni // 2
        if (ni % 2) == 0:
          # only put the single note in...
          full_notes[j + 1, :, si] = this_samples[si].ravel()
        if (ni % 2) == 1:
          full_durations[j + 1, :, si] = (
              this_samples[si + graph.n_notes].ravel())
      i_h1 = h1_l

    for n in range(full_durations.shape[1]):
      polyphonic_rnn_lib.duration_and_pitch_to_midi(
          sample_path + '/sampled%i_%i.mid' % (n, runtime),
          full_durations[:, n], full_notes[:, n], prime)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  checkpoint_dir = os.path.expanduser(FLAGS.checkpoint_dir)
  checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

  note_sequence_input = os.path.expanduser(FLAGS.note_sequence_input)

  sample_path = os.path.expanduser(FLAGS.output_dir)
  if not os.path.exists(sample_path):
    os.makedirs(sample_path)
  tf.logging.info('Writing MIDI files to %s', sample_path)
  sample(
      model_ckpt=checkpoint_file,
      runtime=time.time(),
      note_sequence_input=note_sequence_input,
      sample_path=sample_path,
      sample_len=50,
      temperature=.35)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
