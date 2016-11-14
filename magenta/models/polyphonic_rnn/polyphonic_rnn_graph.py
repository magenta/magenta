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
"""Manages the model's graph."""

# internal imports

import numpy as np
import tensorflow as tf

from magenta.models.polyphonic_rnn import polyphonic_rnn_lib


class Graph(object):

  def __init__(self, examples):
    self.num_epochs = 500
    self.batch_size = 32
    # ~1 step per seconds
    # 30 steps ~= 30 seconds
    sequence_length = 30

    self.train_itr = polyphonic_rnn_lib.TFRecordDurationAndPitchIterator(
        examples, self.batch_size, stop_index=.9,
        sequence_length=sequence_length)

    duration_mb, note_mb = next(self.train_itr)
    self.train_itr.reset()

    self.valid_itr = polyphonic_rnn_lib.TFRecordDurationAndPitchIterator(
        examples, self.batch_size, start_index=.9,
        sequence_length=sequence_length)

    num_note_features = note_mb.shape[-1]
    num_duration_features = duration_mb.shape[-1]
    n_note_symbols = len(self.train_itr.note_classes)
    n_duration_symbols = len(polyphonic_rnn_lib.TIME_CLASSES)
    self.n_notes = self.train_itr.simultaneous_notes
    note_embed_dim = 20
    duration_embed_dim = 4
    n_dim = 64
    h_dim = n_dim
    note_out_dims = self.n_notes * [n_note_symbols]
    duration_out_dims = self.n_notes * [n_duration_symbols]

    rnn_type = 'lstm'
    weight_norm_middle = False
    weight_norm_outputs = False

    share_note_and_target_embeddings = True
    share_all_embeddings = False
    share_output_parameters = False

    if rnn_type == 'lstm':
      rnn_fork = polyphonic_rnn_lib.lstm_fork
      rnn = polyphonic_rnn_lib.lstm
      self.rnn_dim = 2 * h_dim
    elif rnn_type == 'gru':
      rnn_fork = polyphonic_rnn_lib.gru_fork
      rnn = polyphonic_rnn_lib.gru
      self.rnn_dim = h_dim
    else:
      raise ValueError('Unknown rnn_type %s' % rnn_type)

    learning_rate = .0001
    grad_clip = 5.0
    random_state = np.random.RandomState(1999)

    self.duration_inpt = tf.placeholder(
        tf.float32, [None, self.batch_size, num_duration_features])
    self.note_inpt = tf.placeholder(
        tf.float32, [None, self.batch_size, num_note_features])

    self.note_target = tf.placeholder(
        tf.float32, [None, self.batch_size, num_note_features])
    self.duration_target = tf.placeholder(
        tf.float32, [None, self.batch_size, num_duration_features])
    self.init_h1 = tf.placeholder(tf.float32, [self.batch_size, self.rnn_dim])

    if share_note_and_target_embeddings:
      name_dur_emb = 'dur'
      name_note_emb = 'note'
    else:
      name_dur_emb = None
      name_note_emb = None
    duration_embed = polyphonic_rnn_lib.multiembedding(
        self.duration_inpt, n_duration_symbols, duration_embed_dim,
        random_state, name=name_dur_emb, share_all=share_all_embeddings)

    note_embed = polyphonic_rnn_lib.multiembedding(
        self.note_inpt, n_note_symbols, note_embed_dim, random_state,
        name=name_note_emb, share_all=share_all_embeddings)

    scan_inp = tf.concat(2, [duration_embed, note_embed])
    scan_inp_dim = (self.n_notes * duration_embed_dim + self.n_notes *
                    note_embed_dim)

    def step(inp_t, h1_tm1):
      h1_t_proj, h1gate_t_proj = rnn_fork(
          [inp_t], [scan_inp_dim], h_dim, random_state,
          weight_norm=weight_norm_middle)
      h1_t = rnn(h1_t_proj, h1gate_t_proj, h1_tm1, h_dim, h_dim, random_state)
      return h1_t

    h1_f = polyphonic_rnn_lib.scan(step, [scan_inp], [self.init_h1])
    h1 = h1_f
    self.final_h1 = polyphonic_rnn_lib.ni(h1, -1)

    target_note_embed = polyphonic_rnn_lib.multiembedding(
        self.note_target, n_note_symbols, note_embed_dim, random_state,
        name=name_note_emb, share_all=share_all_embeddings)
    target_note_masked = polyphonic_rnn_lib.automask(
        target_note_embed, self.n_notes)
    target_duration_embed = polyphonic_rnn_lib.multiembedding(
        self.duration_target, n_duration_symbols, duration_embed_dim,
        random_state, name=name_dur_emb, share_all=share_all_embeddings)
    target_duration_masked = polyphonic_rnn_lib.automask(
        target_duration_embed, self.n_notes)

    costs = []
    self.note_preds = []
    self.duration_preds = []
    if share_output_parameters:
      name_note = 'note_pred'
      name_dur = 'dur_pred'
    else:
      name_note = None
      name_dur = None
    for i in range(self.n_notes):
      note_pred = polyphonic_rnn_lib.linear(
          [
              h1[:, :, :h_dim], scan_inp, target_note_masked[i],
              target_duration_masked[i]
          ], [
              h_dim, scan_inp_dim,
              self.n_notes * note_embed_dim, self.n_notes * duration_embed_dim
          ],
          note_out_dims[i], random_state, weight_norm=weight_norm_outputs,
          name=name_note)
      duration_pred = polyphonic_rnn_lib.linear(
          [
              h1[:, :, :h_dim], scan_inp, target_note_masked[i],
              target_duration_masked[i]
          ], [
              h_dim, scan_inp_dim, self.n_notes * note_embed_dim,
              self.n_notes * duration_embed_dim
          ],
          duration_out_dims[i], random_state, weight_norm=weight_norm_outputs,
          name=name_dur)
      n = polyphonic_rnn_lib.categorical_crossentropy(
          polyphonic_rnn_lib.softmax(note_pred), self.note_target[:, :, i])
      d = polyphonic_rnn_lib.categorical_crossentropy(
          polyphonic_rnn_lib.softmax(duration_pred),
          self.duration_target[:, :, i])
      cost = (n_duration_symbols * tf.reduce_mean(n) + n_note_symbols *
              tf.reduce_mean(d))
      cost /= (n_duration_symbols + n_note_symbols)
      self.note_preds.append(note_pred)
      self.duration_preds.append(duration_pred)
      costs.append(cost)

    # 4 notes pitch and 4 notes duration
    self.cost = sum(costs) / float(self.n_notes + self.n_notes)

    params = tf.trainable_variables()
    grads = tf.gradients(self.cost, params)
    grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]
    opt = tf.train.AdamOptimizer(learning_rate, use_locking=True)
    self.updates = opt.apply_gradients(zip(grads, params))
