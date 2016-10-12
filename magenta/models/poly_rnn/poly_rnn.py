from __future__ import print_function
import tensorflow as tf
import numpy as np
from tfkdllib import ni, scan
from tfkdllib import Multiembedding, GRUFork, GRU, Linear, Automask
from tfkdllib import LSTM, LSTMFork
from tfkdllib import softmax, categorical_crossentropy
from tfkdllib import run_loop
from tfkdllib import tfrecord_duration_and_pitch_iterator

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None, 'Polyphonic tfrecord file')

def main(unused_argv):
  num_epochs = 500
  batch_size = 32
  # ~1 step per seconds
  # 30 steps ~= 30 seconds
  sequence_length = 30

  train_itr = tfrecord_duration_and_pitch_iterator(FLAGS.input,
                                                   batch_size,
                                                   stop_index=.9,
                                                   sequence_length=sequence_length)

  duration_mb, note_mb = next(train_itr)
  train_itr.reset()

  valid_itr = tfrecord_duration_and_pitch_iterator(FLAGS.input,
                                                   batch_size,
                                                   start_index=.9,
                                                   sequence_length=sequence_length)

  num_note_features = note_mb.shape[-1]
  num_duration_features = duration_mb.shape[-1]
  n_note_symbols = len(train_itr.note_classes)
  n_duration_symbols = len(train_itr.time_classes)
  n_notes = train_itr.simultaneous_notes
  note_embed_dim = 20
  duration_embed_dim = 4
  n_dim = 64
  h_dim = n_dim
  note_out_dims = n_notes * [n_note_symbols]
  duration_out_dims = n_notes * [n_duration_symbols]

  rnn_type = "lstm"
  weight_norm_middle = False
  weight_norm_outputs = False

  share_note_and_target_embeddings = True
  share_all_embeddings = False
  share_output_parameters = False

  if rnn_type == "lstm":
      RNNFork = LSTMFork
      RNN = LSTM
      rnn_dim = 2 * h_dim
  elif rnn_type == "gru":
      RNNFork = GRUFork
      RNN = GRU
      rnn_dim = h_dim
  else:
      raise ValueError("Unknown rnn_type %s" % rnn_type)

  learning_rate = .0001
  grad_clip = 5.0
  random_state = np.random.RandomState(1999)

  duration_inpt = tf.placeholder(tf.float32,
                                 [None, batch_size, num_duration_features])
  note_inpt = tf.placeholder(tf.float32, [None, batch_size, num_note_features])

  note_target = tf.placeholder(tf.float32,
                               [None, batch_size, num_note_features])
  duration_target = tf.placeholder(tf.float32,
                                   [None, batch_size, num_duration_features])
  init_h1 = tf.placeholder(tf.float32, [batch_size, rnn_dim])

  if share_note_and_target_embeddings:
      name_dur_emb = "dur"
      name_note_emb = "note"
  else:
      name_dur_emb = None
      name_note_emb = None
  duration_embed = Multiembedding(duration_inpt, n_duration_symbols,
                                  duration_embed_dim, random_state,
                                  name=name_dur_emb,
                                  share_all=share_all_embeddings)

  note_embed = Multiembedding(note_inpt, n_note_symbols,
                              note_embed_dim, random_state,
                              name=name_note_emb,
                              share_all=share_all_embeddings)

  scan_inp = tf.concat(2, [duration_embed, note_embed])
  scan_inp_dim = n_notes * duration_embed_dim + n_notes * note_embed_dim


  def step(inp_t, h1_tm1):
      h1_t_proj, h1gate_t_proj = RNNFork([inp_t],
                                         [scan_inp_dim],
                                         h_dim,
                                         random_state,
                                         weight_norm=weight_norm_middle)
      h1_t = RNN(h1_t_proj, h1gate_t_proj,
                 h1_tm1, h_dim, h_dim, random_state)
      return h1_t

  h1_f = scan(step, [scan_inp], [init_h1])
  h1 = h1_f
  final_h1 = ni(h1, -1)

  target_note_embed = Multiembedding(note_target, n_note_symbols, note_embed_dim,
                                     random_state,
                                     name=name_note_emb,
                                     share_all=share_all_embeddings)
  target_note_masked = Automask(target_note_embed, n_notes)
  target_duration_embed = Multiembedding(duration_target, n_duration_symbols,
                                         duration_embed_dim, random_state,
                                         name=name_dur_emb,
                                         share_all=share_all_embeddings)
  target_duration_masked = Automask(target_duration_embed, n_notes)

  costs = []
  note_preds = []
  duration_preds = []
  if share_output_parameters:
      name_note = "note_pred"
      name_dur = "dur_pred"
  else:
      name_note = None
      name_dur = None
  for i in range(n_notes):
      note_pred = Linear([h1[:, :, :h_dim],
                          scan_inp,
                          target_note_masked[i], target_duration_masked[i]],
                         [h_dim,
                          scan_inp_dim,
                          n_notes * note_embed_dim, n_notes * duration_embed_dim],
                         note_out_dims[i], random_state,
                         weight_norm=weight_norm_outputs,
                         name=name_note)
      duration_pred = Linear([h1[:, :, :h_dim],
                              scan_inp,
                              target_note_masked[i],
                              target_duration_masked[i]],
                             [h_dim,
                              scan_inp_dim,
                              n_notes * note_embed_dim,
                              n_notes * duration_embed_dim],
                             duration_out_dims[i],
                             random_state, weight_norm=weight_norm_outputs,
                             name=name_dur)
      n = categorical_crossentropy(softmax(note_pred), note_target[:, :, i])
      d = categorical_crossentropy(softmax(duration_pred),
                                   duration_target[:, :, i])
      cost = n_duration_symbols * tf.reduce_mean(n) + n_note_symbols * tf.reduce_mean(d)
      cost /= (n_duration_symbols + n_note_symbols)
      note_preds.append(note_pred)
      duration_preds.append(duration_pred)
      costs.append(cost)

  # 4 notes pitch and 4 notes duration
  cost = sum(costs) / float(n_notes + n_notes)

  params = tf.trainable_variables()
  grads = tf.gradients(cost, params)
  grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]
  opt = tf.train.AdamOptimizer(learning_rate, use_locking=True)
  updates = opt.apply_gradients(zip(grads, params))


  def _loop(itr, sess, inits=None, do_updates=True):
      i_h1 = np.zeros((batch_size, rnn_dim)).astype("float32")
      duration_mb, note_mb = next(itr)
      X_note_mb = note_mb[:-1]
      y_note_mb = note_mb[1:]
      X_duration_mb = duration_mb[:-1]
      y_duration_mb = duration_mb[1:]
      feed = {note_inpt: X_note_mb,
              note_target: y_note_mb,
              duration_inpt: X_duration_mb,
              duration_target: y_duration_mb,
              init_h1: i_h1}
      if do_updates:
          outs = [cost, final_h1, updates]
          train_loss, h1_l, _ = sess.run(outs, feed)
      else:
          outs = [cost, final_h1]
          train_loss, h1_l = sess.run(outs, feed)
      return train_loss, h1_l


  run_loop(_loop, train_itr, valid_itr,
           n_epochs=num_epochs,
           checkpoint_delay=40,
           checkpoint_every_n_epochs=5)

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
