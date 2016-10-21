import numpy as np
import tensorflow as tf
from tfkdllib import Multiembedding, GRUFork, GRU, Linear, Automask
from tfkdllib import ni, scan
from tfkdllib import softmax, categorical_crossentropy
from tfkdllib import tfrecord_duration_and_pitch_iterator
from tfkdllib import LSTM, LSTMFork

class Graph(object):
  def __init__(self, examples):
    num_epochs = 500
    self.batch_size = 32
    # ~1 step per seconds
    # 30 steps ~= 30 seconds
    sequence_length = 30

    train_itr = tfrecord_duration_and_pitch_iterator(examples,
                                                     self.batch_size,
                                                     stop_index=.9,
                                                     sequence_length=sequence_length)

    duration_mb, note_mb = next(train_itr)
    train_itr.reset()

    self.valid_itr = tfrecord_duration_and_pitch_iterator(examples,
                                                     self.batch_size,
                                                     start_index=.9,
                                                     sequence_length=sequence_length)

    num_note_features = note_mb.shape[-1]
    num_duration_features = duration_mb.shape[-1]
    n_note_symbols = len(train_itr.note_classes)
    n_duration_symbols = len(train_itr.time_classes)
    self.n_notes = train_itr.simultaneous_notes
    note_embed_dim = 20
    duration_embed_dim = 4
    n_dim = 64
    h_dim = n_dim
    note_out_dims = self.n_notes * [n_note_symbols]
    duration_out_dims = self.n_notes * [n_duration_symbols]

    rnn_type = "lstm"
    weight_norm_middle = False
    weight_norm_outputs = False

    share_note_and_target_embeddings = True
    share_all_embeddings = False
    share_output_parameters = False

    if rnn_type == "lstm":
        RNNFork = LSTMFork
        RNN = LSTM
        self.rnn_dim = 2 * h_dim
    elif rnn_type == "gru":
        RNNFork = GRUFork
        RNN = GRU
        self.rnn_dim = h_dim
    else:
        raise ValueError("Unknown rnn_type %s" % rnn_type)

    learning_rate = .0001
    grad_clip = 5.0
    random_state = np.random.RandomState(1999)

    self.duration_inpt = tf.placeholder(tf.float32,
                                   [None, self.batch_size, num_duration_features])
    self.note_inpt = tf.placeholder(tf.float32, [None, self.batch_size, num_note_features])

    self.note_target = tf.placeholder(tf.float32,
                                 [None, self.batch_size, num_note_features])
    self.duration_target = tf.placeholder(tf.float32,
                                     [None, self.batch_size, num_duration_features])
    self.init_h1 = tf.placeholder(tf.float32, [self.batch_size, self.rnn_dim])

    if share_note_and_target_embeddings:
        name_dur_emb = "dur"
        name_note_emb = "note"
    else:
        name_dur_emb = None
        name_note_emb = None
    duration_embed = Multiembedding(self.duration_inpt, n_duration_symbols,
                                    duration_embed_dim, random_state,
                                    name=name_dur_emb,
                                    share_all=share_all_embeddings)

    note_embed = Multiembedding(self.note_inpt, n_note_symbols,
                                note_embed_dim, random_state,
                                name=name_note_emb,
                                share_all=share_all_embeddings)

    scan_inp = tf.concat(2, [duration_embed, note_embed])
    scan_inp_dim = self.n_notes * duration_embed_dim + self.n_notes * note_embed_dim


    def step(inp_t, h1_tm1):
        h1_t_proj, h1gate_t_proj = RNNFork([inp_t],
                                           [scan_inp_dim],
                                           h_dim,
                                           random_state,
                                           weight_norm=weight_norm_middle)
        h1_t = RNN(h1_t_proj, h1gate_t_proj,
                   h1_tm1, h_dim, h_dim, random_state)
        return h1_t

    h1_f = scan(step, [scan_inp], [self.init_h1])
    h1 = h1_f
    self.final_h1 = ni(h1, -1)

    target_note_embed = Multiembedding(self.note_target, n_note_symbols, note_embed_dim,
                                       random_state,
                                       name=name_note_emb,
                                       share_all=share_all_embeddings)
    target_note_masked = Automask(target_note_embed, self.n_notes)
    target_duration_embed = Multiembedding(self.duration_target, n_duration_symbols,
                                           duration_embed_dim, random_state,
                                           name=name_dur_emb,
                                           share_all=share_all_embeddings)
    target_duration_masked = Automask(target_duration_embed, self.n_notes)

    costs = []
    self.note_preds = []
    self.duration_preds = []
    if share_output_parameters:
        name_note = "note_pred"
        name_dur = "dur_pred"
    else:
        name_note = None
        name_dur = None
    for i in range(self.n_notes):
        note_pred = Linear([h1[:, :, :h_dim],
                            scan_inp,
                            target_note_masked[i], target_duration_masked[i]],
                           [h_dim,
                            scan_inp_dim,
                            self.n_notes * note_embed_dim, self.n_notes * duration_embed_dim],
                           note_out_dims[i], random_state,
                           weight_norm=weight_norm_outputs,
                           name=name_note)
        duration_pred = Linear([h1[:, :, :h_dim],
                                scan_inp,
                                target_note_masked[i],
                                target_duration_masked[i]],
                               [h_dim,
                                scan_inp_dim,
                                self.n_notes * note_embed_dim,
                                self.n_notes * duration_embed_dim],
                               duration_out_dims[i],
                               random_state, weight_norm=weight_norm_outputs,
                               name=name_dur)
        n = categorical_crossentropy(softmax(note_pred), self.note_target[:, :, i])
        d = categorical_crossentropy(softmax(duration_pred),
                                     self.duration_target[:, :, i])
        cost = n_duration_symbols * tf.reduce_mean(n) + n_note_symbols * tf.reduce_mean(d)
        cost /= (n_duration_symbols + n_note_symbols)
        self.note_preds.append(note_pred)
        self.duration_preds.append(duration_pred)
        costs.append(cost)

    # 4 notes pitch and 4 notes duration
    cost = sum(costs) / float(self.n_notes + self.n_notes)

    params = tf.trainable_variables()
    grads = tf.gradients(cost, params)
    grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]
    opt = tf.train.AdamOptimizer(learning_rate, use_locking=True)
    updates = opt.apply_gradients(zip(grads, params))
