"""embedding_rnn: vector_rnn for embeddings.

embedding_rnn estends vector_rnn / sketch_rnn to NSynth generation.
"""

import math
import os
import pickle
import random
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework import ops

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '/tmp/embedding_rnn/data',
                       'data directory numpy cpickle of vector file.')
tf.flags.DEFINE_string('log_root', '/tmp/embedding_rnn',
                       'directory to store all dumps.')
tf.flags.DEFINE_string('checkpoint_path', '',
                       'directory to checkpoint to restore.')
tf.flags.DEFINE_string('hparam', 'rnn_size=1000,data_set=train_z.npy',
                       'default model params string to be parsed by HParams.')


def default_hps():
  return tf.contrib.training.HParams(
      num_steps=10000000,
      save_every=1000,
      max_seq_len=125,
      seq_width=16,
      rnn_size=1024,
      num_layers=1,
      model='lstm',
      bidir=1,
      enc_rnn_size=256,
      enc_model='lstm',
      z_size=64,
      context_vec_size=128,
      kl_weight=1.0,
      kl_weight_start=0.01,
      kl_tolerance=0.2,
      batch_size=128,
      grad_clip=1.0,
      num_mixture=5,
      clip_z_with_tanh=0,
      learning_rate=0.001,
      decay_rate=0.9999,
      kl_decay_rate=0.99995,
      min_learning_rate=0.00001,
      use_recurrent_dropout=0,
      recurrent_dropout_prob=0.95,
      use_input_dropout=0,
      input_dropout_prob=0.95,
      use_output_dropout=0,
      output_dropout_prob=0.95,
      hyper_lstm_size=256,
      hyper_embedding_size=64,
      hyper_use_layer_norm=1,
      data_set='train_z.npy',
      pitch_set='trian_pitch.npy',
      velocity_set='trian_velocity.npy',
      unconditional=0,
      is_training=1,
      trial=0,
      overfit=0,
      difference_input=1,
      no_sample_vae=0,
      pca_dim_reduce=0,
      n_pca_components=6)


def get_file_path(file_name):
  return os.path.expanduser(os.path.expandvars(file_name))


class DataLoader(object):
  """Dataloader class.

  DataLoader for embedding_rnn
  """

  def __init__(self,
               hps_model=None,
               eval_hps_model=None,
               sample_hps_model=None,
               random_state=1,
               filename='train_z.npy',
               filename_pitches='train_pitch_one_hot.npy',
               filename_vels='train_vels_one_hot.npy'):

    # set the internal values given parameters in the call
    self.batch_size = hps_model.batch_size
    self.valid_batch_size = eval_hps_model.batch_size
    self.test_batch_size = eval_hps_model.batch_size
    self.sample_batch_size = sample_hps_model.batch_size
    self.random_state = random_state

    # load this huge npy file of embedding data.
    f = get_file_path(os.path.join(FLAGS.data_dir, filename))
    raw_data = np.load(f)

    # load the matching pitch and velocity information
    f_pitches = get_file_path(os.path.join(FLAGS.data_dir, filename_pitches))
    f_vels = get_file_path(os.path.join(FLAGS.data_dir, filename_vels))
    pitches = np.load(f_pitches)
    vels = np.load(f_vels)

    # model the difference of reverse sequence
    # and add zero to also model initial dc.
    if hps_model.difference_input:
      # allocate an array, Number of Samples x 1 x 16
      temp_zero_tensor = np.zeros([raw_data.shape[0], 1, 16])
      # concatenate the raw data with single empty dimension along axis=1
      data = np.concatenate((raw_data, temp_zero_tensor), axis=1)
      # reverse the data along axis=1
      data = data[:, ::-1, :]
      # perform a difference along the second dimension for all samples
      # and over all the channels, now all the data will start at 0 and
      # be the correct length
      data = data[:, 1:, :] - data[:, 0:-1, :]
    else:
      # model the raw data with no difference transform
      data = raw_data

    # Handle the splits of the data into training, validation, test
    # first split the raw data into training and testing sets
    (data_train, data_test, pitches_train, pitches_test, vels_train,
     vels_test) = train_test_split(
         data, pitches, vels, random_state=self.random_state, test_size=0.3)

    # then split the test set into holdout testing and validation
    (data_test, data_valid, pitches_test, pitches_valid, vels_test,
     vels_valid) = train_test_split(
         data_test,
         pitches_test,
         vels_test,
         random_state=self.random_state,
         test_size=0.5)

    # expose data and parameters
    self.seq_length = raw_data.shape[1]  # number of timesteps
    self.seq_width = raw_data.shape[2]  # number of dimensions per timestep
    self.raw_data = raw_data

    ## Modify data to only use the principal components of the data
    # build the PCA model on only the training data
    # pca_dim_reduce=0, n_pca_components=5

    # reduce the sequence width dimensionality, as this is the frame covariance
    # RNN should handle the time dependency
    if hps_model.pca_dim_reduce == 1:
      n_components = hps_model.n_pca_components
      print 'PCA dimensionality reduction starting, n={}'.format(n_components)
      # reshape the stack to 2D
      # do not need to do PCA on all training set
      # only do PCA on the first N samples in the training set
      pca_train_slice_size = 100000
      stack = np.reshape(data_train[:pca_train_slice_size],
                         (pca_train_slice_size * self.seq_length, -1))
      # Fit a PCA on the 2D training data stack
      pca = PCA(
          n_components=n_components, svd_solver='arpack',
          whiten=True).fit(stack)

      # transform the data: reduce dimensionality from
      # [n_samples, seq_length, seq_width] to
      # [n_samples, seq_length, n_components]
      # where n_components < seq_width

      # reduce dim training data, return in model_shape
      data_train = np.reshape(
          pca.transform(
              np.reshape(data_train, (data_train.shape[0] * self.seq_length,
                                      self.seq_width))),
          (data_train.shape[0], self.seq_length, n_components))

      # reduce dim validation data, return in model_shape
      data_valid = np.reshape(
          pca.transform(
              np.reshape(data_valid, (data_valid.shape[0] * self.seq_length,
                                      self.seq_width))),
          (data_valid.shape[0], self.seq_length, n_components))

      # reduce dim testing data, return in model_shape
      data_test = np.reshape(
          pca.transform(
              np.reshape(data_test, (data_test.shape[0] * self.seq_length,
                                     self.seq_width))),
          (data_test.shape[0], self.seq_length, n_components))

      # save PCA model for later
      tf.gfile.MkDir(FLAGS.log_root)
      pkl_file = tf.gfile.Open(FLAGS.log_root + '/pca.pkl', 'a+')
      pickle.dump(pca, pkl_file)

      # update the seq_width of the DataLoader
      self.seq_width = hps_model.n_pca_components
      print 'Dimensionality reduced'

    # train
    self.data = data_train
    self.pitches = pitches_train
    self.vels = vels_train

    # validation
    self.data_valid = data_valid
    self.pitches_valid = pitches_valid
    self.vels_valid = vels_valid

    # testing
    self.data_test = data_test
    self.pitches_test = pitches_test
    self.vels_test = vels_test

    ## add a condition to test overfitting
    # in this configuration the training data is the validation data
    if hps_model.overfit:
      self.data = data_valid
      self.pitches = pitches_valid
      self.vels = vels_valid

    print('training set size: ', len(self.data))
    print('validation set size: ', len(self.data_valid))
    print('testing set size: ', len(self.data_test))

    # calculate and expose number of batches by split
    self.num_batches = int(len(self.data) / self.batch_size)
    self.num_valid_batches = int(len(self.data_valid) / self.valid_batch_size)
    self.num_test_batches = int(len(self.data_test) / self.test_batch_size)

  def revert(self, sample):
    # return data in original format
    # sample is in dimensions self.seq_length, self.seq_width
    return np.cumsum(sample, axis=0)[::-1]

  def random_sample(self):
    # padded front with a zero.
    return random.choice(self.data)

  def random_padded_batch(self, split='train'):
    """Given split type, return padded batch."""

    # training batch size
    batch_size = self.batch_size
    # define the batch size for testing and validation
    if split == 'valid':
      batch_size = self.valid_batch_size
      split_data = self.data_valid
      split_pitches = self.pitches_valid
    elif split == 'test':
      batch_size = self.test_batch_size
      split_data = self.data_test
      split_pitches = self.pitches_test
    else:
      split_data = self.data
      split_pitches = self.pitches

    # padded front with a zero.
    indices = np.random.permutation(range(0, len(split_data)))[0:batch_size]
    temp_zero_tensor = np.zeros([batch_size, 1, self.seq_width])
    data = np.concatenate((temp_zero_tensor, split_data[indices]), axis=1)
    pitches_one_hot = split_pitches[indices]
    return data, pitches_one_hot


def load_dataset(hps_model, load_data=True):
  """Load dataset with hyperparameter.

  Loads dataset from a numpy array and split into training, validation
  and testing sets. Build corresponding hyperparameters for each set.

  Args:
    hps_model: hyperparameters of the training model.
    load_data: (bool) set to False to skip dataset loading.

  Returns:
    data_set: DataLoader for data set.
    hps_model: Training hyperparameters.
    eval_hps_model: Evaluation hyperparameters.
    sample_hps_model: Sampling hyperparameters.

  Raises:
    No errors.
  """
  # setup parallel models for eval
  eval_hps_model = default_hps()
  eval_hps_model.parse(FLAGS.hparam)

  eval_hps_model.parse('max_seq_len=%d' % hps_model.max_seq_len)
  eval_hps_model.parse('use_input_dropout=%d' % 0)
  eval_hps_model.parse('use_recurrent_dropout=%d' % 0)
  eval_hps_model.parse('use_output_dropout=%d' % 0)
  eval_hps_model.parse('is_training=%d' % 0)
  eval_hps_model.parse('batch_size=%d' % hps_model.batch_size)

  # setup parallel model for sampling
  sample_hps_model = default_hps()
  sample_hps_model.parse(FLAGS.hparam)
  sample_hps_model.parse('batch_size=%d' % 1)
  sample_hps_model.parse('max_seq_len=%d' % 1)

  # pick a random state, from 0-4294967295
  # ensures consistent random state for train/test/validation
  # each time the dataset is loaded, splits are different
  random_state = np.random.randint(4294967295)

  # create DataLoader
  print 'creating DataLoader'
  data_set = None
  if load_data:
    data_set = DataLoader(
        hps_model=hps_model,
        eval_hps_model=eval_hps_model,
        sample_hps_model=sample_hps_model,
        random_state=random_state)

  return (data_set, hps_model, eval_hps_model, sample_hps_model)


def binary_cross_entropy_with_logits(logits, targets, name=None):
  """Computes binary cross entropy given `logits`.

  from: https://github.com/hardmaru/cppn-gan-vae-tensorflow/blob/master/ops.py
  For brevity, let `x = logits`, `z = targets`.  The logistic loss is
      loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

  Args:
      logits: A `Tensor` of type `float32` or `float64`.
      targets: A `Tensor` of the same type and shape as `logits`.
      name: A string for passing tensorflow scope.

  Returns:
      Returns the binary cross entropy as a Tensor.
  """
  eps = 1e-6
  with ops.op_scope([logits, targets], name, 'bce_loss') as name:
    logits = ops.convert_to_tensor(logits, name='logits')
    targets = ops.convert_to_tensor(targets, name='targets')
    return tf.reduce_mean(-(logits * tf.log(targets + eps) +
                            (1. - logits) * tf.log(1. - targets + eps)))


def orthogonal(shape):
  """Orthogonal Initializer from https://github.com/OlavHN/bnlstm.

  Args:
    shape: shape of the input tensor

  Returns:
    Returns an orthogonally initialized tensor with given input shape.
  """
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)


def orthogonal_initializer(scale=1.0):

  def _initializer(shape, dtype=tf.float32, partition_info=None):
    tf.logging.info('Partition info: %s', partition_info)
    return tf.constant(orthogonal(shape) * scale, dtype)

  return _initializer


def lstm_identity_initializer(scale=1.0):
  """Initialize LSTM as identity matrix."""

  def _initializer(shape, dtype=tf.float32, partition_info=None):
    tf.logging.info('Partition info: %s', partition_info)
    size = shape[0]
    t = np.zeros(shape)
    t[:, size:size * 2] = np.identity(size) * scale  # gate (j) is identity
    t[:, :size] = orthogonal([size, size])
    t[:, size * 2:size * 3] = orthogonal([size, size])
    t[:, size * 3:] = orthogonal([size, size])
    return tf.constant(t, dtype)

  return _initializer


def lstm_ortho_initializer(scale=1.0):
  """LSTM Orthogonal Initializer.

  Args:
    scale: 1.0, scales the intialized tensor.

  Returns:
    Initializer for LSTM Orthogonal initialization
  """

  def _initializer(shape, dtype=tf.float32, partition_info=None):
    tf.logging.info('Partition info: %s', partition_info)
    size_x = shape[0]
    size_h = shape[1] / 4  # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h]) * scale
    t[:, size_h:size_h * 2] = orthogonal([size_x, size_h]) * scale
    t[:, size_h * 2:size_h * 3] = orthogonal([size_x, size_h]) * scale
    t[:, size_h * 3:] = orthogonal([size_x, size_h]) * scale
    return tf.constant(t, dtype)

  return _initializer


class LSTMCell(tf.contrib.rnn.RNNCell):
  """Vanilla LSTM with ortho initializer.

  recurrent dropout without memory loss
  (https://arxiv.org/abs/1603.05118)

  derived from
  https://github.com/OlavHN/bnlstm
  https://github.com/LeavesBreathe/tensorflow_with_latest_papers

  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.9):
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob

  @property
  def state_size(self):
    return 2 * self.num_units

  @property
  def output_size(self):
    return self.num_units

  def get_output(self, state):
    _, h = tf.split(state, 2, 1)
    return h

  def __call__(self, x, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      c, h = tf.split(state, 2, 1)

      #       h_size = self.num_units
      x_size = x.get_shape().as_list()[1]

      # w_init=lstm_ortho_initializer(1.0)
      # w_init=orthogonal_initializer(1.0)
      # w_init=tf.constant_initializer(0.0)
      # w_init=tf.random_normal_initializer(stddev=0.01)
      w_init = None  # uniform

      h_init = lstm_ortho_initializer(1.0)
      # h_init=tf.constant_initializer(0.0)
      # h_init=tf.random_normal_initializer(stddev=0.01)
      # h_init=None # uniform

      # Keep w_xh and w_hh separate here as well
      # to use different initialization methods
      w_xh = tf.get_variable(
          'w_xh', [x_size, 4 * self.num_units], initializer=w_init)
      w_hh = tf.get_variable(
          'w_hh', [self.num_units, 4 * self.num_units], initializer=h_init)
      bias = tf.get_variable(
          'bias', [4 * self.num_units],
          initializer=tf.constant_initializer(0.0))

      concat = tf.concat([x, h], 1)
      w_full = tf.concat([w_xh, w_hh], 0)
      hidden = tf.matmul(concat, w_full) + bias

      i, j, f, o = tf.split(hidden, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j)

      new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * g
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

      return new_h, tf.concat([new_c, new_h], 1)


class LayerNormLSTMCell(tf.contrib.rnn.RNNCell):
  """Layer-Norm, with Ortho Initialization, Recurrent Dropout, no memory loss.

  https://arxiv.org/abs/1607.06450 - Layer Norm
  https://arxiv.org/abs/1603.05118 - Recurrent Dropout without Memory Loss
  derived from
  https://github.com/OlavHN/bnlstm
  https://github.com/LeavesBreathe/tensorflow_with_latest_papers
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.90):
    """Initialize the Layer Norm LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: bool, use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob

  @property
  def input_size(self):
    return self.num_units

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return 2 * self.num_units

  def get_output(self, state):
    h, _ = tf.split(state, 2, 1)
    return h

  def __call__(self, x, state, timestep=0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      h, c = tf.split(state, 2, 1)

      h_size = self.num_units
      x_size = x.get_shape().as_list()[1]
      batch_size = x.get_shape().as_list()[0]

      # w_init=lstm_ortho_initializer(1.0)
      # w_init=orthogonal_initializer(1.0)
      # w_init=tf.constant_initializer(0.0)
      # w_init=tf.random_normal_initializer(stddev=0.01)
      w_init = None  # unifor

      h_init = lstm_ortho_initializer(1.0)
      # h_init=tf.constant_initializer(0.0)
      # h_init=tf.random_normal_initializer(stddev=0.01)
      # h_init=None # uniform

      w_xh = tf.get_variable(
          'w_xh', [x_size, 4 * self.num_units], initializer=w_init)
      w_hh = tf.get_variable(
          'w_hh', [self.num_units, 4 * self.num_units], initializer=h_init)
      # no bias, since there's a bias thing inside layer norm
      # and we don't wanna double task variables.

      concat = tf.concat([x, h], 1)  # concat for speed.
      w_full = tf.concat([w_xh, w_hh], 0)
      concat = tf.matmul(concat, w_full)  #+ bias live life without garbage.

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      concat = layer_norm_all(concat, batch_size, 4, h_size, 'ln_all')
      i, j, f, o = tf.split(concat, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j)

      new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * g
      new_h = tf.tanh(layer_norm(new_c, h_size, 'ln_c')) * tf.sigmoid(o)

    return new_h, tf.concat([new_h, new_c], 1)


class HyperLSTMCell(tf.contrib.rnn.RNNCell):
  """HyperLSTM, with Ortho, Layer Norm, Recurrent Dropout, no Memory Loss.

  https://arxiv.org/abs/1609.09106
  http://blog.otoro.net/2016/09/28/hyper-networks/
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.90,
               use_layer_norm=False,
               hyper_num_units=128,
               hyper_embedding_size=16,
               hyper_use_recurrent_dropout=False):
    """Initialize the Layer Norm HyperLSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: float, User Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
      use_layer_norm: boolean. (default True)
        Use LayerNorm layers in main LSTM and HyperLSTM cell.
      hyper_num_units: int, number of units in HyperLSTM cell.
        (default is 128, recommend experimenting with 256 for larger tasks)
      hyper_embedding_size: int, size of signals emitted from HyperLSTM cell.
        (default is 4, try larger values)
      hyper_use_recurrent_dropout: boolean. (default False)
        HyperLSTM cell uses recurrent dropout. (Not in Paper.)
        Turn on only if hyper_num_units becomes very large (>= 512)
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.use_layer_norm = use_layer_norm
    self.hyper_num_units = hyper_num_units
    self.hyper_embedding_size = hyper_embedding_size
    self.hyper_use_recurrent_dropout = hyper_use_recurrent_dropout

    self.total_num_units = self.num_units + self.hyper_num_units

    if self.use_layer_norm:
      cell_fn = LayerNormLSTMCell
    else:
      cell_fn = LSTMCell
    self.hyper_cell = cell_fn(
        hyper_num_units,
        use_recurrent_dropout=hyper_use_recurrent_dropout,
        dropout_keep_prob=dropout_keep_prob)

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return 2 * self.total_num_units

  def get_output(self, state):
    total_h, _ = tf.split(state, 2, 1)
    #     total_h, total_c = tf.split(state, 2, 1)
    h = total_h[:, 0:self.num_units]
    # c = total_c[:, 0:self.num_units]
    # hyper_state = tf.concat(1, [total_h[:,self.num_units:],
    # total_c[:,self.num_units:]])
    return h

  def hyper_norm(self, layer, scope='hyper', use_bias=True):
    num_units = self.num_units
    embedding_size = self.hyper_embedding_size

    # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
    init_gamma = 0.10  # cooijmans' da man.
    with tf.variable_scope(scope):
      zw = super_linear(
          self.hyper_output,
          embedding_size,
          init_w='constant',
          weight_start=0.00,
          use_bias=True,
          bias_start=1.0,
          scope='zw')
      alpha = super_linear(
          zw,
          num_units,
          init_w='constant',
          weight_start=init_gamma / embedding_size,
          use_bias=False,
          scope='alpha')
      result = tf.multiply(alpha, layer)

      if use_bias:
        zb = super_linear(
            self.hyper_output,
            embedding_size,
            init_w='gaussian',
            weight_start=0.01,
            use_bias=False,
            bias_start=0.0,
            scope='zb')
        beta = super_linear(
            zb,
            num_units,
            init_w='constant',
            weight_start=0.00,
            use_bias=False,
            scope='beta')
        result += beta
    return result

  def __call__(self, x, state, timestep=0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      total_h, total_c = tf.split(state, 2, 1)
      h = total_h[:, 0:self.num_units]
      c = total_c[:, 0:self.num_units]
      self.hyper_state = tf.concat(
          [total_h[:, self.num_units:], total_c[:, self.num_units:]], 1)

      batch_size = x.get_shape().as_list()[0]
      x_size = x.get_shape().as_list()[1]
      self._input_size = x_size

      # w_init=lstm_ortho_initializer(1.0)
      # w_init=orthogonal_initializer(1.0)
      # w_init=tf.constant_initializer(0.0)
      # w_init=tf.random_normal_initializer(stddev=0.01)
      w_init = None  # uniform

      h_init = lstm_ortho_initializer(1.0)
      # h_init=lstm_identity_initializer(1.0)
      # h_init=tf.constant_initializer(0.0)
      # h_init=tf.random_normal_initializer(stddev=0.01)
      # h_init=None # uniform

      w_xh = tf.get_variable(
          'w_xh', [x_size, 4 * self.num_units], initializer=w_init)
      w_hh = tf.get_variable(
          'w_hh', [self.num_units, 4 * self.num_units], initializer=h_init)
      bias = tf.get_variable(
          'bias', [4 * self.num_units],
          initializer=tf.constant_initializer(0.0))

      # concatenate the input and hidden states for hyperlstm input
      hyper_input = tf.concat([x, h], 1)
      hyper_output, hyper_new_state = self.hyper_cell(hyper_input,
                                                      self.hyper_state)
      self.hyper_output = hyper_output
      self.hyper_state = hyper_new_state

      xh = tf.matmul(x, w_xh)
      hh = tf.matmul(h, w_hh)

      # split Wxh contributions
      ix, jx, fx, ox = tf.split(xh, 4, 1)
      ix = self.hyper_norm(ix, 'hyper_ix', use_bias=False)
      jx = self.hyper_norm(jx, 'hyper_jx', use_bias=False)
      fx = self.hyper_norm(fx, 'hyper_fx', use_bias=False)
      ox = self.hyper_norm(ox, 'hyper_ox', use_bias=False)

      # split Whh contributions
      ih, jh, fh, oh = tf.split(hh, 4, 1)
      ih = self.hyper_norm(ih, 'hyper_ih', use_bias=True)
      jh = self.hyper_norm(jh, 'hyper_jh', use_bias=True)
      fh = self.hyper_norm(fh, 'hyper_fh', use_bias=True)
      oh = self.hyper_norm(oh, 'hyper_oh', use_bias=True)

      # split bias
      ib, jb, fb, ob = tf.split(bias, 4, 0)  # bias is to be broadcasted.

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i = ix + ih + ib
      j = jx + jh + jb
      f = fx + fh + fb
      o = ox + oh + ob

      if self.use_layer_norm:
        concat = tf.concat([i, j, f, o], 1)
        concat = layer_norm_all(concat, batch_size, 4, self.num_units, 'ln_all')
        i, j, f, o = tf.split(concat, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j)

      new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * g
      new_h = tf.tanh(layer_norm(new_c, self.num_units, 'ln_c')) * tf.sigmoid(o)

      hyper_h, hyper_c = tf.split(hyper_new_state, 2, 1)
      new_total_h = tf.concat([new_h, hyper_h], 1)
      new_total_c = tf.concat([new_c, hyper_c], 1)
      new_total_state = tf.concat([new_total_h, new_total_c], 1)
    return new_h, new_total_state


def layer_norm_all(h,
                   batch_size,
                   base,
                   num_units,
                   scope='layer_norm',
                   reuse=False,
                   gamma_start=1.0,
                   epsilon=1e-3,
                   use_bias=True):
  """Layer normalization on all.

  Layer Norm (faster version, but not using defun)
  Performs layer norm on multiple base at once (ie, i, g, j, o for lstm)
  Reshapes h in to perform layer norm in parallel

  Args:
    h: hidden layer
    batch_size: size of the batch.
    base: base number
    num_units: number of units in the layer
    scope: scope name
    reuse: reuse the variables
    gamma_start: starting gamma value
    epsilon: epsilion
    use_bias: use bias (boolean)

  Returns:
    Layer normalized h
  """

  h_reshape = tf.reshape(h, [batch_size, base, num_units])
  mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
  var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
  epsilon = tf.constant(epsilon)
  rstd = tf.rsqrt(var + epsilon)
  h_reshape = (h_reshape - mean) * rstd
  # reshape back to original
  h = tf.reshape(h_reshape, [batch_size, base * num_units])
  with tf.variable_scope(scope):
    if reuse:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable(
        'ln_gamma', [4 * num_units],
        initializer=tf.constant_initializer(gamma_start))
    if use_bias:
      beta = tf.get_variable(
          'ln_beta', [4 * num_units], initializer=tf.constant_initializer(0.0))
  if use_bias:
    return gamma * h + beta
  return gamma * h


def layer_norm(x,
               num_units,
               scope='layer_norm',
               reuse=False,
               gamma_start=1.0,
               epsilon=1e-3,
               use_bias=True):
  """Adds a Layer Normalization layer."""
  axes = [1]
  mean = tf.reduce_mean(x, axes, keep_dims=True)
  x_shifted = x - mean
  var = tf.reduce_mean(tf.square(x_shifted), axes, keep_dims=True)
  inv_std = tf.rsqrt(var + epsilon)
  with tf.variable_scope(scope):
    if reuse:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable(
        'ln_gamma', [num_units],
        initializer=tf.constant_initializer(gamma_start))
    if use_bias:
      beta = tf.get_variable(
          'ln_beta', [num_units], initializer=tf.constant_initializer(0.0))
  output = gamma * (x_shifted) * inv_std
  if use_bias:
    output += beta
  return output


def raw_layer_norm(x, epsilon=1e-3):
  axes = [1]
  mean = tf.reduce_mean(x, axes, keep_dims=True)
  std = tf.sqrt(
      tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)
  output = (x - mean) / (std)
  return output


def super_linear(x,
                 output_size,
                 scope=None,
                 reuse=False,
                 init_w='ortho',
                 weight_start=0.0,
                 use_bias=True,
                 bias_start=0.0,
                 input_size=None):
  # support function doing linear operation.
  # uses ortho initializer defined earlier.
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or 'linear'):
    if reuse:
      tf.get_variable_scope().reuse_variables()

    w_init = None  # uniform
    if input_size is None:
      x_size = shape[1]
    else:
      x_size = input_size

    # h_size = output_size

    if init_w == 'zeros':
      w_init = tf.constant_initializer(0.0)
    elif init_w == 'constant':
      w_init = tf.constant_initializer(weight_start)
    elif init_w == 'gaussian':
      w_init = tf.random_normal_initializer(stddev=weight_start)
    elif init_w == 'ortho':
      w_init = lstm_ortho_initializer(1.0)

    w = tf.get_variable(
        'super_linear_w', [x_size, output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable(
          'super_linear_b', [output_size],
          tf.float32,
          initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)


def reset_graph(sess=None):
  """Reset the graph if the session has started."""
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()


class Model(object):
  """General class for the neural network model."""

  def __init__(self, hps, gpu_mode=True, reuse=False):
    self.hps = hps
    with tf.variable_scope('embedding_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          #           print 'model using cpu'
          self.build_model(hps)
      else:
        #         print "model using gpu"
        self.build_model(hps)

  def encoder(self, batch, sequence_lengths):
    """Encoder for the the input batch."""
    if self.hps.bidir == 1:
      _, last_states = tf.nn.bidirectional_dynamic_rnn(
          self.enc_cell_fw,
          self.enc_cell_bw,
          batch,
          sequence_length=sequence_lengths,
          time_major=False,
          swap_memory=True,
          dtype=tf.float32,
          scope='ENC_RNN')

      # output_fw, output_bw = outputs
      last_state_fw, last_state_bw = last_states

      last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
      last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
      last_h = tf.concat([last_h_fw, last_h_bw], 1)

      # project the last hidden layer to the embedding
      if self.hps.no_sample_vae:
        last_h_proj = super_linear(
            last_h,
            self.hps.z_size,
            input_size=self.hps.enc_rnn_size * 2,
            scope='ENC_RNN_last_h_proj',
            init_w='gaussian',
            weight_start=0.001)
      mu = super_linear(
          last_h,
          self.hps.z_size,
          input_size=self.hps.enc_rnn_size * 2,
          scope='ENC_RNN_mu',
          init_w='gaussian',
          weight_start=0.001)
      presig = super_linear(
          last_h,
          self.hps.z_size,
          input_size=self.hps.enc_rnn_size * 2,
          scope='ENC_RNN_sigma',
          init_w='gaussian',
          weight_start=0.001)
    else:
      # get output and last state
      _, last_state = tf.nn.dynamic_rnn(
          self.enc_cell,
          batch,
          sequence_length=sequence_lengths,
          time_major=False,
          swap_memory=True,
          dtype=tf.float32,
          scope='ENC_RNN')
      last_h = self.enc_cell.get_output(last_state)

      if self.hps.no_sample_vae == 1:
        last_h_proj = super_linear(
            last_h,
            self.hps.z_size,
            input_size=self.hps.enc_rnn_size,
            scope='ENC_RNN_last_h_proj',
            init_w='gaussian',
            weight_start=0.001)
      mu = super_linear(
          last_h,
          self.hps.z_size,
          input_size=self.hps.enc_rnn_size,
          scope='ENC_RNN_mu',
          init_w='gaussian',
          weight_start=0.001)
      presig = super_linear(
          last_h,
          self.hps.z_size,
          input_size=self.hps.enc_rnn_size,
          scope='ENC_RNN_sigma',
          init_w='gaussian',
          weight_start=0.001)

    # return mean, standard deviation, and last hidden layer
    if self.hps.no_sample_vae == 1:
      return mu, presig, last_h_proj
    else:
      return mu, presig, _

  def build_model(self, hps):
    """Assembles the model given the hyperparameter definition."""

    self.num_mixture = hps.num_mixture
    kmix = self.num_mixture  # 5 mixtures
    width = hps.seq_width  # 16 channels
    if hps.pca_dim_reduce == 1:
      width = hps.n_pca_components
    length = self.hps.max_seq_len  # 125 timesteps

    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.hyper_use_layer_norm = False
    if self.hps.hyper_use_layer_norm == 1:
      self.hyper_use_layer_norm = True

    if hps.model == 'lstm':
      cell_fn = LSTMCell
    elif hps.model == 'layer_norm':
      cell_fn = LayerNormLSTMCell
    elif hps.model == 'hyper':
      cell_fn = HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    if hps.enc_model == 'lstm':
      enc_cell_fn = LSTMCell
    elif hps.enc_model == 'layer_norm':
      enc_cell_fn = LayerNormLSTMCell
    elif hps.enc_model == 'hyper':
      enc_cell_fn = HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    if self.hps.use_recurrent_dropout == 0:
      use_recurrent_dropout = False
    else:
      use_recurrent_dropout = True
    use_input_dropout = False if self.hps.use_input_dropout == 0 else True
    use_output_dropout = False if self.hps.use_output_dropout == 0 else True

    if hps.model == 'hyper':
      cell = cell_fn(
          hps.rnn_size,
          use_recurrent_dropout=use_recurrent_dropout,
          dropout_keep_prob=self.hps.recurrent_dropout_prob,
          use_layer_norm=self.hyper_use_layer_norm,
          hyper_num_units=self.hps.hyper_lstm_size,
          hyper_embedding_size=self.hps.hyper_embedding_size,
          hyper_use_recurrent_dropout=False)

    else:
      cell = cell_fn(
          hps.rnn_size,
          use_recurrent_dropout=use_recurrent_dropout,
          dropout_keep_prob=self.hps.recurrent_dropout_prob)

    if hps.unconditional == 0:  # vae mode:
      if hps.enc_model == 'hyper':
        if hps.bidir == 1:
          self.enc_cell_fw = enc_cell_fn(
              hps.enc_rnn_size,
              use_recurrent_dropout=use_recurrent_dropout,
              dropout_keep_prob=self.hps.recurrent_dropout_prob,
              use_layer_norm=self.hyper_use_layer_norm,
              hyper_num_units=self.hps.hyper_lstm_size,
              hyper_embedding_size=self.hps.hyper_embedding_size,
              hyper_use_recurrent_dropout=False)

          self.enc_cell_bw = enc_cell_fn(
              hps.enc_rnn_size,
              use_recurrent_dropout=use_recurrent_dropout,
              dropout_keep_prob=self.hps.recurrent_dropout_prob,
              use_layer_norm=self.hyper_use_layer_norm,
              hyper_num_units=self.hps.hyper_lstm_size,
              hyper_embedding_size=self.hps.hyper_embedding_size,
              hyper_use_recurrent_dropout=False)
        else:
          self.enc_cell = enc_cell_fn(
              hps.enc_rnn_size,
              use_recurrent_dropout=use_recurrent_dropout,
              dropout_keep_prob=self.hps.recurrent_dropout_prob,
              use_layer_norm=self.hyper_use_layer_norm,
              hyper_num_units=self.hps.hyper_lstm_size,
              hyper_embedding_size=self.hps.hyper_embedding_size,
              hyper_use_recurrent_dropout=False)
      else:
        if hps.bidir == 1:
          self.enc_cell_fw = enc_cell_fn(
              hps.enc_rnn_size,
              use_recurrent_dropout=use_recurrent_dropout,
              dropout_keep_prob=self.hps.recurrent_dropout_prob)

          self.enc_cell_bw = enc_cell_fn(
              hps.enc_rnn_size,
              use_recurrent_dropout=use_recurrent_dropout,
              dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
          self.enc_cell = cell_fn(
              hps.enc_rnn_size,
              use_recurrent_dropout=use_recurrent_dropout,
              dropout_keep_prob=self.hps.recurrent_dropout_prob)

    # multi-layer, and dropout:
    # print 'input dropout mode =', use_input_dropout
    # print 'output dropout mode =', use_output_dropout
    # print 'recurrent dropout mode =', use_recurrent_dropout
    if use_input_dropout:
      print('applying dropout to input with keep_prob =',
            self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=self.hps.input_dropout_prob)
    if hps.num_layers > 1:
      print('applying multilayer rnn, num_layers =', hps.num_layers)
      cell = tf.contrib.rnn.MultiRNNCell([cell] * hps.num_layers)
    if use_output_dropout:
      print('applying dropout to output with keep_prob =',
            self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    self.sequence_lengths = length  # assume every sample has same length.
    self.input_data = tf.placeholder(
        dtype=tf.float32,
        shape=[self.hps.batch_size, self.hps.max_seq_len + 1, width])

    self.input_x = self.input_data[:, :self.hps.max_seq_len, :]

    # y is the real-deal.
    self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]

    # start auto-encoder
    # vae mode, conditional generation
    if hps.unconditional == 0:
      sequence_lengths_tensor = tf.constant([length] * self.hps.batch_size)
      if hps.no_sample_vae == 1:
        # use a linear projection of the last hidden layer instead of sampling
        # the mean and sigma from the VAE
        print 'use last hidden layer instead of sampling VAE'
        (self.mean, self.presig, self.last_h_proj) = self.encoder(
            self.output_x, sequence_lengths_tensor)
        # z embeddings are projection from last hidden layer
        self.batch_z = self.last_h_proj

      else:
        print 'vae mode, conditional generation'
        (self.mean, self.presig, _) = self.encoder(self.output_x,
                                                   sequence_lengths_tensor)

        # force sigma > 0
        self.sigma = tf.sqrt(tf.exp(self.presig))
        eps = tf.random_normal(
            (self.hps.batch_size, self.hps.z_size), 0, 1, dtype=tf.float32)
        # sample mean and sigma for batch_z embeddings
        self.batch_z = self.mean + tf.multiply(self.sigma, eps)

      # can use a nonlinear activation to transform batch_z to batch_y
      if hps.clip_z_with_tanh == 1:
        self.batch_y = tf.tanh(self.batch_z)
      else:
        self.batch_y = self.batch_z

      self.pitches_one_hot = tf.placeholder(
          dtype=tf.float32,
          shape=[self.hps.batch_size, self.hps.context_vec_size],
          name='pitches')

      # concatenate batch data and pitches data here
      self.batch_y = tf.concat([self.batch_y, self.pitches_one_hot], axis=1)
      self.kl_cost = -0.5 * tf.reduce_mean(
          1 + self.presig - tf.square(self.mean) - tf.exp(self.presig))
      self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance)
      pre_tile_y = tf.reshape(self.batch_y, [
          self.hps.batch_size, 1, self.hps.z_size + self.hps.context_vec_size
      ])
      overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
      actual_input_x = tf.concat([self.input_x, overlay_x], 2)

      # initial state to decoder RNN with the conditional information
      # based on the super_linear function which needs to be modified to
      # handle multipe layers
      if hps.num_layers == 1:
        self.initial_state = tf.nn.tanh(
            super_linear(
                self.batch_y,
                cell.state_size,
                init_w='gaussian',
                weight_start=0.001,
                input_size=self.hps.z_size + self.hps.context_vec_size))
      else:
        # initial states for the multilayer RNN
        initial_states_temp = []
        for i in range(hps.num_layers):
          initial_state_temp = tf.nn.tanh(
              super_linear(
                  self.batch_y,
                  cell.state_size[0],
                  scope='initial_state_%d' % i,
                  init_w='gaussian',
                  weight_start=0.001,
                  input_size=self.hps.z_size + 128),
              name='initial_state_%d' % i)
          # print('initial_state_temp', initial_state_temp)
          initial_states_temp.append(initial_state_temp)
        self.initial_state = tuple(initial_states_temp)

    else:  # unconditional generation
      self.batch_z = tf.zeros(
          (self.hps.batch_size, self.hps.z_size), dtype=tf.float32)

      # can use a nonlinear activation to transform batch_z to batch_y
      if hps.clip_z_with_tanh == 1:
        self.batch_y = tf.tanh(self.batch_z)
      else:
        self.batch_y = self.batch_z

      self.kl_cost = tf.zeros([], dtype=tf.float32)
      actual_input_x = self.input_x
      self.initial_state = cell.zero_state(
          batch_size=hps.batch_size, dtype=tf.float32)
    # end auto-encoder
    num_outputs = width * kmix * 3

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.rnn_size, num_outputs])
      output_b = tf.get_variable('output_b', [num_outputs])

    output, last_state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=actual_input_x,
        initial_state=self.initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN')

    output = tf.reshape(output, [-1, hps.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    output = tf.reshape(output, [-1, kmix * 3])
    self.final_state = last_state

    log_sqt_two_pi = math.log(math.sqrt(2.0 * math.pi))

    def tf_lognormal(y, mean, logstd):
      return -0.5 * ((y - mean) / tf.exp(logstd))**2 - logstd - log_sqt_two_pi

    def get_lossfunc(logmix, mean, logstd, y):
      v = logmix + tf_lognormal(y, mean, logstd)
      v = tf.reduce_logsumexp(v, 1, keep_dims=True)
      return -tf.reduce_mean(v)

    def get_mdn_coef(output):
      logmix, mean, logstd = tf.split(output, 3, 1)
      logmix -= tf.reduce_logsumexp(logmix, 1, keep_dims=True)
      return logmix, mean, logstd

    # expose the mixture parameters
    out_logmix, out_mean, out_logstd = get_mdn_coef(output)
    self.out_logmix = out_logmix
    self.out_mean = out_mean
    self.out_logstd = out_logstd

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.output_x, [-1, 1])

    # define the loss function using the outputs and the targets
    lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)
    self.r_cost = tf.reduce_mean(lossfunc)

    # if evaluation
    if self.hps.is_training == 0:
      # compute cost as weighted sum of reconstruction and kl cost
      self.cost = self.r_cost + self.kl_cost * self.hps.kl_weight

    # if training
    if self.hps.is_training == 1:
      # set the learning rate
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      # define the optimizer
      optimizer = tf.train.AdamOptimizer(self.lr)
      # set the kl_weight, decays during training
      self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
      # cost is weighted sum of reconstruction and kl cost
      self.cost = self.r_cost + self.kl_cost * self.kl_weight
      # comput the gradients
      gvs = optimizer.compute_gradients(self.cost)
      # clip gradients
      capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip,
                                      self.hps.grad_clip), var)
                    for grad, var in gvs]
      # apply gradients with training step update
      self.train_op = optimizer.apply_gradients(
          capped_gvs, global_step=self.global_step, name='train_step')


def evaluate_model(sess, model, data_set, split='valid'):
  """Evaluates the model on the evaluation dataset."""
  # returns the avg weighted cost, avg reconstruction cost, and avg kl cost
  total_cost = 0.0
  total_r_cost = 0.0
  total_kl_cost = 0.0

  # evaluation happens on different splits of the data
  if split == 'valid':
    num_batches = data_set.num_valid_batches
    random_batch = data_set.random_padded_batch(split='valid')
  elif split == 'test':
    num_batches = data_set.num_test_batches
    random_batch = data_set.random_padded_batch(split='test')

  # evaluate on batches
  print 'evaluating batches: ', num_batches
  for batch in xrange(num_batches):
    batch, pitches = random_batch
    if model.hps.unconditional:
      feed = {model.input_data: batch}
    else:
      feed = {model.input_data: batch, model.pitches_one_hot: pitches}
    (cost, r_cost,
     kl_cost) = sess.run([model.cost, model.r_cost, model.kl_cost], feed)
    total_cost += cost
    total_r_cost += r_cost
    total_kl_cost += kl_cost
  total_cost /= (num_batches)
  total_r_cost /= (num_batches)
  total_kl_cost /= (num_batches)
  return (total_cost, total_r_cost, total_kl_cost)


def save_model(sess, model_save_path):
  saver = tf.train.Saver(tf.global_variables())  # all_variables() replaced
  checkpoint_path = os.path.join(model_save_path, 'embedding_rnn')
  print 'saving model: ', checkpoint_path
  saver.save(sess, checkpoint_path)


def load_model(sess, model_path):
  saver = tf.train.Saver(tf.global_variables())  # all_variables() replaced
  checkpoint_path = model_path
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)

  if ckpt and ckpt.model_checkpoint_path:
    print 'Loading model from checkpoint:', ckpt.model_checkpoint_path
    tf.logging.info('Loading model from checkpoint: %s',
                    ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print 'Skipping evaluation. No checkpoint found.'
    tf.logging.warning('Skipping evaluation. No checkpoint found in: %s',
                       checkpoint_path)


def train(sess, model, eval_model, data_set):
  """Train a model with the given model parameters, flags and data."""
  # setup summary writer
  summary_writer = tf.summary.FileWriter(FLAGS.log_root)

  # count trainable model parameters
  t_vars = tf.trainable_variables()
  count_t_vars = 0
  for var in t_vars:
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars += num_param
    print var.name, var.get_shape(), num_param
  print 'total trainable variables = %d' % (count_t_vars)
  model_summ = tf.summary.Summary()
  model_summ.value.add(
      tag='Num_Trainable_Params', simple_value=float(count_t_vars))
  summary_writer.add_summary(model_summ, 0)
  summary_writer.flush()

  # setup eval stats
  best_valid_cost = 100000000.0
  valid_cost = 0.0

  hps = model.hps

  ## Main training loop
  # start timer and start training
  start = time.time()
  print 'starting training at %d' % start
  for _ in xrange(hps.num_steps):
    step = sess.run(model.global_step)
    curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                          (hps.decay_rate)**step + hps.min_learning_rate)

    # modify the kl_weight on each step
    curr_kl_weight = (hps.kl_weight - (hps.kl_weight - hps.kl_weight_start) *
                      (hps.kl_decay_rate)**step)

    batch, pitches = data_set.random_padded_batch(split='train')

    # model feed depends on conditional generation
    if hps.unconditional:
      feed = {
          model.input_data: batch,
          model.lr: curr_learning_rate,
          model.kl_weight: curr_kl_weight
      }
    else:
      feed = {
          model.input_data: batch,
          model.pitches_one_hot: pitches,
          model.lr: curr_learning_rate,
          model.kl_weight: curr_kl_weight
      }

    (train_cost, r_cost, kl_cost, _, train_step, _) = sess.run([
        model.cost, model.r_cost, model.kl_cost, model.final_state,
        model.global_step, model.train_op
    ], feed)

    if step % 20 == 0 and step > 0:
      end = time.time()
      time_taken = end - start

      cost_summ = tf.summary.Summary()
      cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost))
      reconstr_summ = tf.summary.Summary()
      reconstr_summ.value.add(
          tag='Train_Reconstr_Cost', simple_value=float(r_cost))
      kl_summ = tf.summary.Summary()
      kl_summ.value.add(tag='Train_KL_Cost', simple_value=float(kl_cost))
      lr_summ = tf.summary.Summary()
      lr_summ.value.add(
          tag='Learning_Rate', simple_value=float(curr_learning_rate))
      kl_weight_summ = tf.summary.Summary()
      kl_weight_summ.value.add(
          tag='KL_Weight', simple_value=float(curr_kl_weight))
      time_summ = tf.summary.Summary()
      time_summ.value.add(
          tag='Time_Taken_Train', simple_value=float(time_taken))

      output_log = ('step: %d, lr: %.6f, klw: %0.4f, '
                    'cost: %.4f, recon: %.4f, kl: %.4f, '
                    'train_time_taken: %.4f') % (step, curr_learning_rate,
                                                 curr_kl_weight, train_cost,
                                                 r_cost, kl_cost, time_taken)
      print output_log
      tf.logging.info(output_log + '\n')

      summary_writer.add_summary(cost_summ, train_step)
      summary_writer.add_summary(reconstr_summ, train_step)
      summary_writer.add_summary(kl_summ, train_step)
      summary_writer.add_summary(lr_summ, train_step)
      summary_writer.add_summary(kl_weight_summ, train_step)
      summary_writer.add_summary(time_summ, train_step)
      summary_writer.flush()
      start = time.time()

    if step % hps.save_every == 0 and step > 0:

      print('step: %d, evaluating model') % (step)

      # evaluate the model on the validation set
      (valid_cost, valid_r_cost, valid_kl_cost) = evaluate_model(
          sess, eval_model, data_set, split='valid')

      end = time.time()
      time_taken_valid = end - start
      start = time.time()

      # write summaries for tensorboard: VALIDATION
      valid_cost_summ = tf.summary.Summary()
      valid_cost_summ.value.add(
          tag='Valid_Cost', simple_value=float(valid_cost))
      valid_reconstr_summ = tf.summary.Summary()
      valid_reconstr_summ.value.add(
          tag='Valid_Reconstr_Cost', simple_value=float(valid_r_cost))
      valid_kl_summ = tf.summary.Summary()
      valid_kl_summ.value.add(
          tag='Valid_KL_Cost', simple_value=float(valid_kl_cost))
      valid_time_summ = tf.summary.Summary()
      valid_time_summ.value.add(
          tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

      output_log = ('best_valid_cost: %0.4f, valid_cost: %.4f, '
                    'valid_recon: %.4f, valid_kl: %.4f, '
                    'valid_time_taken: %.4f') % (
                        min(best_valid_cost, valid_cost), valid_cost,
                        valid_r_cost, valid_kl_cost, time_taken_valid)
      print output_log
      tf.logging.info(output_log + '\n')

      summary_writer.add_summary(valid_cost_summ, train_step)
      summary_writer.add_summary(valid_reconstr_summ, train_step)
      summary_writer.add_summary(valid_kl_summ, train_step)
      summary_writer.add_summary(valid_time_summ, train_step)
      summary_writer.flush()

      # early stopping, only update/eval the model if it is the best so far
      if valid_cost < best_valid_cost:
        best_valid_cost = valid_cost

        save_model(sess, FLAGS.log_root)

        end = time.time()
        time_taken_save = end - start
        start = time.time()

        print 'time_taken_save: %.4f' % (time_taken_save)

        best_valid_cost_summ = tf.summary.Summary()
        best_valid_cost_summ.value.add(
            tag='Best_Valid_Cost', simple_value=float(best_valid_cost))

        summary_writer.add_summary(best_valid_cost_summ, train_step)
        summary_writer.flush()

        # evaluate the model on the test set
        (eval_cost, eval_r_cost, eval_kl_cost) = evaluate_model(
            sess, eval_model, data_set, split='test')

        end = time.time()
        time_taken_eval = end - start
        start = time.time()

        # write summaries for tensorboard: EVALUATION
        eval_cost_summ = tf.summary.Summary()
        eval_cost_summ.value.add(tag='Eval_Cost', simple_value=float(eval_cost))
        eval_reconstr_summ = tf.summary.Summary()
        eval_reconstr_summ.value.add(
            tag='Eval_Reconstr_Cost', simple_value=float(eval_r_cost))
        eval_kl_summ = tf.summary.Summary()
        eval_kl_summ.value.add(
            tag='Eval_KL_Cost', simple_value=float(eval_kl_cost))
        eval_time_summ = tf.summary.Summary()
        eval_time_summ.value.add(
            tag='Time_Taken_Eval', simple_value=float(time_taken_eval))

        output_log = ('eval_cost: %.4f, eval_recon: %.4f, eval_kl: %.4f, '
                      'eval_time_taken: %.4f') % (eval_cost, eval_r_cost,
                                                  eval_kl_cost, time_taken_eval)
        print output_log
        tf.logging.info(output_log + '\n')

        summary_writer.add_summary(eval_cost_summ, train_step)
        summary_writer.add_summary(eval_reconstr_summ, train_step)
        summary_writer.add_summary(eval_kl_summ, train_step)
        summary_writer.add_summary(eval_time_summ, train_step)
        summary_writer.flush()


def main(_):

  np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

  print 'embedding_rnn'

  print 'hyper params:'
  hps_model = default_hps()
  hps_model.parse(FLAGS.hparam)
  print hps_model.values()

  print 'loading data files'
  [data_set, hps_model, eval_hps_model, _] = load_dataset(hps_model)

  print 'reset graph'
  reset_graph()

  print 'build training model'
  model = Model(hps_model)
  print 'build evaluation model'
  eval_model = Model(eval_hps_model, reuse=True)

  # start the session and start training
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  if FLAGS.checkpoint_path:
    load_model(sess, FLAGS.checkpoint_path)
  train(sess, model, eval_model, data_set)


if __name__ == '__main__':
  tf.app.run(main)
