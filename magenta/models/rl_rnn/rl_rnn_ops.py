"""Helper functions to support the MelodyQNetwork and MelodyRNN classes."""

import os
import random

from magenta.lib import tf_lib

import numpy as np
import tensorflow as tf

LSTM_STATE_NAME = 'lstm'

# Number of output note classes. This is a property of the dataset.
NUM_CLASSES = 38
BATCH_SIZE = 128
INITIAL_MIDI_VALUE = 48
NUM_SPECIAL_EVENTS = 2
NOTE_OFF = -1
NO_EVENT = -2


def default_hparams():
  """Generates the default hparams used to train a large basic rnn."""
  return tf_lib.HParams(use_dynamic_rnn=True,
                        batch_size=BATCH_SIZE,
                        lr=0.0002,
                        l2_reg=2.5e-5,
                        clip_norm=5,
                        initial_learning_rate=0.5,
                        decay_steps=1000,
                        decay_rate=0.85,
                        rnn_layer_sizes=[2500],
                        skip_first_n_losses=8,
                        one_hot_length=NUM_CLASSES,
                        exponentially_decay_learning_rate=True)


def small_model_hparams():
  """Generates the hparams used to train a small basic rnn."""
  return tf_lib.HParams(use_dynamic_rnn=True,
                        batch_size=BATCH_SIZE,
                        lr=0.0002,
                        l2_reg=2.5e-5,
                        clip_norm=5,
                        initial_learning_rate=0.5,
                        decay_steps=1000,
                        decay_rate=0.85,
                        rnn_layer_sizes=[100],
                        skip_first_n_losses=32,
                        one_hot_length=NUM_CLASSES,
                        exponentially_decay_learning_rate=True)


def default_dqn_hparams():
  return tf_lib.HParams(random_action_probability=0.1,
                        store_every_nth=1,
                        train_every_nth=5,
                        minibatch_size=32,
                        discount_rate=0.95,
                        max_experience=100000,
                        target_network_update_rate=0.01)


def autocorrelate(signal, lag=1):
  """Gives the correlation coefficient for the signal's correlation with itself.

  Args:
    signal: The signal on which to compute the autocorrelation. Can be a list.
    lag: The offset at which to correlate the signal with itself. E.g. if lag
      is 1, will compute the correlation between the signal and itself 1 beat
      later.
  Returns:
    Correlation coefficient.
  """
  n = len(signal)
  x = np.asarray(signal) - np.mean(signal)
  c0 = np.var(signal)

  return (x[lag:] * x[:n - lag]).sum() / float(n) / c0


def sample_softmax(softmax):
  """Samples a note from an array of softmax probabilities.

  Tries to do this with numpy, which requires that the probabilities add to 1.0
  with extreme precision. If this fails, uses a manual implementation.

  Args:
    softmax: An array of probabilities.
  Returns:
    The index of the note that was chosen/sampled.
  """
  try:
    sample = np.argmax(np.random.multinomial(1, pvals=softmax))
    return sample
  except:  # pylint: disable=bare-except
    r = random.uniform(0, np.sum(softmax))
    upto = 0
    for i in range(len(softmax)):
      if upto + softmax[i] >= r:
        return i
      upto += softmax[i]
    logging.warn("Error! sample softmax function shouldn't get here")
    return len(softmax) - 1


def decoder(event_list, transpose_amount):
  return [e - NUM_SPECIAL_EVENTS if e < NUM_SPECIAL_EVENTS else
          e + INITIAL_MIDI_VALUE - transpose_amount for e in event_list]


def make_onehot(int_list, one_hot_length):
  """Convert each int to a one-hot vector.

  A one-hot vector is 0 everywhere except at the index equal to the
  encoded value.

  For example: 5 as a one-hot vector is [0, 0, 0, 0, 0, 1, 0, 0, 0, ...]

  Args:
    int_list: A list of ints, each of which will get a one-hot encoding.
    one_hot_length: The length of the one-hot vector to be created.
  Returns:
    A list of one-hot encodings of the ints.
  """
  return [[1.0 if j == i else 0.0 for j in xrange(one_hot_length)]
          for i in int_list]


def get_inner_scope(scope_str):
  """Takes a tensorflow scope string and finds the inner scope.

  Inner scope is one layer more internal.

  Args:
    scope_str: Tensorflow variable scope string.
  Returns:
    Scope string with outer scope stripped off.
  """
  idx = scope_str.find('/')
  return scope_str[idx + 1:]


def trim_variable_postfixes(scope_str):
  """Trims any extra numbers added to a tensorflow scope string.

  Necessary to align variables in graph and checkpoint

  Args:
    scope_str: Tensorflow variable scope string.
  Returns:
    Scope string with extra numbers trimmed off.
  """
  idx = scope_str.find(':')
  return scope_str[:idx]


def get_variable_names(graph, scope):
  """Finds all the variable names in a graph that begin with a given scope.

  Args:
    graph: A tensorflow graph.
    scope: A string scope.
  Returns:
    List of variables.
  """
  with graph.as_default():
    return [v.name for v in tf.all_variables() if v.name.startswith(scope)]


def get_next_file_name(directory, prefix, extension):
  """Finds next available filename in directory by appending numbers to prefix.

  E.g. If prefix is 'myfile', extenstion is '.png', and 'directory' already
  contains 'myfile.png' and 'myfile1.png', this function will return
  'myfile2.png'.

  Args:
    directory: Path to the relevant directory.
    prefix: The filename prefix to use.
    extension: String extension of the file, eg. '.mid'.
  Returns:
    String name of the file.
  """
  name = directory + '/' + prefix + '.' + extension
  i = 0
  while os.path.isfile(name):
    i += 1
    name = directory + '/' + prefix + str(i) + '.' + extension
  return name

def make_cell(hparams, state_is_tuple=False):
  cells = []
  for num_units in hparams.rnn_layer_sizes:
    cell = tf.nn.rnn_cell.LSTMCell(
        num_units, state_is_tuple=state_is_tuple)
    #cell = tf.nn.rnn_cell.DropoutWrapper(
    #    cell, output_keep_prob=hparams.dropout_keep_prob)
    cells.append(cell)

  cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if hparams.attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, hparams.attn_length, state_is_tuple=state_is_tuple)

  return cell
