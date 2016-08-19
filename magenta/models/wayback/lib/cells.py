"""Recurrent transitions in Tensorflow.
"""
import numpy as np
import tensorflow as tf

import magenta.models.wayback.lib.tfutil as tfutil


class BaseCell(tf.nn.rnn_cell.RNNCell):
  """RNNCell-compatible cell base class.

  Instances of this class can be used wherever RNNCell objects can be used.

  NOTE: this base class requires that all subclasses represent state as a
  sequence of Tensors.
  """

  @property
  def state_placeholders(self):
    return [tf.placeholder(dtype=tf.float32, shape=[None, size])
            for size in self.state_size]

  def initial_state(self, batch_size):
    return [np.zeros([batch_size, size])
            for size in self.state_size]

  def __call__(self, inputs, state, scope=None):
    if not isinstance(inputs, (list, tuple)):
      inputs = [inputs]
    state = self.transition(inputs, state, scope=scope)
    output = self.get_output(state)
    return output, state

  def transition(self, inputs, state, scope=None):
    """Update recurrent state.

    Args:
      inputs: list of input Tensors.
      state: list of state Tensors.
      scope: VariableScope for the created subgraph.

    Returns:
      List of updated state Tensors.
    """
    raise NotImplementedError()

  def get_output(self, state):
    """Get output state.

    Args:
      state: list of state Tensors.

    Returns:
      Output state Tensor.
    """
    raise NotImplementedError()


class LSTM(BaseCell):
  """Long Short-Term Memory recurrent transition."""

  def __init__(self, num_units, forget_bias=1.0,
               activation=tf.nn.tanh, use_bn=False, scope=None):
    """Initialize an LSTM object.

    Args:
      num_units: number of hidden units
      forget_bias: bias to add to the forget gate
      activation: activation function on input and output
      use_bn: whether to use Recurrent Batch Normalization
      scope: default scope name of cell variables
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.activation = activation
    self.use_bn = use_bn
    self.scope = scope if scope is not None else "lstm-%x" % id(self)

  @property
  def state_size(self):
    return 2 * [self.num_units]

  @property
  def output_size(self):
    return self.num_units

  def get_output(self, state):
    return state[1]

  def transition(self, inputs, state, scope=None):
    with tf.variable_scope(scope or self.scope):
      c, h = state
      total_input = tfutil.project_terms(
          [h] + inputs, output_dim=4 * self.num_units,
          use_bn=self.use_bn, scope="ijfo")
      i, j, f, o = tf.split(1, 4, total_input)
      f += self.forget_bias

      new_c = (tf.nn.sigmoid(f) * c +
               tf.nn.sigmoid(i) * self.activation(j))
      output_c = new_c
      if self.use_bn:
        output_c = tfutil.batch_normalize(output_c, scope="c")
      new_h = tf.nn.sigmoid(o) * self.activation(output_c)
    return new_c, new_h


class GRU(BaseCell):
  """Gated Recurrent Unit recurrent transition."""

  def __init__(self, num_units, forget_bias=1,
               activation=tf.nn.tanh, use_bn=False,
               scope=None):
    """Initialize a GRU object.

    Args:
      num_units: number of hidden units
      forget_bias: bias to subtract from the reset gate
      activation: activation function on input and output
      use_bn: whether to use Recurrent Batch Normalization
      scope: default scope name of cell variables
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.activation = activation
    self.use_bn = use_bn
    self.scope = scope if scope is not None else "gru-%x" % id(self)

  @property
  def state_size(self):
    return [self.num_units]

  @property
  def output_size(self):
    return self.num_units

  def get_output(self, state):
    return state[0]

  def transition(self, inputs, state, scope=None):
    with tf.variable_scope(scope or self.scope):
      h, = state
      r = tfutil.project_terms([h] + inputs, output_dim=self.num_units,
                               use_bn=self.use_bn, scope="r")
      rh = tf.nn.sigmoid(r - self.forget_bias) * h
      g, z = tf.split(1, 2, tfutil.project_terms(
          [rh] + inputs,
          output_dim=2 * self.num_units,
          use_bn=self.use_bn, scope="gz"))
      new_h = ((1 - tf.nn.sigmoid(z)) * h +
               tf.nn.sigmoid(z) * self.activation(g))
      if self.use_bn:
        new_h = tfutil.batch_normalize(new_h, scope="h")
    return [new_h]


class RNN(BaseCell):
  """Simple recurrent transition."""

  def __init__(self, num_units, activation=tf.nn.tanh, use_bn=False,
               scope=None):
    """Initialize an RNN object.

    Args:
      num_units: number of hidden units
      activation: activation function
      use_bn: whether to use Recurrent Batch Normalization
      scope: default scope name of cell variables
    """
    self.num_units = num_units
    self.activation = activation
    self.use_bn = use_bn
    self.scope = scope if scope is not None else "rnn-%x" % id(self)

  @property
  def state_size(self):
    return [self.num_units]

  @property
  def output_size(self):
    return self.num_units

  def get_output(self, state):
    return state[0]

  def transition(self, inputs, state, scope=None):
    with tf.variable_scope(scope or self.scope):
      h, = state
      g = tfutil.project_terms([h] + inputs, output_dim=self.num_units,
                               use_bn=self.use_bn)
      new_h = self.activation(g)
    return [new_h]
