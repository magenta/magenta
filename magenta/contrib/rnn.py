# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Forked classes and functions from `tf.contrib.rnn`."""
import tensorflow.compat.v1 as tf
import tf_slim

from tensorflow.python.ops import gen_nn_ops  # pylint:disable=g-direct-tensorflow-import

rnn_cell = tf.nn.rnn_cell

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"


def _hasattr(obj, attr_name):
  try:
    getattr(obj, attr_name)
  except AttributeError:
    return False
  else:
    return True


def _is_sequence(maybe_seq):
  return isinstance(maybe_seq, (list, tuple, dict))


def assert_like_rnncell(cell_name, cell):
  """Raises a TypeError if cell is not like an RNNCell.

  NOTE: Do not rely on the error message (in particular in tests) which can be
  subject to change to increase readability. Use
  ASSERT_LIKE_RNNCELL_ERROR_REGEXP.

  Args:
    cell_name: A string to give a meaningful error referencing to the name of
      the functionargument.
    cell: The object which should behave like an RNNCell.

  Raises:
    TypeError: A human-friendly exception.
  """
  conditions = [
      _hasattr(cell, "output_size"),
      _hasattr(cell, "state_size"),
      _hasattr(cell, "get_initial_state") or _hasattr(cell, "zero_state"),
      callable(cell),
  ]
  errors = [
      "'output_size' property is missing", "'state_size' property is missing",
      "either 'zero_state' or 'get_initial_state' method is required",
      "is not callable"
  ]

  if not all(conditions):

    errors = [error for error, cond in zip(errors, conditions) if not cond]
    raise TypeError("The argument {!r} ({}) is not an RNNCell: {}.".format(
        cell_name, cell, ", ".join(errors)))


class _Linear(object):  # pylint:disable=g-classes-have-attributes
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch, n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (_is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not _is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1] is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += int(shape[1])

    dtype = [a.dtype for a in args][0]

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
      self._weights = tf.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with tf.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
          self._biases = tf.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = tf.matmul(args[0], self._weights)
    else:
      # Explicitly creating a one for a minor performance improvement.
      one = tf.constant(1, dtype=tf.int32)
      res = tf.matmul(tf.concat(args, one), self._weights)
    if self._build_bias:
      res = tf.nn.bias_add(res, self._biases)
    return res


class InputProjectionWrapper(rnn_cell.RNNCell):
  """Operator adding an input projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the projection on this batch-concatenated sequence, then split it.
  """

  def __init__(self,
               cell,
               num_proj,
               activation=None,
               input_size=None,
               reuse=None):
    """Create a cell with input projection.

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      num_proj: Python integer.  The dimension to project to.
      activation: (optional) an optional activation function.
      input_size: Deprecated and unused.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    super(InputProjectionWrapper, self).__init__(_reuse=reuse)
    if input_size is not None:
      tf.logging.warn("%s: The input_size parameter is deprecated.", self)
    assert_like_rnncell("cell", cell)
    self._cell = cell
    self._num_proj = num_proj
    self._activation = activation
    self._linear = None

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def call(self, inputs, state):
    """Run the input projection and then the cell."""
    # Default scope: "InputProjectionWrapper"
    if self._linear is None:
      self._linear = _Linear(inputs, self._num_proj, True)
    projected = self._linear(inputs)
    if self._activation:
      projected = self._activation(projected)
    return self._cell(projected, state)


class AttentionCellWrapper(rnn_cell.RNNCell):
  """Basic attention cell wrapper.

  Implementation based on https://arxiv.org/abs/1601.06733.
  """

  def __init__(self,
               cell,
               attn_length,
               attn_size=None,
               attn_vec_size=None,
               input_size=None,
               state_is_tuple=True,
               reuse=None):
    """Create a cell with attention.

    Args:
      cell: an RNNCell, an attention is added to it.
      attn_length: integer, the size of an attention window.
      attn_size: integer, the size of an attention vector. Equal to
          cell.output_size by default.
      attn_vec_size: integer, the number of convolutional features calculated
          on attention state and a size of the hidden layer built from
          base cell state. Equal attn_size to by default.
      input_size: integer, the size of a hidden linear layer,
          built from inputs and attention. Derived from the input tensor
          by default.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  By default (False), the states are all
        concatenated along the column axis.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if cell returns a state tuple but the flag
          `state_is_tuple` is `False` or if attn_length is zero or less.
    """
    super(AttentionCellWrapper, self).__init__(_reuse=reuse)
    assert_like_rnncell("cell", cell)
    if _is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError(
          "Cell returns tuple of states, but the flag "
          "state_is_tuple is not set. State size is: %s" % str(cell.state_size))
    if attn_length <= 0:
      raise ValueError(
          "attn_length should be greater than zero, got %s" % str(attn_length))
    if not state_is_tuple:
      tf.logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if attn_size is None:
      attn_size = cell.output_size
    if attn_vec_size is None:
      attn_vec_size = attn_size
    self._state_is_tuple = state_is_tuple
    self._cell = cell
    self._attn_vec_size = attn_vec_size
    self._input_size = input_size
    self._attn_size = attn_size
    self._attn_length = attn_length
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

  @property
  def state_size(self):
    size = (self._cell.state_size, self._attn_size,
            self._attn_size * self._attn_length)
    if self._state_is_tuple:
      return size
    else:
      return sum(list(size))

  @property
  def output_size(self):
    return self._attn_size

  def call(self, inputs, state):
    """Long short-term memory cell with attention (LSTMA)."""
    if self._state_is_tuple:
      state, attns, attn_states = state
    else:
      states = state
      state = tf.slice(states, [0, 0], [-1, self._cell.state_size])
      attns = tf.slice(states, [0, self._cell.state_size],
                       [-1, self._attn_size])
      attn_states = tf.slice(
          states, [0, self._cell.state_size + self._attn_size],
          [-1, self._attn_size * self._attn_length])
    attn_states = tf.reshape(attn_states,
                             [-1, self._attn_length, self._attn_size])
    input_size = self._input_size
    if input_size is None:
      input_size = inputs.get_shape().as_list()[1]
    if self._linear1 is None:
      self._linear1 = _Linear([inputs, attns], input_size, True)
    inputs = self._linear1([inputs, attns])
    cell_output, new_state = self._cell(inputs, state)
    if self._state_is_tuple:
      new_state_cat = tf.concat(tf.nest.flatten(new_state), 1)
    else:
      new_state_cat = new_state
    new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
    with tf.variable_scope("attn_output_projection"):
      if self._linear2 is None:
        self._linear2 = _Linear([cell_output, new_attns], self._attn_size, True)
      output = self._linear2([cell_output, new_attns])
    new_attn_states = tf.concat(
        [new_attn_states, tf.expand_dims(output, 1)], 1)
    new_attn_states = tf.reshape(
        new_attn_states, [-1, self._attn_length * self._attn_size])
    new_state = (new_state, new_attns, new_attn_states)
    if not self._state_is_tuple:
      new_state = tf.concat(list(new_state), 1)
    return output, new_state

  def _attention(self, query, attn_states):
    conv2d = tf.nn.conv2d
    reduce_sum = tf.math.reduce_sum
    softmax = tf.nn.softmax
    tanh = tf.math.tanh

    with tf.variable_scope("attention"):
      k = tf.get_variable("attn_w",
                          [1, 1, self._attn_size, self._attn_vec_size])
      v = tf.get_variable("attn_v", [self._attn_vec_size])
      hidden = tf.reshape(attn_states,
                          [-1, self._attn_length, 1, self._attn_size])
      hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
      if self._linear3 is None:
        self._linear3 = _Linear(query, self._attn_vec_size, True)
      y = self._linear3(query)
      y = tf.reshape(y, [-1, 1, 1, self._attn_vec_size])
      s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
      a = softmax(s)
      d = reduce_sum(
          tf.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
      new_attns = tf.reshape(d, [-1, self._attn_size])
      new_attn_states = tf.slice(attn_states, [0, 1, 0], [-1, -1, -1])
      return new_attns, new_attn_states


def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    time_major=False,
                                    scope=None,
                                    swap_memory=False):
  """Creates a dynamic bidirectional recurrent neural network.

  Stacks several bidirectional rnn layers. The combined forward and backward
  layer outputs are used as input of the next layer. tf.bidirectional_rnn
  does not allow to share forward and backward information between layers.
  The input_size of the first forward and backward cells must match.
  The initial state for both directions is zero and no intermediate states
  are returned.

  Args:
    cells_fw: List of instances of RNNCell, one per layer,
      to be used for forward direction.
    cells_bw: List of instances of RNNCell, one per layer,
      to be used for backward direction.
    inputs: The RNN inputs. this must be a tensor of shape:
      `[batch_size, max_time, ...]`, or a nested tuple of such elements.
    initial_states_fw: (optional) A list of the initial states (one per layer)
      for the forward RNN.
      Each tensor must has an appropriate type and shape
      `[batch_size, cell_fw.state_size]`.
    initial_states_bw: (optional) Same as for `initial_states_fw`, but using
      the corresponding properties of `cells_bw`.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    time_major: The shape format of the inputs and outputs Tensors. If true,
      these Tensors must be shaped [max_time, batch_size, depth]. If false,
      these Tensors must be shaped [batch_size, max_time, depth]. Using
      time_major = True is a bit more efficient because it avoids transposes at
      the beginning and end of the RNN calculation. However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to None.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.

  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs: Output `Tensor` shaped:
        `[batch_size, max_time, layers_output]`. Where layers_output
        are depth-concatenated forward and backward outputs.
      output_states_fw is the final states, one tensor per layer,
        of the forward rnn.
      output_states_bw is the final states, one tensor per layer,
        of the backward rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is `None`.
  """
  if not cells_fw:
    raise ValueError("Must specify at least one fw cell for BidirectionalRNN.")
  if not cells_bw:
    raise ValueError("Must specify at least one bw cell for BidirectionalRNN.")
  if not isinstance(cells_fw, list):
    raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
  if not isinstance(cells_bw, list):
    raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
  if len(cells_fw) != len(cells_bw):
    raise ValueError("Forward and Backward cells must have the same depth.")
  if (initial_states_fw is not None and
      (not isinstance(initial_states_fw, list) or
       len(initial_states_fw) != len(cells_fw))):
    raise ValueError(
        "initial_states_fw must be a list of state tensors (one per layer).")
  if (initial_states_bw is not None and
      (not isinstance(initial_states_bw, list) or
       len(initial_states_bw) != len(cells_bw))):
    raise ValueError(
        "initial_states_bw must be a list of state tensors (one per layer).")

  states_fw = []
  states_bw = []
  prev_layer = inputs

  with tf.variable_scope(scope or "stack_bidirectional_rnn"):
    for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
      initial_state_fw = None
      initial_state_bw = None
      if initial_states_fw:
        initial_state_fw = initial_states_fw[i]
      if initial_states_bw:
        initial_state_bw = initial_states_bw[i]

      with tf.variable_scope("cell_%d" % i):
        outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            prev_layer,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=sequence_length,
            parallel_iterations=parallel_iterations,
            dtype=dtype,
            swap_memory=swap_memory,
            time_major=time_major)
        # Concat the outputs to create the new input.
        prev_layer = tf.concat(outputs, 2)
      states_fw.append(state_fw)
      states_bw.append(state_bw)

  return prev_layer, tuple(states_fw), tuple(states_bw)


def _lstm_block_cell(x,
                     cs_prev,
                     h_prev,
                     w,
                     b,
                     wci=None,
                     wcf=None,
                     wco=None,
                     forget_bias=None,
                     cell_clip=None,
                     use_peephole=None,
                     name=None):
  r"""Computes the LSTM cell forward propagation for 1 time step.

  This implementation uses 1 weight matrix and 1 bias vector, and there's an
  optional peephole connection.

  This kernel op implements the following mathematical equations:

  ```python
  xh = [x, h_prev]
  [i, ci, f, o] = xh * w + b
  f = f + forget_bias

  if not use_peephole:
    wci = wcf = wco = 0

  i = sigmoid(cs_prev * wci + i)
  f = sigmoid(cs_prev * wcf + f)
  ci = tanh(ci)

  cs = ci .* i + cs_prev .* f
  cs = clip(cs, cell_clip)

  o = sigmoid(cs * wco + o)
  co = tanh(cs)
  h = co .* o
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      The input to the LSTM cell, shape (batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the cell state at previous time step.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Output of the previous cell at previous time step.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `-1` (no clipping).
      Value to clip the 'cs' value to. Disable by setting to negative value.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).
    i: A `Tensor`. Has the same type as `x`. The input gate.
    cs: A `Tensor`. Has the same type as `x`. The cell state before the tanh.
    f: A `Tensor`. Has the same type as `x`. The forget gate.
    o: A `Tensor`. Has the same type as `x`. The output gate.
    ci: A `Tensor`. Has the same type as `x`. The cell input.
    co: A `Tensor`. Has the same type as `x`. The cell after the tanh.
    h: A `Tensor`. Has the same type as `x`. The output h vector.

  Raises:
    ValueError: If cell_size is None.
  """
  if wci is None:
    cell_size = int(cs_prev.shape.with_rank(2)[1])
    if cell_size is None:
      raise ValueError("cell_size from `cs_prev` should not be None.")
    wci = tf.constant(0, dtype=tf.float32, shape=[cell_size])
    wcf = wci
    wco = wci

  return tf.raw_ops.LSTMBlockCell(
      x=x,
      cs_prev=cs_prev,
      h_prev=h_prev,
      w=w,
      wci=wci,
      wcf=wcf,
      wco=wco,
      b=b,
      forget_bias=forget_bias,
      cell_clip=cell_clip if cell_clip is not None else -1,
      use_peephole=use_peephole,
      name=name)


class LayerRNNCell(rnn_cell.RNNCell):
  """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.compat.v1.get_variable`.  The
  underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.compat.v1.get_variable`.
  """

  def __call__(self, inputs, state, *args, scope=None, **kwargs):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple with
        shapes `[batch_size, s] for s in self.state_size`.
      *args: Additional positional arguments.
      scope: optional cell scope.
      **kwargs: Additional keyword arguments.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    return tf.layers.Layer.__call__(
        self, inputs, state, scope=scope, *args, **kwargs)


@tf.RegisterGradient("LSTMBlockCell")
def _LSTMBlockCellGrad(op, *grad):  # pylint:disable=invalid-name,g-wrong-blank-lines
  """Gradient for LSTMBlockCell."""
  (x, cs_prev, h_prev, w, wci, wcf, wco, b) = op.inputs
  (i, cs, f, o, ci, co, _) = op.outputs
  (_, cs_grad, _, _, _, _, h_grad) = grad

  batch_size = x.get_shape().with_rank(2).dims[0].value
  if batch_size is None:
    batch_size = -1
  input_size = x.get_shape().with_rank(2).dims[1].value
  if input_size is None:
    raise ValueError("input_size from `x` should not be None.")
  cell_size = cs_prev.get_shape().with_rank(2).dims[1].value
  if cell_size is None:
    raise ValueError("cell_size from `cs_prev` should not be None.")

  (cs_prev_grad, dgates, wci_grad, wcf_grad,
   wco_grad) = tf.raw_ops.LSTMBlockCellGrad(
       x=x,
       cs_prev=cs_prev,
       h_prev=h_prev,
       w=w,
       wci=wci,
       wcf=wcf,
       wco=wco,
       b=b,
       i=i,
       cs=cs,
       f=f,
       o=o,
       ci=ci,
       co=co,
       cs_grad=cs_grad,
       h_grad=h_grad,
       use_peephole=op.get_attr("use_peephole"))

  # Backprop from dgates to xh.
  xh_grad = tf.matmul(dgates, w, transpose_b=True)

  x_grad = tf.slice(xh_grad, (0, 0), (batch_size, input_size))
  x_grad.get_shape().merge_with(x.get_shape())

  h_prev_grad = tf.slice(xh_grad, (0, input_size), (batch_size, cell_size))
  h_prev_grad.get_shape().merge_with(h_prev.get_shape())

  # Backprop from dgates to w.
  xh = tf.concat([x, h_prev], 1)
  w_grad = tf.matmul(xh, dgates, transpose_a=True)
  w_grad.get_shape().merge_with(w.get_shape())

  # Backprop from dgates to b.
  b_grad = gen_nn_ops.bias_add_grad(dgates)
  b_grad.get_shape().merge_with(b.get_shape())

  return (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
          wco_grad, b_grad)


class LSTMBlockCell(LayerRNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add `forget_bias` (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  Unlike `rnn_cell.LSTMCell`, this is a monolithic op and should be much
  faster.  The weight and bias matrices should be compatible as long as the
  variable scope matches.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               cell_clip=None,
               use_peephole=False,
               dtype=None,
               reuse=None,
               name="lstm_cell"):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      cell_clip: An optional `float`. Defaults to `-1` (no clipping).
      use_peephole: Whether to use peephole connections or not.
      dtype: the variable dtype of this layer. Default to tf.float32.
      reuse: (optional) boolean describing whether to reuse variables in an
        existing scope.  If not `True`, and the existing scope already has the
        given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.  By default this is "lstm_cell", for variable-name compatibility
        with `tf.compat.v1.nn.rnn_cell.LSTMCell`.

      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMBlockCell instead.
    """
    super(LSTMBlockCell, self).__init__(_reuse=reuse, dtype=dtype, name=name)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._use_peephole = use_peephole
    self._cell_clip = cell_clip if cell_clip is not None else -1
    self._names = {
        "W": "kernel",
        "b": "bias",
        "wci": "w_i_diag",
        "wcf": "w_f_diag",
        "wco": "w_o_diag",
        "scope": "lstm_cell"
    }
    # Inputs must be 2-dimensional.
    self.input_spec = tf.layers.InputSpec(ndim=2)

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if not inputs_shape[1]:
      raise ValueError(
          "Expecting inputs_shape[1] to be set: %s" % str(inputs_shape))
    input_size = int(inputs_shape[1])
    self._kernel = self.add_variable(
        self._names["W"], [input_size + self._num_units, self._num_units * 4])
    self._bias = self.add_variable(
        self._names["b"], [self._num_units * 4],
        initializer=tf.constant_initializer(0.0))
    if self._use_peephole:
      self._w_i_diag = self.add_variable(self._names["wci"], [self._num_units])
      self._w_f_diag = self.add_variable(self._names["wcf"], [self._num_units])
      self._w_o_diag = self.add_variable(self._names["wco"], [self._num_units])

    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM)."""
    if len(state) != 2:
      raise ValueError("Expecting state to be a tuple with length 2.")

    if self._use_peephole:
      wci = self._w_i_diag
      wcf = self._w_f_diag
      wco = self._w_o_diag
    else:
      wci = wcf = wco = tf.zeros([self._num_units], dtype=self.dtype)

    (cs_prev, h_prev) = state
    (_, cs, _, _, _, _, h) = _lstm_block_cell(
        inputs,
        cs_prev,
        h_prev,
        self._kernel,
        self._bias,
        wci=wci,
        wcf=wcf,
        wco=wco,
        forget_bias=self._forget_bias,
        cell_clip=self._cell_clip,
        use_peephole=self._use_peephole)

    new_state = rnn_cell.LSTMStateTuple(cs, h)
    return h, new_state


class LayerNormBasicLSTMCell(rnn_cell.RNNCell):
  """LSTM unit with layer normalization and recurrent dropout.

  This class adds layer normalization and recurrent dropout to a
  basic LSTM unit. Layer normalization implementation is based on:

    https://arxiv.org/abs/1607.06450.

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  and is applied before the internal nonlinearities.
  Recurrent dropout is base on:

    https://arxiv.org/abs/1603.05118

  "Recurrent Dropout without Memory Loss"
  Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               input_size=None,
               activation=tf.math.tanh,
               layer_norm=True,
               norm_gain=1.0,
               norm_shift=0.0,
               dropout_keep_prob=1.0,
               dropout_prob_seed=None,
               reuse=None):
    """Initializes the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LayerNormBasicLSTMCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      tf.logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._norm_gain = norm_gain
    self._norm_shift = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope, dtype=tf.float32):
    shape = inp.get_shape()[-1:]
    gamma_init = tf.constant_initializer(self._norm_gain)
    beta_init = tf.constant_initializer(self._norm_shift)
    with tf.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      tf.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
      tf.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
    normalized = tf_slim.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    dtype = args.dtype
    weights = tf.get_variable("kernel", [proj_size, out_size], dtype=dtype)
    out = tf.matmul(args, weights)
    if not self._layer_norm:
      bias = tf.get_variable("bias", [out_size], dtype=dtype)
      out = tf.nn.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    args = tf.concat([inputs, h], 1)
    concat = self._linear(args)
    dtype = args.dtype

    i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
    if self._layer_norm:
      i = self._norm(i, "input", dtype=dtype)
      j = self._norm(j, "transform", dtype=dtype)
      f = self._norm(f, "forget", dtype=dtype)
      o = self._norm(o, "output", dtype=dtype)

    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = tf.nn.dropout(g, self._keep_prob, seed=self._seed)

    new_c = (
        c * tf.math.sigmoid(f + self._forget_bias) + tf.math.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state", dtype=dtype)
    new_h = self._activation(new_c) * tf.math.sigmoid(o)

    new_state = rnn_cell.LSTMStateTuple(new_c, new_h)
    return new_h, new_state
