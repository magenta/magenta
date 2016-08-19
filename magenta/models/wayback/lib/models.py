"""Teacher-forcing sequence models."""
import functools as ft
import numpy as np
import tensorflow as tf

import magenta.models.wayback.lib.cells as cells
from magenta.models.wayback.lib.namespace import Namespace as NS
import magenta.models.wayback.lib.tfutil as tfutil
import magenta.models.wayback.lib.util as util


def construct(hp):
  """Construct a model according to given hyperparameters.

  Args:
    hp: model hyperparameters.

  Returns:
    An instance of `BaseModel`.
  """
  activation = dict(tanh=tf.nn.tanh,
                    identity=lambda x: x,
                    elu=tf.nn.elu)[hp.activation]
  cell_implementation = dict(lstm=cells.LSTM,
                             gru=cells.GRU,
                             rnn=cells.RNN)[hp.cell]
  cells_ = [cell_implementation(layer_size, use_bn=hp.use_bn,
                                activation=activation, scope="cell%i" % i)
            for i, layer_size in enumerate(hp.layer_sizes)]
  model_implementation = dict(stack=Stack, wayback=Wayback)[hp.layout]
  model = model_implementation(cells_=cells_, hp=hp)
  assert hp.segment_length >= model.period
  return model


class BaseModel(object):
  """Base class for sequence models.

  The symbolic state for a model is passed around in the form of a Namespace
  tree. This allows arbitrary compositions of models without assumptions on
  the form of their internal state.
  """

  def __init__(self, hp):
    """Initialize a BaseModel instance.

    Args:
      hp: model hyperparameters.
    """
    self.hp = hp

  @property
  def period(self):
    """Recurrent period.

    Recurrent models are periodic, but some (e.g. Wayback) take multiple time
    steps to complete a cycle. The period is the number of steps taken to
    complete such a cycle.

    Returns:
      The model's period.

    """
    return 1

  def state_placeholders(self):
    """Get the Tensorflow placeholders for the model's states.

    Returns:
      A Namespace tree containing the placeholders.
    """
    return NS.Copy(self._state_placeholders)

  def initial_state(self, batch_size):
    """Get initial values for the model's states.

    Args:
      batch_size: the batch size.

    Returns:
      A Namespace tree containing the values.
    """
    raise NotImplementedError()

  def get_output(self, state):
    """Get model output from the model's states.

    Args:
      state: the model state.

    Returns:
      The model's output.
    """
    raise NotImplementedError()

  def feed_dict(self, state):
    """Construct a feed dict for the model's states.

    Args:
      state: the model state.

    Returns:
      A feed dict mapping each of the model's placeholders to the corresponding
      numerical value in `state`.
    """
    return util.odict(NS.FlatZip([self.state_placeholders(), state]))

  def __call__(self, inputs, state, context=None):
    """Perform a step of the model.

    Args:
      inputs: a `Tensor` denoting the batch of input vectors.
      state: a Namespace tree containing the model's symbolic state.
      context: a `Tensor` denoting context, e.g. for conditioning.

    Returns:
      A tuple of two values: the output and the updated symbolic state.
    """
    raise NotImplementedError()

  # pylint: disable=g-doc-return-or-yield,g-doc-args
  def _make_sequence_graph(self, **kwargs):
    """See the module-level `_make_sequence_graph`."""
    if kwargs.get("model_state", None) is None:
      kwargs["model_state"] = self.state_placeholders()
    def transition(input_, state, context=None):
      state = self(input_, state, context=context)
      h = self.get_output(state)
      return h, state
    return _make_sequence_graph(transition=transition, **kwargs)
  # pylint: enable=g-doc-return-or-yield,g-doc-args

  def make_training_graph(self, x, length=None, context=None,
                          model_state=None):
    """Make a graph to train the model by teacher-forcing.

    `x` is processed in chunks of size determined by the hyperparameter
    `chunk_size`. At step `i`, the model receives the `i`th nonoverlapping
    chunk as input, and its output is used to predict the `i + 1`th chunk.

    The last chunk is not processed, as there would be no further chunk
    available to compare against and compute loss. To ensure all data is
    processed during TBPTT, segments `x` fed into successive computations
    of the graph should overlap by `chunk_size`.

    Args:
      x: Sequence of integer (categorical) inputs, shaped [time, batch].
      length: Optional length of sequence. Inferred from `x` if possible.
      context: Optional Tensor denoting context, shaped [batch, ?].
      model_state: Initial state of the model.

    Returns:
      Namespace containing relevant symbolic variables.
    """
    return self._make_sequence_graph(x=x, model_state=model_state,
                                     length=length, context=context,
                                     back_prop=True, hp=self.hp)

  def make_evaluation_graph(self, x, length=None, context=None,
                            model_state=None):
    """Make a graph to evaluate the model.

    `x` is processed in chunks of size determined by the hyperparameter
    `chunk_size`. At step `i`, the model receives the `i`th nonoverlapping
    chunk as input, and its output is used to predict the `i + 1`th chunk.

    The last chunk is not processed, as there would be no further chunk
    available to compare against and compute loss. To ensure all data is
    processed, segments `x` fed into successive computations of the graph should
    overlap by `chunk_size`.

    Args:
      x: Sequence of integer (categorical) inputs, shaped [time, batch].
      length: Optional length of sequence. Inferred from `x` if possible.
      context: Optional Tensor denoting context, shaped [batch, ?].
      model_state: Initial state of the model.

    Returns:
      Namespace containing relevant symbolic variables.
    """
    return self._make_sequence_graph(x=x, model_state=model_state,
                                     length=length, context=context, hp=self.hp)

  def make_sampling_graph(self, initial_xchunk, length, context=None,
                          model_state=None, temperature=1.0):
    """Make a graph to sample from the model.

    The graph generates a sequence `xhat` in chunks of size determined by the
    hyperparameter `chunk_size`. At the first step, the model receives
    `initial_xchunk` as input, and generates a chunk to follow it. The generated
    chunk is used as input during the next time step. This process is repeated
    until a sequence of the desired length has been generated.

    Args:
      initial_xchunk: Initial model input, shaped [chunk_size, batch].
      length: Desired length of generated sequence.
      context: Optional Tensor denoting context, shaped [batch, ?].
      model_state: Initial state of the model.
      temperature: Softmax temperature to use for sampling.

    Returns:
      Namespace containing relevant symbolic variables.
    """
    return self._make_sequence_graph(initial_xchunk=initial_xchunk,
                                     model_state=model_state, length=length,
                                     context=context, temperature=temperature,
                                     hp=self.hp)


class Stack(BaseModel):
  """A model that consists of a stack of cells.
  """

  def __init__(self, cells_, hp):
    """Initialize a `Stack` instance.

    Args:
      cells_: recurrent transition cells, from bottom to top.
      hp: model hyperparameters.
    """
    super(Stack, self).__init__(hp)
    self.cells = list(cells_)
    self._state_placeholders = NS(
        cells=[cell.state_placeholders for cell in self.cells])

  def initial_state(self, batch_size):
    return NS(cells=[cell.initial_state(batch_size) for cell in self.cells])

  def get_output(self, state):
    return self.cells[-1].get_output(state.cells[-1])

  def __call__(self, x, state, context=None):
    state = NS.Copy(state)
    for i, _ in enumerate(self.cells):
      cell_inputs = []
      if i == 0:
        cell_inputs.append(x)
      if context is not None and i == len(self.cells) - 1:
        cell_inputs.append(context)
      if self.hp.vskip:
        # feed in state of all other layers
        cell_inputs.extend(self.cells[j].get_output(state.cells[j])
                           for j in range(len(self.cells)) if j != i)
      else:
        # feed in state of layer below
        if i > 0:
          cell_inputs.append(self.cells[i - 1].get_output(state.cells[i - 1]))
      state.cells[i] = self.cells[i].transition(cell_inputs, state.cells[i],
                                                scope="cell%i" % i)
    return state


class Wayback(BaseModel):
  """The Wayback machine.

  Essentially a stack of cells, with upper layers moving more slowly than
  lower ones.
  """

  def __init__(self, cells_, hp):
    """Initialize a `Wayback` instance.

    The following hyperparameters are specific to this model:
      periods: update interval of each layer, from top to bottom. As layer 0
          always runs at every step, periods[0] gives the number of steps
          of layer 0 before layer 1 is updated. periods[-1] gives the
          number of steps to run at the highest layer before the model
          should be considered to have completed a cycle.
      unroll_layer_count: number of upper layers to unroll. Unrolling allows
          for gradient truncation on the levels below.
      carry: whether to carry over each cell's state from one cycle to the next
          or break the chain and compute new initial states based on the state
          of the cell above.

    Args:
      cells_: recurrent transition cells, from top to bottom.
      hp: model hyperparameters.

    Raises:
      ValueError: If the number of cells and the number of periods differ.
    """
    super(Wayback, self).__init__(hp)

    if len(cells_) != len(self.hp.periods):
      raise ValueError("must specify one period for each cell")
    self.cells = list(cells_)

    cutoff = len(cells_) - self.hp.unroll_layer_count
    self.inner_indices = list(range(cutoff))
    self.outer_indices = list(range(cutoff, len(cells_)))
    self.inner_slice = slice(cutoff)
    self.outer_slice = slice(cutoff, len(cells_))

    self._state_placeholders = NS(
        time=tf.placeholder(dtype=tf.int32, name="time"),
        cells=[cell.state_placeholders for cell in self.cells])

  @property
  def period(self):
    return int(np.prod(self.hp.periods))

  def initial_state(self, batch_size):
    return NS(time=0,
              cells=[cell.initial_state(batch_size) for cell in self.cells])

  def get_output(self, state):
    return self.cells[0].get_output(state.cells[0])

  def __call__(self, x, state, context=None):
    # construct the usual graph without unrolling
    state = NS.Copy(state)
    state.cells = Wayback.transition(state.time, state.cells, self.cells,
                                     below=x, above=context, hp=self.hp,
                                     symbolic=True)
    state.time += 1
    state.time %= self.period
    return state

  @staticmethod
  def transition(time, cell_states, cells_, subset=None,
                 below=None, above=None, hp=None, symbolic=True):
    """Perform one Wayback transition.

    This function updates `cell_states` according to the wayback connection
    pattern. Note that although `subset` selects a subset of cells to update,
    this function will reach outside that subset to properly condition the
    states within and potentially to disconnect gradient on cells below those in
    `subset`.

    Args:
      time: model time as kept by model state. Must be a Python int if
          `symbolic` is true.
      cell_states: list of cell states as kept by model state.
      cells_: list of cell instances.
      subset: indices of cells to update
      below: Tensor context from below.
      above: Tensor context from above.
      hp: model hyperparameters.
      symbolic: whether the transition occurs inside a dynamic loop or not.

    Returns:
      Updated cell states. Updating `time` is the caller's responsibility.
    """
    def _is_due(i):
      countdown = time % np.prod(hp.periods[:i])
      return tf.equal(countdown, 0) if symbolic else countdown == 0

    if not symbolic:
      # Make a pass over all cells to disconnect gradient on the states of those
      # that just completed a cycle. They should be considered constant for both
      # the edge going up to the layer above and the edge going rightward for
      # the next step of the same layer.
      for i in range(len(cells_)):
        if i != 0 and _is_due(i):
          cell_states[i - 1] = list(map(tf.stop_gradient, cell_states[i - 1]))

    subset = list(range(len(cells_))) if subset is None else subset
    for i in reversed(sorted(subset)):
      is_top = i == len(cells_) - 1
      is_bottom = i == 0

      cell_inputs = []
      if is_bottom and below is not None:
        cell_inputs.append(below)
      if is_top and above is not None:
        cell_inputs.append(above)

      if hp.vskip:
        # feed in states of all other layers
        cell_inputs.extend(cells_[j].get_output(cell_states[j])
                           for j in range(len(cells_)) if j != i)
      else:
        # feed in state of layers below and above
        if not is_bottom:
          cell_inputs.append(cells_[i - 1].get_output(cell_states[i - 1]))
        if not is_top:
          cell_inputs.append(cells_[i + 1].get_output(cell_states[i + 1]))

      # NOTE: Branch functions passed to `cond` don't get called until way after
      # we're out of this loop. That means we need to be careful not to pass in
      # a closure with a loop variable.

      cell_state = cell_states[i]
      carry_context = (above if is_top else
                       cells_[i + 1].get_output(cell_states[i + 1]))
      if not hp.carry and carry_context is not None:
        # start every cycle with a new initial state determined from state above
        def _reinit_cell(cell, context, scope):
          return [tfutil.layer([context], output_dim=size, use_bn=hp.use_bn,
                               scope="%s_%i" % (scope, j))
                  for j, size in enumerate(cell.state_size)]
        reinit_cell = ft.partial(_reinit_cell, cell=cells_[i],
                                 context=carry_context, scope="reinit_%i" % i)
        preserve_cell = util.constantly(cell_state)

        # reinitialize cell[i] if cell above was just updated
        if symbolic:
          cell_state = tfutil.cond(_is_due(i + 1),
                                   reinit_cell, preserve_cell,
                                   prototype=cell_state)
        else:
          if _is_due(i + 1):
            cell_state = reinit_cell()

      update_cell = ft.partial(cells_[i].transition, cell_inputs, cell_state)
      preserve_cell = util.constantly(cell_state)

      if symbolic:
        if is_bottom:
          # skip the cond; bottom layer updates each step
          cell_states[i] = update_cell()
        else:
          cell_states[i] = tfutil.cond(_is_due(i), update_cell, preserve_cell,
                                       prototype=cell_state)
      else:
        if _is_due(i):
          cell_states[i] = update_cell()

    return cell_states

  def _make_sequence_graph(self, **kwargs):
    """Create a (partially unrolled) sequence graph.

    Where possible, this method calls `BaseModel._make_sequence_graph` to
    construct a simple graph with a single while loop.

    If `back_prop` is true and the model is configured for partial unrolling,
    this method dispatches to `Wayback._make_sequence_graph_with_unroll`. In
    that case, `length` must be an int.

    Args:
      **kwargs: passed onto `Wayback._make_sequence_graph_with_unroll` or
                `Wayback._make_sequence_graph`.

    Returns:
      A Namespace containing relevant symbolic variables.
    """
    if kwargs.get("back_prop", False) and self.outer_indices:
      return self._make_sequence_graph_with_unroll(**kwargs)
    else:
      return super(Wayback, self)._make_sequence_graph(**kwargs)

  def _make_sequence_graph_with_unroll(self, model_state=None, x=None,
                                       initial_xchunk=None, context=None,
                                       length=None, temperature=1.0, hp=None,
                                       back_prop=False):
    """Create a sequence graph by unrolling upper layers.

    This method is similar to `_make_sequence_graph`, except that `length` must
    be provided. The resulting graph behaves in the same way as that constructed
    by `_make_sequence_graph`, except that the upper layers are outside of the
    while loop and so the gradient can actually be truncated between runs of
    lower layers.

    If `x` is given, the graph processes the sequence `x` in chunks of size
    determined by the hyperparameter `chunk_size`.  At step `i`, the model
    receives the `i`th nonoverlapping chunk as input, and its output is used to
    predict the `i + 1`th chunk.

    The last chunk is not processed, as there would be no further chunk
    available to compare against and compute loss. To ensure all data is
    processed during TBPTT, segments `x` fed into successive computations
    of the graph should overlap by `chunk_size`.

    If `x` is not given, `initial_xchunk` must be given as the first input
    to the model.  Further chunks are constructed from the model's predictions.

    Args:
      model_state: initial state of the model.
      x: Sequence of integer (categorical) inputs. Not needed if sampling.
          Axes [time, batch].
      initial_xchunk: When sampling, x is not given; initial_xchunk specifies
          the input x[:chunk_size] to the first timestep.
      context: a `Tensor` denoting context, e.g. for conditioning.
          Axes [batch, features].
      length: Optional length of sequence. Inferred from `x` if possible.
      temperature: Softmax temperature to use for sampling.
      hp: Model hyperparameters.
      back_prop: Whether the graph will be backpropagated through.

    Raises:
      ValueError: if `length` is not an int.

    Returns:
      Namespace containing relevant symbolic variables.
    """
    if length is None or not isinstance(length, int):
      raise ValueError("For partial unrolling, length must be known at graph"
                       " construction time.")

    if model_state is None:
      model_state = self.state_placeholders()

    state = NS(model=model_state, inner_initial_xchunk=initial_xchunk,
               xhats=[], losses=[], errors=[])

    # i suspect ugly gradient biases may occur if gradients are truncated
    # somewhere halfway through the cycle. ensure we start at a cycle boundary.
    chunk_size = hp.chunk_size
    outer_alignment_assertion = tf.Assert(
        tf.equal(state.model.time, 0),
        [state.model.time],
        name="outer_alignment_assertion")
    state.model.time = tf.with_dependencies(
        [outer_alignment_assertion], state.model.time)
    # ensure we end at a cycle boundary too.
    assert (length - chunk_size) % (self.period * chunk_size) == 0

    inner_period = int(np.prod(hp.periods[self.inner_slice]))
    outer_step_count = length // (chunk_size * inner_period)
    for outer_time in range(outer_step_count):
      is_first_iteration = outer_time == 0
      is_last_iteration = outer_time == outer_step_count - 1

      if not is_first_iteration:
        tf.get_variable_scope().reuse_variables()

      # update outer layers (wrap in seq scope to be consistent with the fully
      # symbolic version of this graph)
      with tf.variable_scope("seq"):
        state.model.cells = Wayback.transition(
            outer_time * inner_period, state.model.cells, self.cells,
            below=None, above=context, subset=self.outer_indices, hp=hp,
            symbolic=False)

      # run inner layers on subsequence
      if x is None:
        inner_x = None
      else:
        start = inner_period * chunk_size * outer_time
        stop = inner_period * chunk_size * (outer_time + 1) + chunk_size
        inner_x = x[start:stop, :]

      # grab a copy of the outer states. they will not be updated in the inner
      # loop, so we can put back the copy after the inner loop completes.
      # this avoids the gradient truncation due to calling `while_loop` with
      # `back_prop=False`.
      outer_cell_states = NS.Copy(state.model.cells[self.outer_slice])

      # pylint: disable=missing-docstring
      def _inner_transition(input_, state, context=None):
        assert not context
        state.cells = Wayback.transition(
            state.time, state.cells, self.cells, below=input_, above=None,
            subset=self.inner_indices, hp=hp, symbolic=True)
        state.time += 1
        state.time %= self.period
        h = self.get_output(state)
        return h, state
      # pylint: enable=missing-docstring

      inner_ts = _make_sequence_graph(
          transition=_inner_transition, model_state=state.model,
          x=inner_x, initial_xchunk=state.inner_initial_xchunk,
          temperature=temperature, hp=hp,
          # backprop through inner loop only on last iteration
          back_prop=back_prop and is_last_iteration)

      state.model = inner_ts.final_state.model
      state.inner_initial_xchunk = (inner_ts.final_xchunk if x is not None
                                    else inner_ts.final_xhatchunk)
      state.final_xhatchunk = inner_ts.final_xhatchunk
      if x is not None:
        state.final_xchunk = inner_ts.final_xchunk
        state.losses.append(inner_ts.loss)
        state.errors.append(inner_ts.error)
      state.xhats.append(inner_ts.xhat)

      # double check alignment to be safe
      inner_alignment_assertion = tf.Assert(
          tf.equal(state.model.time % inner_period, 0),
          [state.model.time, tf.shape(inner_x)],
          name="inner_alignment_assertion")
      state.model.time = tf.with_dependencies(
          [inner_alignment_assertion], state.model.time)

      # restore static outer states
      state.model.cells[self.outer_slice] = outer_cell_states

    ts = NS()
    ts.xhat = tf.concat(0, state.xhats)
    ts.final_xhatchunk = state.final_xhatchunk
    ts.final_state = state
    if x is not None:
      ts.final_xchunk = state.final_xchunk
      if back_prop:
        # take only the last subsequence's losses so we don't bypass the
        # truncation boundary.
        ts.loss = state.losses[-1]
        ts.error = state.errors[-1]
      else:
        ts.loss = tf.concat(0, state.losses)
        ts.error = tf.concat(0, state.errors)
    return ts


def _make_sequence_graph(transition=None, model_state=None, x=None,
                         initial_xchunk=None, context=None, length=None,
                         temperature=1.0, hp=None, back_prop=False):
  """Construct the graph to process a sequence of categorical integers.

  If `x` is given, the graph processes the sequence `x` in chunks of size
  determined by the hyperparameter `chunk_size`.  At step `i`, the model
  receives the `i`th nonoverlapping chunk as input, and its output is used to
  predict the `i + 1`th chunk.

  The last chunk is not processed, as there would be no further chunk
  available to compare against and compute loss. To ensure all data is
  processed during TBPTT, segments `x` fed into successive computations
  of the graph should overlap by `chunk_size`.

  If `x` is not given, `initial_xchunk` must be given as the first input
  to the model.  Further chunks are constructed from the model's predictions.

  Args:
    transition: model transition function mapping (xchunk, model_state,
        context) to (output, new_model_state).
    model_state: initial state of the model.
    x: Sequence of integer (categorical) inputs. Not needed if sampling.
        Axes [time, batch].
    initial_xchunk: When sampling, x is not given; initial_xchunk specifies
        the input x[:chunk_size] to the first timestep.
    context: a `Tensor` denoting context, e.g. for conditioning.
    length: Optional length of sequence. Inferred from `x` if possible.
    temperature: Softmax temperature to use for sampling.
    hp: Model hyperparameters.
    back_prop: Whether the graph will be backpropagated through.

  Returns:
    Namespace containing relevant symbolic variables.
  """
  with tf.variable_scope("seq") as scope:
    # if the caching device is not set explicitly, set it such that the
    # variables for the RNN are all cached locally.
    if scope.caching_device is None:
      scope.set_caching_device(lambda op: op.device)

    if length is None:
      length = tf.shape(x)[0]

    chunk_assertion = tf.Assert(tf.equal(length % hp.chunk_size, 0),
                                [length, hp.chunk_size],
                                name="chunk_assertion")
    length = tf.with_dependencies([chunk_assertion], length)
    chunk_count = length // hp.chunk_size

    def _make_ta(name, **kwargs):
      # infer_shape=False because it is too strict; it considers unknown
      # dimensions to be incompatible with anything else. Effectively that
      # requires all shapes to be fully defined at graph construction time.
      return tf.tensor_array_ops.TensorArray(
          tensor_array_name=name, infer_shape=False, **kwargs)

    state = NS(i=tf.constant(0), model=model_state)

    state.xhats = _make_ta("xhats", dtype=tf.int32, size=length,
                           clear_after_read=False)
    # populate the initial chunk of xhats
    state.xhats = _put_chunk(
        state.xhats, 0, tf.unpack(
            (initial_xchunk if x is None else x)[:hp.chunk_size, :],
            num=hp.chunk_size))

    if x is not None:
      state.losses = _make_ta("losses", dtype=tf.float32,
                              size=length - hp.chunk_size)
      state.errors = _make_ta("errors", dtype=tf.bool,
                              size=length - hp.chunk_size)

    state = tfutil.while_loop(
        cond=lambda state: state.i < chunk_count - 1,
        body=ft.partial(make_transition_graph,
                        transition=transition, x=x, context=context,
                        temperature=temperature, hp=hp),
        loop_vars=state,
        back_prop=back_prop)

    ts = NS()
    ts.final_state = state
    full_xhat = state.xhats.pack()
    ts.xhat = full_xhat[hp.chunk_size:, :]
    ts.final_xhatchunk = _get_chunk(
        full_xhat, length - hp.chunk_size, hp.chunk_size)
    if x is not None:
      ts.loss = tf.reduce_mean(state.losses.pack())
      ts.error = tf.reduce_mean(tf.to_float(state.errors.pack()))
      # expose the final, unprocessed chunk of x for convenience
      ts.final_xchunk = _get_chunk(x, length - hp.chunk_size, hp.chunk_size)
    return ts


def make_transition_graph(state, transition, x=None, context=None,
                          temperature=1.0, hp=None):
  """Make the graph that processes a single sequence element.

  Args:
    state: `_make_sequence_graph` loop state.
    transition: Model transition function mapping (xchunk, model_state,
        context) to (output, new_model_state).
    x: Sequence of integer (categorical) inputs. Axes [time, batch].
    context: Optional Tensor denoting context, shaped [batch, ?].
    temperature: Softmax temperature to use for sampling.
    hp: Model hyperparameters.

  Returns:
    Updated loop state.
  """
  state = NS.Copy(state)

  xchunk = _get_flat_chunk(state.xhats if x is None else x,
                           state.i * hp.chunk_size, hp.chunk_size,
                           depth=hp.data_dim)
  embedding = tfutil.layers([xchunk], sizes=hp.io_sizes, use_bn=hp.use_bn)
  h, state.model = transition(embedding, state.model, context=context)

  # predict the next chunk
  exhats = []
  with tf.variable_scope("xhat") as scope:
    for j in range(hp.chunk_size):
      if j > 0:
        scope.reuse_variables()

      xchunk = _get_flat_chunk(state.xhats if x is None else x,
                               state.i * hp.chunk_size + j, hp.chunk_size,
                               depth=hp.data_dim)
      embedding = tfutil.layers([h, xchunk], sizes=hp.io_sizes,
                                use_bn=hp.use_bn)
      exhat = tfutil.project(embedding, output_dim=hp.data_dim)
      exhats.append(exhat)

      state.xhats = state.xhats.write((state.i + 1) * hp.chunk_size + j,
                                      tfutil.sample(exhat, temperature))

  if x is not None:
    targets = tf.unpack(_get_1hot_chunk(x, (state.i + 1) * hp.chunk_size,
                                        hp.chunk_size, depth=hp.data_dim),
                        num=hp.chunk_size, axis=1)
    state.losses = _put_chunk(
        state.losses, state.i * hp.chunk_size,
        [tf.nn.softmax_cross_entropy_with_logits(exhat, target)
         for exhat, target in util.equizip(exhats, targets)])
    state.errors = _put_chunk(
        state.errors, state.i * hp.chunk_size,
        [tf.not_equal(tf.nn.top_k(exhat)[1], tf.nn.top_k(target)[1])
         for exhat, target in util.equizip(exhats, targets)])

  state.i += 1
  return state


def _get_chunk(array, start, size):
  """Get a chunk from a Tensor or TensorArray.

  Args:
    array: Tensor or TensorArray to take a chunk from.
    start: Index of the first element of the chunk.
    size: Length of the chunk.

  Returns:
    The subsequence array[start:start + size].
  """
  if isinstance(array, tf.Tensor):
    return tf.slice(array, tf.pack([start, 0]),
                    tf.pack([size, tf.shape(array)[1]]))
  elif isinstance(array, tf.TensorArray):
    return tf.pack([array.read(start + j) for j in range(size)])
  else:
    assert False


def _get_1hot_chunk(array, start, size, depth):
  """Get a one-hot chunk from a Tensor or TensorArray.

  Args:
    array: Tensor or TensorArray to read from.
    start: Index of the first element of the chunk.
    size: Length of the chunk.
    depth: Number of categories.

  Returns:
    The subsequence array[start:start + size], one-hot encoded.
  """
  return tfutil.shaped_one_hot(
      tf.transpose(_get_chunk(array, start, size)),
      [None, size, depth])


def _get_flat_chunk(array, start, size, depth):
  """Get a flattened one-hot chunk from a Tensor or TensorArray.

  Args:
    array: Tensor or TensorArray to read from.
    start: Index of the first element of the chunk.
    size: Length of the chunk.
    depth: Number of categories.

  Returns:
    The subsequence array[start:start + size], one-hot encoded and flattened
    to shape [batch, size * depth].
  """
  return tf.reshape(_get_1hot_chunk(array, start, size, depth),
                    [-1, size * depth])


def _put_chunk(array, start, values):
  """Write a chunk of values to a TensorArray.

  Args:
    array: TensorArray to write to.
    start: Index for the first element of the chunk.
    values: The Tensors to write to `array`.

  Returns:
    The updated TensorArray, with the values assigned to
    array[start:start + len(values)].
  """
  for j, value in enumerate(values):
    array = array.write(start + j, value)
  return array
