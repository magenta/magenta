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
"""Forked classes and functions from `tf.contrib.seq2seq`."""

import abc
import collections

from magenta.contrib import rnn as contrib_rnn
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import tensor_util  # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.ops import control_flow_util  # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.ops import rnn_cell_impl  # pylint:disable=g-direct-tensorflow-import


_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


def _transpose_batch_time(x):
  """Transposes the batch and time dimensions of a Tensor.

  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A Tensor.

  Returns:
    x transposed along the first two dimensions.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.rank is not None and x_static_shape.rank < 2:
    return x

  x_rank = tf.rank(x)
  x_t = tf.transpose(
      x, tf.concat(([1, 0], tf.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tf.TensorShape(
          [x_static_shape[1], x_static_shape[0]]).concatenate(
              x_static_shape[2:]))
  return x_t


class Decoder(object, metaclass=abc.ABCMeta):
  """An RNN Decoder abstract interface object.

  Concepts used by this interface:
  - `inputs`: (structure of) tensors and TensorArrays that is passed as input to
    the RNNCell composing the decoder, at each time step.
  - `state`: (structure of) tensors and TensorArrays that is passed to the
    RNNCell instance as the state.
  - `finished`: boolean tensor telling whether each sequence in the batch is
    finished.
  - `outputs`: Instance of BasicDecoderOutput. Result of the decoding, at each
    time step.
  """

  @property
  def batch_size(self):
    """The batch size of input values."""
    raise NotImplementedError

  @property
  def output_size(self):
    """A (possibly nested tuple of...) integer[s] or `TensorShape` object[s]."""
    raise NotImplementedError

  @property
  def output_dtype(self):
    """A (possibly nested tuple of...) dtype[s]."""
    raise NotImplementedError

  @abc.abstractmethod
  def initialize(self, name=None):
    """Called before any decoding iterations.

    This methods must compute initial input values and initial state.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, initial_inputs, initial_state)`: initial values of
      'finished' flags, inputs and state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def step(self, time, inputs, state, name=None):
    """Called per step of decoding (but only once for dynamic decoding).

    Args:
      time: Scalar `int32` tensor. Current step number.
      inputs: RNNCell input (possibly nested tuple of) tensor[s] for this time
        step.
      state: RNNCell state (possibly nested tuple of) tensor[s] from previous
        time step.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`: `outputs` is an object
      containing the decoder output, `next_state` is a (structure of) state
      tensors and TensorArrays, `next_inputs` is the tensor that should be used
      as input for the next step, `finished` is a boolean tensor telling whether
      the sequence is complete, for each sequence in the batch.
    """
    raise NotImplementedError

  def finalize(self, outputs, final_state, sequence_lengths):
    """Called after decoding iterations complete.

    Args:
      outputs: RNNCell outputs (possibly nested tuple of) tensor[s] for all time
        steps.
      final_state: RNNCell final state (possibly nested tuple of) tensor[s] for
        last time step.
      sequence_lengths: 1-D `int32` tensor containing lengths of each sequence.

    Returns:
      `(final_outputs, final_state)`: `final_outputs` is an object containing
      the final decoder output, `final_state` is a (structure of) state tensors
      and TensorArrays.
    """
    raise NotImplementedError

  @property
  def tracks_own_finished(self):
    """Describes whether the Decoder keeps track of finished states.

    Most decoders will emit a true/false `finished` value independently
    at each time step.  In this case, the `dynamic_decode` function keeps track
    of which batch entries are already finished, and performs a logical OR to
    insert new batches to the finished set.

    Some decoders, however, shuffle batches / beams between time steps and
    `dynamic_decode` will mix up the finished state across these entries because
    it does not track the reshuffle across time steps.  In this case, it is
    up to the decoder to declare that it will keep track of its own finished
    state by setting this property to `True`.

    Returns:
      Python bool.
    """
    return False


class BaseDecoder(tf.layers.Layer):
  """An RNN Decoder that is based on a Keras layer.

  Concepts used by this interface:
  - `inputs`: (structure of) tensors and TensorArrays that is passed as input to
    the RNNCell composing the decoder, at each time step.
  - `state`: (structure of) tensors and TensorArrays that is passed to the
    RNNCell instance as the state.
  - `memory`: (sturecute of) tensors that is usually the full output of the
    encoder, which will be used for the attention wrapper for the RNNCell.
  - `finished`: boolean tensor telling whether each sequence in the batch is
    finished.
  - `outputs`: Instance of BasicDecoderOutput. Result of the decoding, at each
    time step.
  """

  def __init__(self,
               output_time_major=False,
               impute_finished=False,
               maximum_iterations=None,
               parallel_iterations=32,
               swap_memory=False,
               **kwargs):
    self.output_time_major = output_time_major
    self.impute_finished = impute_finished
    self.maximum_iterations = maximum_iterations
    self.parallel_iterations = parallel_iterations
    self.swap_memory = swap_memory
    super(BaseDecoder, self).__init__(**kwargs)

  def call(self, inputs, initial_state=None, **kwargs):
    init_kwargs = kwargs
    init_kwargs["initial_state"] = initial_state
    return dynamic_decode(self,
                          output_time_major=self.output_time_major,
                          impute_finished=self.impute_finished,
                          maximum_iterations=self.maximum_iterations,
                          parallel_iterations=self.parallel_iterations,
                          swap_memory=self.swap_memory,
                          decoder_init_input=inputs,
                          decoder_init_kwargs=init_kwargs)

  @property
  def batch_size(self):
    """The batch size of input values."""
    raise NotImplementedError

  @property
  def output_size(self):
    """A (possibly nested tuple of...) integer[s] or `TensorShape` object[s]."""
    raise NotImplementedError

  @property
  def output_dtype(self):
    """A (possibly nested tuple of...) dtype[s]."""
    raise NotImplementedError

  def initialize(self, inputs, initial_state=None, **kwargs):
    """Called before any decoding iterations.

    This methods must compute initial input values and initial state.

    Args:
      inputs: (structure of) tensors that contains the input for the decoder. In
        the normal case, its a tensor with shape [batch, timestep, embedding].
      initial_state: (structure of) tensors that contains the initial state for
        the RNNCell.
      **kwargs: Other arguments that are passed in from layer.call() method. It
        could contains item like input sequence_length, or masking for input.

    Returns:
      `(finished, initial_inputs, initial_state)`: initial values of
      'finished' flags, inputs and state.
    """
    raise NotImplementedError

  def step(self, time, inputs, state):
    """Called per step of decoding (but only once for dynamic decoding).

    Args:
      time: Scalar `int32` tensor. Current step number.
      inputs: RNNCell input (possibly nested tuple of) tensor[s] for this time
        step.
      state: RNNCell state (possibly nested tuple of) tensor[s] from previous
        time step.

    Returns:
      `(outputs, next_state, next_inputs, finished)`: `outputs` is an object
      containing the decoder output, `next_state` is a (structure of) state
      tensors and TensorArrays, `next_inputs` is the tensor that should be used
      as input for the next step, `finished` is a boolean tensor telling whether
      the sequence is complete, for each sequence in the batch.
    """
    raise NotImplementedError

  def finalize(self, outputs, final_state, sequence_lengths):
    raise NotImplementedError

  @property
  def tracks_own_finished(self):
    """Describes whether the Decoder keeps track of finished states.

    Most decoders will emit a true/false `finished` value independently
    at each time step.  In this case, the `dynamic_decode` function keeps track
    of which batch entries are already finished, and performs a logical OR to
    insert new batches to the finished set.

    Some decoders, however, shuffle batches / beams between time steps and
    `dynamic_decode` will mix up the finished state across these entries because
    it does not track the reshuffle across time steps.  In this case, it is
    up to the decoder to declare that it will keep track of its own finished
    state by setting this property to `True`.

    Returns:
      Python bool.
    """
    return False

  # TODO(scottzhu): Add build/get_config/from_config and other layer methods.


def _create_zero_outputs(size, dtype, batch_size):
  """Create a zero outputs Tensor structure."""
  def _create(s, d):
    return _zero_state_tensors(s, batch_size, d)

  return tf.nest.map_structure(_create, size, dtype)


def _should_cache_variables():
  """Returns True if a default caching device should be set, otherwise False."""
  # Don't set a caching device when running in a loop, since it is possible that
  # train steps could be wrapped in a tf.while_loop. In that scenario caching
  # prevents forward computations in loop iterations from re-reading the
  # updated weights.
  graph = tf.get_default_graph()
  ctxt = graph._get_control_flow_context()  # pylint: disable=protected-access
  in_v1_while_loop = (
      control_flow_util.GetContainingWhileContext(ctxt) is not None)
  return not in_v1_while_loop


def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None,
                   **kwargs):
  """Perform dynamic decoding with `decoder`.

  Calls initialize() once and step() repeatedly on the Decoder object.

  Args:
    decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.
    **kwargs: dict, other keyword arguments for dynamic_decode. It might contain
      arguments for `BaseDecoder` to initialize, which takes all tensor inputs
      during call().

  Returns:
    `(final_outputs, final_state, final_sequence_lengths)`.

  Raises:
    TypeError: if `decoder` is not an instance of `Decoder`.
    ValueError: if `maximum_iterations` is provided but is not a scalar.
  """
  if not isinstance(decoder, (Decoder, BaseDecoder)):
    raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                    type(decoder))

  with tf.variable_scope(scope, "decoder") as varscope:
    # Enable variable caching if it is safe to do so.
    if _should_cache_variables():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    if maximum_iterations is not None:
      maximum_iterations = tf.convert_to_tensor(
          maximum_iterations, dtype=tf.int32, name="maximum_iterations")
      if maximum_iterations.get_shape().ndims != 0:
        raise ValueError("maximum_iterations must be a scalar")

    if isinstance(decoder, Decoder):
      initial_finished, initial_inputs, initial_state = decoder.initialize()
    else:
      # For BaseDecoder that takes tensor inputs during call.
      decoder_init_input = kwargs.pop("decoder_init_input", None)
      decoder_init_kwargs = kwargs.pop("decoder_init_kwargs", {})
      initial_finished, initial_inputs, initial_state = decoder.initialize(
          decoder_init_input, **decoder_init_kwargs)

    zero_outputs = _create_zero_outputs(decoder.output_size,
                                        decoder.output_dtype,
                                        decoder.batch_size)

    # If we are in an XLA context, we set maximum_iterations on the while loop
    # and set a fixed size for TensorArrays.
    is_xla = control_flow_util.GraphOrParentsInXlaContext(
        tf.get_default_graph())
    if is_xla and maximum_iterations is None:
      raise ValueError("maximum_iterations is required for XLA compilation.")
    if maximum_iterations is not None:
      initial_finished = tf.logical_or(
          initial_finished, 0 >= maximum_iterations)
    initial_sequence_lengths = tf.zeros_like(
        initial_finished, dtype=tf.int32)
    initial_time = tf.constant(0, dtype=tf.int32)

    def _shape(batch_size, from_shape):
      if (not isinstance(from_shape, tf.TensorShape) or
          from_shape.ndims == 0):
        return None
      else:
        batch_size = tensor_util.constant_value(
            tf.convert_to_tensor(batch_size, name="batch_size"))
        return tf.TensorShape([batch_size]).concatenate(from_shape)

    dynamic_size = maximum_iterations is None or not is_xla

    def _create_ta(s, d):
      return tf.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=_shape(decoder.batch_size, s))

    initial_outputs_ta = tf.nest.map_structure(_create_ta, decoder.output_size,
                                               decoder.output_dtype)

    def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                  finished, unused_sequence_lengths):
      return tf.logical_not(tf.reduce_all(finished))

    def body(time, outputs_ta, state, inputs, finished, sequence_lengths):
      """Internal while_loop body.

      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: bool tensor (keeping track of what's finished).
        sequence_lengths: int32 tensor (keeping track of time of finish).

      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
          next_sequence_lengths)`.
        ```
      """
      (next_outputs, decoder_state, next_inputs,
       decoder_finished) = decoder.step(time, inputs, state)
      decoder_state_sequence_lengths = False
      if decoder.tracks_own_finished:
        next_finished = decoder_finished
        lengths = getattr(decoder_state, "lengths", None)
        if lengths is not None:
          # sequence lengths are provided by decoder_state.lengths; overwrite
          # our sequence lengths.
          decoder_state_sequence_lengths = True
          sequence_lengths = tf.cast(lengths, tf.int32)
      else:
        next_finished = tf.logical_or(decoder_finished, finished)

      if decoder_state_sequence_lengths:
        # Just pass something through the loop; at the next iteration we'll pull
        # the sequence lengths from the decoder_state again.
        next_sequence_lengths = sequence_lengths
      else:
        next_sequence_lengths = tf.where(
            tf.logical_not(finished),
            tf.fill(tf.shape(sequence_lengths), time + 1),
            sequence_lengths)

      tf.nest.assert_same_structure(state, decoder_state)
      tf.nest.assert_same_structure(outputs_ta, next_outputs)
      tf.nest.assert_same_structure(inputs, next_inputs)

      # Zero out output values past finish
      if impute_finished:
        emit = tf.nest.map_structure(
            lambda out, zero: tf.where(finished, zero, out),
            next_outputs,
            zero_outputs)
      else:
        emit = next_outputs

      # Copy through states past finish
      def _maybe_copy_state(new, cur):
        # TensorArrays and scalar states get passed through.
        if isinstance(cur, tf.TensorArray):
          pass_through = True
        else:
          new.set_shape(cur.shape)
          pass_through = (new.shape.ndims == 0)
        return new if pass_through else tf.where(finished, cur, new)

      if impute_finished:
        next_state = tf.nest.map_structure(
            _maybe_copy_state, decoder_state, state)
      else:
        next_state = decoder_state

      outputs_ta = tf.nest.map_structure(lambda ta, out: ta.write(time, out),
                                         outputs_ta, emit)
      return (time + 1, outputs_ta, next_state, next_inputs, next_finished,
              next_sequence_lengths)

    res = tf.while_loop(
        condition,
        body,
        loop_vars=(
            initial_time,
            initial_outputs_ta,
            initial_state,
            initial_inputs,
            initial_finished,
            initial_sequence_lengths,
        ),
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
        swap_memory=swap_memory)

    final_outputs_ta = res[1]
    final_state = res[2]
    final_sequence_lengths = res[5]

    final_outputs = tf.nest.map_structure(
        lambda ta: ta.stack(), final_outputs_ta)

    try:
      final_outputs, final_state = decoder.finalize(
          final_outputs, final_state, final_sequence_lengths)
    except NotImplementedError:
      pass

    if not output_time_major:
      final_outputs = tf.nest.map_structure(
          _transpose_batch_time, final_outputs)

  return final_outputs, final_state, final_sequence_lengths


# The following sample functions (_call_sampler, bernoulli_sample,
# categorical_sample) mimic TensorFlow Probability distribution semantics.


def _call_sampler(sample_n_fn, sample_shape, name=None):
  """Reshapes vector of samples."""
  with tf.name_scope(name, "call_sampler", values=[sample_shape]):
    sample_shape = tf.convert_to_tensor(
        sample_shape, dtype=tf.int32, name="sample_shape")
    # Ensure sample_shape is a vector (vs just a scalar).
    pad = tf.cast(tf.equal(tf.rank(sample_shape), 0), tf.int32)
    sample_shape = tf.reshape(
        sample_shape,
        tf.pad(tf.shape(sample_shape),
               paddings=[[pad, 0]],
               constant_values=1))
    samples = sample_n_fn(tf.reduce_prod(sample_shape))
    batch_event_shape = tf.shape(samples)[1:]
    final_shape = tf.concat([sample_shape, batch_event_shape], 0)
    return tf.reshape(samples, final_shape)


def bernoulli_sample(probs=None, logits=None, dtype=tf.int32,
                     sample_shape=(), seed=None):
  """Samples from Bernoulli distribution."""
  if probs is None:
    probs = tf.sigmoid(logits, name="probs")
  else:
    probs = tf.convert_to_tensor(probs, name="probs")
  batch_shape_tensor = tf.shape(probs)
  def _sample_n(n):
    """Sample vector of Bernoullis."""
    new_shape = tf.concat([[n], batch_shape_tensor], 0)
    uniform = tf.random_uniform(
        new_shape, seed=seed, dtype=probs.dtype)
    return tf.cast(tf.less(uniform, probs), dtype)
  return _call_sampler(_sample_n, sample_shape)


def categorical_sample(logits, dtype=tf.int32,
                       sample_shape=(), seed=None):
  """Samples from categorical distribution."""
  logits = tf.convert_to_tensor(logits, name="logits")
  event_size = tf.shape(logits)[-1]
  batch_shape_tensor = tf.shape(logits)[:-1]
  def _sample_n(n):
    """Sample vector of categoricals."""
    if logits.shape.ndims == 2:
      logits_2d = logits
    else:
      logits_2d = tf.reshape(logits, [-1, event_size])
    sample_dtype = tf.int64 if logits.dtype.size > 4 else tf.int32
    draws = tf.multinomial(
        logits_2d, n, seed=seed, output_dtype=sample_dtype)
    draws = tf.reshape(
        tf.transpose(draws),
        tf.concat([[n], batch_shape_tensor], 0))
    return tf.cast(draws, dtype)
  return _call_sampler(_sample_n, sample_shape)


def _unstack_ta(inp):
  return tf.TensorArray(
      dtype=inp.dtype, size=tf.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)


class Helper(object, metaclass=abc.ABCMeta):
  """Interface for implementing sampling in seq2seq decoders.

  Helper instances are used by `BasicDecoder`.
  """

  @abc.abstractproperty
  def batch_size(self):
    """Batch size of tensor returned by `sample`.

    Returns a scalar int32 tensor.
    """
    raise NotImplementedError("batch_size has not been implemented")

  @abc.abstractproperty
  def sample_ids_shape(self):
    """Shape of tensor returned by `sample`, excluding the batch dimension.

    Returns a `TensorShape`.
    """
    raise NotImplementedError("sample_ids_shape has not been implemented")

  @abc.abstractproperty
  def sample_ids_dtype(self):
    """DType of tensor returned by `sample`.

    Returns a DType.
    """
    raise NotImplementedError("sample_ids_dtype has not been implemented")

  @abc.abstractmethod
  def initialize(self, name=None):
    """Returns `(initial_finished, initial_inputs)`."""
    pass

  @abc.abstractmethod
  def sample(self, time, outputs, state, name=None):
    """Returns `sample_ids`."""
    pass

  @abc.abstractmethod
  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """Returns `(finished, next_inputs, next_state)`."""
    pass


class CustomHelper(Helper):
  """Base abstract class that allows the user to customize sampling."""

  def __init__(self, initialize_fn, sample_fn, next_inputs_fn,
               sample_ids_shape=None, sample_ids_dtype=None):
    """Initializer.

    Args:
      initialize_fn: callable that returns `(finished, next_inputs)`
        for the first iteration.
      sample_fn: callable that takes `(time, outputs, state)`
        and emits tensor `sample_ids`.
      next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
        and emits `(finished, next_inputs, next_state)`.
      sample_ids_shape: Either a list of integers, or a 1-D Tensor of type
        `int32`, the shape of each value in the `sample_ids` batch. Defaults to
        a scalar.
      sample_ids_dtype: The dtype of the `sample_ids` tensor. Defaults to int32.
    """
    self._initialize_fn = initialize_fn
    self._sample_fn = sample_fn
    self._next_inputs_fn = next_inputs_fn
    self._batch_size = None
    self._sample_ids_shape = tf.TensorShape(sample_ids_shape or [])
    self._sample_ids_dtype = sample_ids_dtype or tf.int32

  @property
  def batch_size(self):
    if self._batch_size is None:
      raise ValueError("batch_size accessed before initialize was called")
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return self._sample_ids_shape

  @property
  def sample_ids_dtype(self):
    return self._sample_ids_dtype

  def initialize(self, name=None):
    with tf.name_scope(name, "%sInitialize" % type(self).__name__):
      (finished, next_inputs) = self._initialize_fn()
      if self._batch_size is None:
        self._batch_size = tf.size(finished)
    return (finished, next_inputs)

  def sample(self, time, outputs, state, name=None):
    with tf.name_scope(
        name, "%sSample" % type(self).__name__, (time, outputs, state)):
      return self._sample_fn(time=time, outputs=outputs, state=state)

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with tf.name_scope(
        name, "%sNextInputs" % type(self).__name__, (time, outputs, state)):
      return self._next_inputs_fn(
          time=time, outputs=outputs, state=state, sample_ids=sample_ids)


class TrainingHelper(Helper):
  """A helper for use during training.  Only reads inputs.

  Returned sample_ids are the argmax of the RNN output logits.
  """

  def __init__(self, inputs, sequence_length, time_major=False, name=None):
    """Initializer.

    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.

    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    with tf.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
      inputs = tf.convert_to_tensor(inputs, name="inputs")
      self._inputs = inputs
      if not time_major:
        inputs = tf.nest.map_structure(_transpose_batch_time, inputs)

      self._input_tas = tf.nest.map_structure(_unstack_ta, inputs)
      self._sequence_length = tf.convert_to_tensor(
          sequence_length, name="sequence_length")
      if self._sequence_length.get_shape().ndims != 1:
        raise ValueError(
            "Expected sequence_length to be a vector, but received shape: %s" %
            self._sequence_length.get_shape())

      self._zero_inputs = tf.nest.map_structure(
          lambda inp: tf.zeros_like(inp[0, :]), inputs)

      self._batch_size = tf.size(sequence_length)

  @property
  def inputs(self):
    return self._inputs

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return tf.int32

  def initialize(self, name=None):
    with tf.name_scope(name, "TrainingHelperInitialize"):
      finished = tf.equal(0, self._sequence_length)
      all_finished = tf.reduce_all(finished)
      next_inputs = tf.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: tf.nest.map_structure(  # pylint:disable=g-long-lambda
              lambda inp: inp.read(0), self._input_tas))
      return (finished, next_inputs)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    with tf.name_scope(name, "TrainingHelperSample", [time, outputs]):
      sample_ids = tf.cast(
          tf.argmax(outputs, axis=-1), tf.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for TrainingHelper."""
    with tf.name_scope(name, "TrainingHelperNextInputs",
                       [time, outputs, state]):
      next_time = time + 1
      finished = (next_time >= self._sequence_length)
      all_finished = tf.reduce_all(finished)
      def read_from_ta(inp):
        return inp.read(next_time)
      next_inputs = tf.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: tf.nest.map_structure(read_from_ta, self._input_tas))
      return (finished, next_inputs, state)


class ScheduledOutputTrainingHelper(TrainingHelper):
  """A training helper that adds scheduled sampling directly to outputs.

  Returns False for sample_ids where no sampling took place; True elsewhere.
  """

  def __init__(self, inputs, sequence_length, sampling_probability,
               time_major=False, seed=None, next_inputs_fn=None,
               auxiliary_inputs=None, name=None):
    """Initializer.

    Args:
      inputs: A (structure) of input tensors.
      sequence_length: An int32 vector tensor.
      sampling_probability: A 0D `float32` tensor: the probability of sampling
        from the outputs instead of reading directly from the inputs.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      seed: The sampling seed.
      next_inputs_fn: (Optional) callable to apply to the RNN outputs to create
        the next input when sampling. If `None` (default), the RNN outputs will
        be used as the next inputs.
      auxiliary_inputs: An optional (structure of) auxiliary input tensors with
        a shape that matches `inputs` in all but (potentially) the final
        dimension. These tensors will be concatenated to the sampled output or
        the `inputs` when not sampling for use as the next input.
      name: Name scope for any created operations.

    Raises:
      ValueError: if `sampling_probability` is not a scalar or vector.
    """
    with tf.name_scope(name, "ScheduledOutputTrainingHelper",
                       [inputs, auxiliary_inputs, sampling_probability]):
      self._sampling_probability = tf.convert_to_tensor(
          sampling_probability, name="sampling_probability")
      if self._sampling_probability.get_shape().ndims not in (0, 1):
        raise ValueError(
            "sampling_probability must be either a scalar or a vector. "
            "saw shape: %s" % (self._sampling_probability.get_shape()))

      if auxiliary_inputs is None:
        maybe_concatenated_inputs = inputs
      else:
        inputs = tf.convert_to_tensor(inputs, name="inputs")
        auxiliary_inputs = tf.convert_to_tensor(
            auxiliary_inputs, name="auxiliary_inputs")
        maybe_concatenated_inputs = tf.nest.map_structure(
            lambda x, y: tf.concat((x, y), -1),
            inputs, auxiliary_inputs)
        if not time_major:
          auxiliary_inputs = tf.nest.map_structure(
              _transpose_batch_time, auxiliary_inputs)

      self._auxiliary_input_tas = (
          tf.nest.map_structure(_unstack_ta, auxiliary_inputs)
          if auxiliary_inputs is not None else None)

      self._seed = seed

      self._next_inputs_fn = next_inputs_fn

      super(ScheduledOutputTrainingHelper, self).__init__(
          inputs=maybe_concatenated_inputs,
          sequence_length=sequence_length,
          time_major=time_major,
          name=name)

  def initialize(self, name=None):
    return super(ScheduledOutputTrainingHelper, self).initialize(name=name)

  def sample(self, time, outputs, state, name=None):
    with tf.name_scope(name, "ScheduledOutputTrainingHelperSample",
                       [time, outputs, state]):
      return bernoulli_sample(
          probs=self._sampling_probability,
          sample_shape=self.batch_size,
          seed=self._seed)

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with tf.name_scope(name, "ScheduledOutputTrainingHelperNextInputs",
                       [time, outputs, state, sample_ids]):
      (finished, base_next_inputs, state) = (
          super(ScheduledOutputTrainingHelper, self).next_inputs(
              time=time,
              outputs=outputs,
              state=state,
              sample_ids=sample_ids,
              name=name))
      sample_ids = tf.cast(sample_ids, tf.bool)

      def maybe_sample():
        """Perform scheduled sampling."""

        def maybe_concatenate_auxiliary_inputs(outputs_, indices=None):
          """Concatenate outputs with auxiliary inputs, if they exist."""
          if self._auxiliary_input_tas is None:
            return outputs_

          next_time = time + 1
          auxiliary_inputs = tf.nest.map_structure(
              lambda ta: ta.read(next_time), self._auxiliary_input_tas)
          if indices is not None:
            auxiliary_inputs = tf.gather_nd(auxiliary_inputs, indices)
          return tf.nest.map_structure(
              lambda x, y: tf.concat((x, y), -1),
              outputs_, auxiliary_inputs)

        if self._next_inputs_fn is None:
          return tf.where(
              sample_ids, maybe_concatenate_auxiliary_inputs(outputs),
              base_next_inputs)

        where_sampling = tf.cast(
            tf.where(sample_ids), tf.int32)
        where_not_sampling = tf.cast(
            tf.where(tf.logical_not(sample_ids)), tf.int32)
        outputs_sampling = tf.gather_nd(outputs, where_sampling)
        inputs_not_sampling = tf.gather_nd(base_next_inputs,
                                           where_not_sampling)
        sampled_next_inputs = maybe_concatenate_auxiliary_inputs(
            self._next_inputs_fn(outputs_sampling), where_sampling)

        base_shape = tf.shape(base_next_inputs)
        return (tf.scatter_nd(indices=where_sampling,
                              updates=sampled_next_inputs,
                              shape=base_shape)
                + tf.scatter_nd(indices=where_not_sampling,
                                updates=inputs_not_sampling,
                                shape=base_shape))

      all_finished = tf.reduce_all(finished)
      no_samples = tf.logical_not(tf.reduce_any(sample_ids))
      next_inputs = tf.cond(
          tf.logical_or(all_finished, no_samples),
          lambda: base_next_inputs, maybe_sample)
      return (finished, next_inputs, state)


class InferenceHelper(Helper):
  """A helper to use during inference with a custom sampling function."""

  def __init__(self, sample_fn, sample_shape, sample_dtype,
               start_inputs, end_fn, next_inputs_fn=None):
    """Initializer.

    Args:
      sample_fn: A callable that takes `outputs` and emits tensor `sample_ids`.
      sample_shape: Either a list of integers, or a 1-D Tensor of type `int32`,
        the shape of the each sample in the batch returned by `sample_fn`.
      sample_dtype: the dtype of the sample returned by `sample_fn`.
      start_inputs: The initial batch of inputs.
      end_fn: A callable that takes `sample_ids` and emits a `bool` vector
        shaped `[batch_size]` indicating whether each sample is an end token.
      next_inputs_fn: (Optional) A callable that takes `sample_ids` and returns
        the next batch of inputs. If not provided, `sample_ids` is used as the
        next batch of inputs.
    """
    self._sample_fn = sample_fn
    self._end_fn = end_fn
    self._sample_shape = tf.TensorShape(sample_shape)
    self._sample_dtype = sample_dtype
    self._next_inputs_fn = next_inputs_fn
    self._batch_size = tf.shape(start_inputs)[0]
    self._start_inputs = tf.convert_to_tensor(
        start_inputs, name="start_inputs")

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return self._sample_shape

  @property
  def sample_ids_dtype(self):
    return self._sample_dtype

  def initialize(self, name=None):
    finished = tf.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    del time, state  # unused by sample
    return self._sample_fn(outputs)

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    del time, outputs  # unused by next_inputs
    if self._next_inputs_fn is None:
      next_inputs = sample_ids
    else:
      next_inputs = self._next_inputs_fn(sample_ids)
    finished = self._end_fn(sample_ids)
    return (finished, next_inputs, state)


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
  pass


class BasicDecoder(Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, initial_state, output_layer=None):
    """Initialize BasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.compat.v1.layers.Layer`, i.e.,
        `tf.compat.v1.layers.Dense`. Optional layer to apply to the RNN output
        prior to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    contrib_rnn.assert_like_rnncell("cell", cell)
    if not isinstance(helper, Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None and
        not isinstance(output_layer, tf.layers.Layer)):
      raise TypeError("output_layer must be a Layer, received: %s" %
                      type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    """Compute output size of RNN."""
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = tf.nest.map_structure(
          lambda s: tf.TensorShape([None]).concatenate(s), size)
      layer_output_shape = self._output_layer.compute_output_shape(
          output_shape_with_unknown_batch)
      return tf.nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and the sample_ids_dtype from the helper.
    dtype = tf.nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        tf.nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        self._helper.sample_ids_dtype)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with tf.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)
