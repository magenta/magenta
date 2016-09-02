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
"""Provides function to build the attention RNN model's graph."""

# internal imports
import tensorflow as tf
from tensorflow.python.util import nest

from magenta.lib import sequence_example_lib
from magenta.lib import tf_lib


def default_hparams():
  return tf_lib.HParams(
      batch_size=128,
      rnn_layer_sizes=[128, 128],
      dropout_keep_prob=0.5,
      skip_first_n_losses=0,
      attn_length=40,
      clip_norm=3,
      initial_learning_rate=0.001,
      decay_steps=1000,
      decay_rate=0.97)


def build_graph(mode, hparams_string, input_size, num_classes,
                sequence_example_file=None):
  """Builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    hparams_string: A string literal of a Python dictionary, where keys are
        hyperparameter names and values replace default values. For example:
        '{"batch_size":64,"rnn_layer_sizes":[128,128]}'
    input_size: The size of the input vectors in the inputs batch. Each
        inputs batch should have a shape [batch_size, num_steps, input_size].
    num_classes: The number of classes the labels can be.
    sequence_example_file: A string path to a TFRecord file containing
        tf.train.SequenceExamples. Only needed for training and evaluation.

  Returns:
    A tf.Graph instance which contains the TF ops.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate', or if
        sequence_example_file does not match a file when mode is 'train' or
        'eval'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError('The mode parameter must be \'train\', \'eval\', '
                     'or \'generate\'. The mode parameter was: %s' % mode)

  with tf.Graph().as_default() as graph:
    hparams = default_hparams()
    hparams = hparams.parse(hparams_string)
    tf.logging.info('hparams = %s', hparams.values())

    inputs, labels, lengths, = None, None, None
    state_is_tuple = True

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = sequence_example_lib.get_padded_batch(
          [sequence_example_file], hparams.batch_size, input_size)

    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                           input_size])
      # If state_is_tuple is True, the output RNN cell state will be a tuple
      # instead of a tensor. During training and evaluation this improves
      # performance. However, during generation, the RNN cell state is fed
      # back into the graph with a feed dict. Feed dicts require passed in
      # values to be tensors and not tuples, so state_is_tuple is set to False.
      state_is_tuple = False

    cells = []
    for num_units in hparams.rnn_layer_sizes:
      cell = tf.nn.rnn_cell.BasicLSTMCell(
          num_units, state_is_tuple=state_is_tuple)
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, output_keep_prob=hparams.dropout_keep_prob)
      cells.append(cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
    cell = AttentionCellWrapper(cell, hparams.attn_length,
                                state_is_tuple=state_is_tuple)

    initial_state = cell.zero_state(hparams.batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell, inputs, lengths, initial_state, parallel_iterations=1,
        swap_memory=True)

    outputs_flat = tf.reshape(outputs, [-1, hparams.rnn_layer_sizes[-1]])
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    if mode == 'train' or mode == 'eval':
      if hparams.skip_first_n_losses:
        logits = tf.reshape(logits_flat, [hparams.batch_size, -1, num_classes])
        logits = logits[:, hparams.skip_first_n_losses:, :]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        labels = labels[:, hparams.skip_first_n_losses:]

      labels_flat = tf.reshape(labels, [-1])
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits_flat, labels_flat))
      perplexity = tf.exp(loss)

      correct_predictions = tf.nn.in_top_k(logits_flat, labels_flat, 1)
      accuracy = tf.reduce_mean(tf.to_float(correct_predictions)) * 100

      global_step = tf.Variable(0, trainable=False, name='global_step')

      tf.add_to_collection('loss', loss)
      tf.add_to_collection('perplexity', perplexity)
      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('global_step', global_step)

      if mode == 'train':
        learning_rate = tf.train.exponential_decay(
            hparams.initial_learning_rate, global_step, hparams.decay_steps,
            hparams.decay_rate, staircase=True, name='learning_rate')

        opt = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      hparams.clip_norm)
        train_op = opt.apply_gradients(zip(clipped_gradients, params),
                                       global_step)
        tf.add_to_collection('learning_rate', learning_rate)
        tf.add_to_collection('train_op', train_op)

        tf.scalar_summary('loss', loss)
        tf.scalar_summary('perplexity', perplexity)
        tf.scalar_summary('accuracy', accuracy)
        tf.scalar_summary('learning_rate', learning_rate)

      if mode == 'eval':
        summary_op = tf.merge_summary([
            tf.scalar_summary('loss', loss),
            tf.scalar_summary('perplexity', perplexity),
            tf.scalar_summary('accuracy', accuracy)])

        tf.add_to_collection('summary_op', summary_op)

    elif mode == 'generate':
      if hparams.temperature != 1.0:
        logits_flat /= hparams.temperature

      softmax_flat = tf.nn.softmax(logits_flat)
      softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('initial_state', initial_state)
      tf.add_to_collection('final_state', final_state)
      tf.add_to_collection('softmax', softmax)

  return graph


_is_sequence = nest.is_sequence
_unpacked_state = nest.flatten


# TODO(elliotwaite): Merge with tf.contrib.rnn.rnn_cell.AttentionCellWrapper
# and use that instead when it's available in the next TensorFlow release.
class AttentionCellWrapper(tf.nn.rnn_cell.RNNCell):
  """Basic attention cell wrapper.

  Implementation based on http://arxiv.org/pdf/1412.7449v3.pdf, except
  instead of applying attention to the decoder outputs, attention is applied
  to the last attn_length cell outputs.
  """

  def __init__(self, cell, attn_length, attn_vec_size=None,
               state_is_tuple=False):
    """Create a cell with attention.

    Args:
      cell: an RNNCell, attention is added to it.
      attn_length: integer, the number of previous cell outputs to apply
          attention to.
      attn_vec_size: integer, the size of the attention vector. Defaults to
          cell.output_size.
      state_is_tuple: If True, accepted and returned states are 3-tuples of
        cell state, attns, and attn_states. By default (False), they are
        concatenated along the column axis. This default behavior will soon
        be deprecated.
    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if cell returns a state tuple but the flag
          `state_is_tuple` is `False` or if attn_length is zero or less.
    """
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
      raise TypeError('The parameter cell is not RNNCell.')
    if _is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError('Cell returns tuple of states, but the flag '
                       'state_is_tuple is not set. State size is: %s'
                       % str(cell.state_size))
    if attn_length <= 0:
      raise ValueError('attn_length should be greater than zero, got %s'
                       % str(attn_length))
    if not state_is_tuple:
      tf.logging.warn(
          '%s: Using a concatenated state is slower and will soon be '
          'deprecated. Use state_is_tuple=True.' % self)
    self._cell = cell
    self._attn_length = attn_length
    self._attn_size = cell.output_size
    self._attn_vec_size = attn_vec_size if attn_vec_size else cell.output_size
    self._state_is_tuple = state_is_tuple
    self._input_size = None

  @property
  def state_size(self):
    size = (self._cell.state_size, self._attn_size,
            self._attn_size * self._attn_length)
    if self._state_is_tuple:
      return size
    return sum(size)

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell with attention (LSTM+A)."""
    with tf.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        state, attns, attn_states = state
      else:
        state, attns, attn_states = (
            tf.slice(state, [0, 0], [-1, self._cell.state_size]),
            tf.slice(state, [0, self._cell.state_size], [-1, self._attn_size]),
            tf.slice(state, [0, self._cell.state_size + self._attn_size],
                     [-1, -1]))
      attn_states = tf.reshape(attn_states,
                               [-1, self._attn_length, self._attn_size])
      if self._input_size is None:
        self._input_size = inputs.get_shape().as_list()[1]
      inputs = tf.contrib.layers.linear(tf.concat(1, [inputs, attns]),
                                        self._input_size,
                                        scope='InputsWithAttns')
      output, new_state = self._cell(inputs, state)
      if self._state_is_tuple:
        new_state_cat = tf.concat(1, _unpacked_state(new_state))
      else:
        new_state_cat = new_state
      new_attns = self._attention(new_state_cat, attn_states)
      output = tf.contrib.layers.linear(tf.concat(1, [output, new_attns]),
                                        self._cell.output_size,
                                        scope='OutputWithAttns')
      new_attn_states = tf.concat(1, [
          tf.slice(attn_states, [0, 1, 0], [-1, -1, -1]),
          tf.expand_dims(output, 1)])
      new_attn_states = tf.reshape(new_attn_states,
                                   [-1, self._attn_length * self._attn_size])
      new_state = (new_state, new_attns, new_attn_states)
      if not self._state_is_tuple:
        new_state = tf.concat(1, list(new_state))
      return output, new_state

  def _attention(self, state, attn_states):
    with tf.variable_scope('Attention'):
      v = tf.get_variable('V', [self._attn_vec_size])
      attn_states_flat = tf.reshape(attn_states, [-1, self._attn_size])
      attn_states_vec = tf.contrib.layers.linear(
          attn_states_flat, self._attn_vec_size, scope='AttnStatesVec')
      attn_states_vec = tf.reshape(
          attn_states_vec, [-1, self._attn_length, self._attn_vec_size])
      state_vec = tf.contrib.layers.linear(
          state, self._attn_vec_size, scope='StateVec')
      state_vec = tf.expand_dims(state_vec, 1)
      attn = tf.reduce_sum(v * tf.tanh(attn_states_vec + state_vec), 2)
      attn_mask = tf.nn.softmax(attn)
      attn_mask = tf.expand_dims(attn_mask, 2)
      return tf.reduce_sum(attn_states * attn_mask, 1)
