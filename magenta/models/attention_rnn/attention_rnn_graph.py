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
    cell = tf.contrib.rnn.AttentionCellWrapper(cell, hparams.attn_length,
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
