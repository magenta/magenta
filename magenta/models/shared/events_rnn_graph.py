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
"""Provides function to build an event sequence RNN model's graph."""

# internal imports
import tensorflow as tf
import magenta


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=False):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
        and cell matrix as a state instead of a concatenated matrix.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units, state_is_tuple=state_is_tuple)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple)

  return cell


def build_graph(mode, config, sequence_example_file_paths=None):
  """Builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
        to use.
    sequence_example_file_paths: A list of paths to TFRecord files containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation. May be a sharded file of the form.

  Returns:
    A tf.Graph instance which contains the TF ops.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError("The mode parameter must be 'train', 'eval', "
                     "or 'generate'. The mode parameter was: %s" % mode)

  hparams = config.hparams
  encoder_decoder = config.encoder_decoder

  tf.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size
  num_classes = encoder_decoder.num_classes
  no_event_label = encoder_decoder.default_event_label

  with tf.Graph().as_default() as graph:
    inputs, labels, lengths, = None, None, None
    state_is_tuple = True

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams.batch_size, input_size)

    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                           input_size])
      # If state_is_tuple is True, the output RNN cell state will be a tuple
      # instead of a tensor. During training and evaluation this improves
      # performance. However, during generation, the RNN cell state is fed
      # back into the graph with a feed dict. Feed dicts require passed in
      # values to be tensors and not tuples, so state_is_tuple is set to False.
      state_is_tuple = False

    cell = make_rnn_cell(hparams.rnn_layer_sizes,
                         dropout_keep_prob=hparams.dropout_keep_prob,
                         attn_length=hparams.attn_length,
                         state_is_tuple=state_is_tuple)

    initial_state = cell.zero_state(hparams.batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell, inputs, initial_state=initial_state, parallel_iterations=1,
        swap_memory=True)

    outputs_flat = tf.reshape(outputs, [-1, cell.output_size])
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    if mode == 'train' or mode == 'eval':
      labels_flat = tf.reshape(labels, [-1])
      mask = tf.sequence_mask(lengths)
      if hparams.skip_first_n_losses:
        skip = tf.minimum(lengths, hparams.skip_first_n_losses)
        skip_mask = tf.sequence_mask(skip, maxlen=tf.reduce_max(lengths))
        mask = tf.logical_and(mask, tf.logical_not(skip_mask))
      mask = tf.cast(mask, tf.float32)
      mask_flat = tf.reshape(mask, [-1])

      num_logits = tf.to_float(tf.reduce_sum(lengths))

      with tf.control_dependencies(
          [tf.Assert(tf.greater(num_logits, 0.), [num_logits])]):
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_flat, logits=logits_flat)
      loss = tf.reduce_sum(mask_flat * softmax_cross_entropy) / num_logits
      perplexity = (tf.reduce_sum(mask_flat * tf.exp(softmax_cross_entropy)) /
                    num_logits)

      correct_predictions = tf.to_float(
          tf.nn.in_top_k(logits_flat, labels_flat, 1)) * mask_flat
      accuracy = tf.reduce_sum(correct_predictions) / num_logits * 100

      event_positions = (
          tf.to_float(tf.not_equal(labels_flat, no_event_label)) * mask_flat)
      event_accuracy = (
          tf.reduce_sum(tf.multiply(correct_predictions, event_positions)) /
          tf.reduce_sum(event_positions) * 100)

      no_event_positions = (
          tf.to_float(tf.equal(labels_flat, no_event_label)) * mask_flat)
      no_event_accuracy = (
          tf.reduce_sum(tf.multiply(correct_predictions, no_event_positions)) /
          tf.reduce_sum(no_event_positions) * 100)

      global_step = tf.Variable(0, trainable=False, name='global_step')

      tf.add_to_collection('loss', loss)
      tf.add_to_collection('perplexity', perplexity)
      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('global_step', global_step)

      summaries = [
          tf.summary.scalar('loss', loss),
          tf.summary.scalar('perplexity', perplexity),
          tf.summary.scalar('accuracy', accuracy),
          tf.summary.scalar(
              'event_accuracy', event_accuracy),
          tf.summary.scalar(
              'no_event_accuracy', no_event_accuracy),
      ]

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

        summaries.append(tf.summary.scalar(
            'learning_rate', learning_rate))

      if mode == 'eval':
        summary_op = tf.summary.merge(summaries)
        tf.add_to_collection('summary_op', summary_op)

    elif mode == 'generate':
      temperature = tf.placeholder(tf.float32, [])
      softmax_flat = tf.nn.softmax(
          tf.div(logits_flat, tf.fill([num_classes], temperature)))
      softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('initial_state', initial_state)
      tf.add_to_collection('final_state', final_state)
      tf.add_to_collection('temperature', temperature)
      tf.add_to_collection('softmax', softmax)

  return graph
