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
"""Provides function to build a melody RNN model's graph."""

# internal imports
import tensorflow as tf
import magenta


def build_graph(mode, hparams, encoder_decoder, sequence_example_file=None):
  """Builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    hparams: A magenta.common.HParams object containing the hyperparameters to
        use.
    encoder_decoder: The MelodyEncoderDecoder being used by the model.
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

  tf.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size
  num_classes = encoder_decoder.num_classes
  no_event_label = encoder_decoder.no_event_label

  with tf.Graph().as_default() as graph:
    inputs, labels, lengths, = None, None, None
    state_is_tuple = True

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = magenta.common.get_padded_batch(
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
    if hparams.attn_length:
      cell = tf.contrib.rnn.AttentionCellWrapper(
          cell, hparams.attn_length, state_is_tuple=state_is_tuple)
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
      softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits_flat, labels_flat)
      loss = tf.reduce_mean(softmax_cross_entropy)
      perplexity = tf.reduce_mean(tf.exp(softmax_cross_entropy))

      correct_predictions = tf.to_float(
          tf.nn.in_top_k(logits_flat, labels_flat, 1))
      accuracy = tf.reduce_mean(correct_predictions) * 100

      event_positions = tf.to_float(tf.not_equal(labels_flat, no_event_label))
      event_accuracy = tf.truediv(
          tf.reduce_sum(tf.mul(correct_predictions, event_positions)),
          tf.reduce_sum(event_positions)) * 100

      no_event_positions = tf.to_float(tf.equal(labels_flat, no_event_label))
      no_event_accuracy = tf.truediv(
          tf.reduce_sum(tf.mul(correct_predictions, no_event_positions)),
          tf.reduce_sum(no_event_positions)) * 100

      global_step = tf.Variable(0, trainable=False, name='global_step')

      tf.add_to_collection('loss', loss)
      tf.add_to_collection('perplexity', perplexity)
      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('global_step', global_step)

      summaries = [
          tf.scalar_summary('loss', loss),
          tf.scalar_summary('perplexity', perplexity),
          tf.scalar_summary('accuracy', accuracy),
          tf.scalar_summary('event_accuracy', event_accuracy),
          tf.scalar_summary('no_event_accuracy', no_event_accuracy),
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

        summaries.append(tf.scalar_summary('learning_rate', learning_rate))

      if mode == 'eval':
        summary_op = tf.merge_summary(summaries)
        tf.add_to_collection('summary_op', summary_op)

    elif mode == 'generate':
      if hparams.temperature and hparams.temperature != 1.0:
        logits_flat /= hparams.temperature

      softmax_flat = tf.nn.softmax(logits_flat)
      softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('initial_state', initial_state)
      tf.add_to_collection('final_state', final_state)
      tf.add_to_collection('softmax', softmax)

  return graph
