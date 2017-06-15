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

from tensorflow.python.util import nest as tf_nest


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units)
    if attn_length and not cells:
      # Add attention wrapper to first layer.
      cell = tf.contrib.rnn.AttentionCellWrapper(
          cell, attn_length, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells)

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
        evaluation.

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
    inputs, labels, lengths = None, None, None

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams.batch_size, input_size,
          shuffle=mode == 'train')

    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                           input_size])

    cell = make_rnn_cell(
        hparams.rnn_layer_sizes,
        dropout_keep_prob=(
            1.0 if mode == 'generate' else hparams.dropout_keep_prob),
        attn_length=(
            hparams.attn_length if hasattr(hparams, 'attn_length') else 0))

    initial_state = cell.zero_state(hparams.batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell, inputs, sequence_length=lengths, initial_state=initial_state,
        swap_memory=True)

    outputs_flat = magenta.common.flatten_maybe_padded_sequences(
        outputs, lengths)
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    if mode == 'train' or mode == 'eval':
      labels_flat = magenta.common.flatten_maybe_padded_sequences(
          labels, lengths)

      softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels_flat, logits=logits_flat)

      predictions_flat = tf.argmax(logits_flat, axis=1)
      correct_predictions = tf.to_float(
          tf.equal(labels_flat, predictions_flat))
      event_positions = tf.to_float(tf.not_equal(labels_flat, no_event_label))
      no_event_positions = tf.to_float(tf.equal(labels_flat, no_event_label))

      if mode == 'train':
        loss = tf.reduce_mean(softmax_cross_entropy)
        perplexity = tf.reduce_mean(tf.exp(softmax_cross_entropy))
        accuracy = tf.reduce_mean(correct_predictions)
        event_accuracy = (
            tf.reduce_sum(correct_predictions * event_positions) /
            tf.reduce_sum(event_positions))
        no_event_accuracy = (
            tf.reduce_sum(correct_predictions * no_event_positions) /
            tf.reduce_sum(no_event_positions))

        optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparams.clip_norm)
        tf.add_to_collection('train_op', train_op)

        vars_to_summarize = {
            'loss': loss,
            'metrics/perplexity': perplexity,
            'metrics/accuracy': accuracy,
            'metrics/event_accuracy': event_accuracy,
            'metrics/no_event_accuracy': no_event_accuracy,
        }
      elif mode == 'eval':
        vars_to_summarize, update_ops = tf.contrib.metrics.aggregate_metric_map(
            {
                'loss': tf.metrics.mean(softmax_cross_entropy),
                'metrics/accuracy': tf.metrics.accuracy(
                    labels_flat, predictions_flat),
                'metrics/per_class_accuracy':
                    tf.metrics.mean_per_class_accuracy(
                        labels_flat, predictions_flat, num_classes),
                'metrics/event_accuracy': tf.metrics.recall(
                    event_positions, correct_predictions),
                'metrics/no_event_accuracy': tf.metrics.recall(
                    no_event_positions, correct_predictions),
                'metrics/perplexity': tf.metrics.mean(
                    tf.exp(softmax_cross_entropy)),
            })
        for updates_op in update_ops.values():
          tf.add_to_collection('eval_ops', updates_op)

      for var_name, var_value in vars_to_summarize.iteritems():
        tf.summary.scalar(var_name, var_value)
        tf.add_to_collection(var_name, var_value)

    elif mode == 'generate':
      temperature = tf.placeholder(tf.float32, [])
      softmax_flat = tf.nn.softmax(
          tf.div(logits_flat, tf.fill([num_classes], temperature)))
      softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('temperature', temperature)
      tf.add_to_collection('softmax', softmax)
      # Flatten state tuples for metagraph compatibility.
      for state in tf_nest.flatten(initial_state):
        tf.add_to_collection('initial_state', state)
      for state in tf_nest.flatten(final_state):
        tf.add_to_collection('final_state', state)

  return graph
