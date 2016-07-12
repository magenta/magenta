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
"""Graph building functions for state_saving_rnn and dynamic_rnn LSTMs.

Each graph component is produced by a seperate function.
"""

import ast

# internal imports
import numpy as np
import tensorflow as tf

from magenta.lib import melodies_lib

# Number of classification classes. This is a property of the dataset.
NUM_CLASSES = 38

ENCODER_MIN_NOTE = 48
ENCODER_MAX_NOTE = 84
ENCODER_TRANSPOSE_TO_KEY = 0


class HParams(object):
  """Holds hyperparameters.

  Acts like a dictionary, but keys can be accessed as class attributes.
  An instance is initialized with default hyperparameter values. Use the
  `parse` method to set new values from a string representation of a
  Python dictionary. Use the `values` method to retrieve a Python dictionary
  version of the hyperparameter settings.
  """

  def __init__(self, **init_hparams):
    object.__setattr__(self, 'keyvals', init_hparams)

  def __getattr__(self, key):
    return self.keyvals[key]

  def __setattr__(self, key, value):
    self.keyvals[key] = value

  def parse(self, string):
    new_hparams = ast.literal_eval(string)
    return HParams(**dict(self.keyvals, **new_hparams))

  def values(self):
    return self.keyvals


def default_hparams():
  return HParams(batch_size=128,
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


def input_sequence_example(file_list, hparams):
  """Deserializes SequenceExamples from TFRecord.

  Args:
    file_list: List of TFRecord files containing SequenceExamples.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    seq_key: Key of SequenceExample as a string.
    context: Context of SequenceExample as dictionary key -> Tensor.
    sequence: Sequence of SequenceExample as dictionary key -> Tensor.
  """
  file_queue = tf.train.string_input_producer(file_list)
  reader = tf.TFRecordReader()
  seq_key, serialized_example = reader.read(file_queue)

  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[hparams.one_hot_length],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
  }

  context, sequence = tf.parse_single_sequence_example(
      serialized_example,
      sequence_features=sequence_features)
  return seq_key, context, sequence


def dynamic_rnn_batch(file_list, hparams):
  """Reads batches of SequenceExamples from TFRecord and pads them.

  Can deal with variable length SequenceExamples by padding each batch to the
  length of the longest sequence with zeros.

  Args:
    file_list: List of TFRecord files containing SequenceExamples.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    inputs: Tensor of shape [batch_size, examples_per_sequence, one_hot_length]
        with floats indicating the next note event.
    labels: Tensor of shape [batch_size, examples_per_sequence] with int64s
        indicating the prediction for next note event given the notes up to
        this point in the inputs sequence.
    lengths: Tensor vector of shape [batch_size] with the length of the
        SequenceExamples before padding.
  """
  _, _, sequences = input_sequence_example(file_list, hparams)

  length = tf.shape(sequences['inputs'])[0]

  queue = tf.PaddingFIFOQueue(
      capacity=1000,
      dtypes=[tf.float32, tf.int64, tf.int32],
      shapes=[(None, hparams.one_hot_length), (None,), ()])

  # The number of threads for enqueuing.
  num_threads = 4
  enqueue_ops = [queue.enqueue([sequences['inputs'], sequences['labels'],
                                length])] * num_threads
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
  return queue.dequeue_many(hparams.batch_size)


def dynamic_rnn_inference(inputs,
                          lengths,
                          cell,
                          hparams,
                          zero_initial_state=True,
                          parallel_iterations=1,
                          swap_memory=True):
  """Creates possibly layered LSTM cells with a linear projection layer.

  Uses dynamic_rnn which dynamically unrolls for each minibatch allowing truely
  variable length minibatches.

  Args:
    inputs: Tensor of shape [batch_size, batch_sequence_length, one_hot_length).
    lengths: Tensor of shape [batch_size] with the length of the
        SequenceExample before padding.
    cell: An RNNCell instance.
    hparams: HParams instance containing model hyperparameters.
    zero_initial_state: If true, a constant tensor of 0s is used as the initial
        RNN state. If false, a placeholder is created to hold the initial state.
    parallel_iterations: The number of iterations to run in parallel. Those
        operations which do not have any temporal dependency
        and can be run in parallel, will be. This parameter trades off
        time for space. Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU. This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.

  Returns:
    logits: Output logits. A tensor of shape
        [batch_size, batch_sequence_length, one_hot_length].
    initial_state: The tensor fed into dynamic_rnn as the initial state. When
        zero_initial_state is true, this will be the placeholder.
    final_state: The final internal state after computing the number of steps
        given in lengths for each sample. Same shape as cell.state_size.
  """
  if zero_initial_state:
    initial_state = cell.zero_state(batch_size=hparams.batch_size,
                                    dtype=tf.float32)
  else:
    initial_state = tf.placeholder(tf.float32,
                                   [hparams.batch_size, cell.state_size])

  outputs, final_state = tf.nn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=lengths,
      initial_state=initial_state,
      swap_memory=swap_memory,
      parallel_iterations=parallel_iterations)

  # create projection layer to logits.
  outputs_flat = tf.reshape(outputs, [-1, hparams.rnn_layer_sizes[-1]])
  logits_flat = tf.contrib.layers.legacy_linear(outputs_flat,
                                                hparams.one_hot_length)
  logits = tf.reshape(logits_flat,
                      [hparams.batch_size, -1, hparams.one_hot_length])

  return logits, initial_state, final_state


def make_cell(hparams):
  """Instantiates an RNNCell object.

  Will construct an appropriate RNN cell given hyperparameters. This will
  specifically be a stack of LSTM cells. The height of the stack is specified in
  hparams.

  Args:
    hparams: HParams instance containing model hyperparameters.

  Returns:
    RNNCell instance.
  """
  lstm_layers = [
      tf.nn.rnn_cell.LSTMCell(num_units=layer_size)
      for layer_size in hparams.rnn_layer_sizes
  ]
  multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
  return multi_cell


def log_perplexity_loss(logits, labels):
  """Computes the log-perplexity of the predictions given the labels.

  log-perplexity = -1/N sum{i=1..N}(log(p(x_i)))
      where x_i's are the correct classes given by labels,
      and p(x) is the model's prediction for class x.

  Softmax is applied to logits to obtain probability predictions.

  Both scaled and unscaled log-perplexities are returned (unscaled does not
  divide by N). Unscaled log-perplexity is simply cross entropy. Use cross
  entropy for training loss so that the gradient magnitudes are not affected
  by sequence length. Use log-perplexity to monitor training progress and
  compare models.

  Args:
    logits: Output tensor of a linear layer of shape
      [batch * batch_sequence_length, one_hot_length]. Must be unscaled logits.
      Do not put through softmax! This function applies softmax.
    labels: tensor of ints between 0 and one_hot_length-1 of shape
      [batch * batch_sequence_length].

  Returns:
    cross_entropy: Unscaled average log-perplexityacross minibatch samples,
        which is just the cross entropy loss. Use this loss for backprop.
    log_perplexity: Average log-perplexity across minibatch samples. Use this
        loss for monitoring training progress and comparing models.
  """
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  cross_entropy = tf.reduce_sum(losses)
  log_perplexity = cross_entropy / tf.to_float(tf.size(losses))
  return cross_entropy, log_perplexity


def train_op(loss, global_step, hparams):
  """Uses a gradient descent optimizer to minimize loss.

  Gradient descent is applied to the loss function with an exponentially
  decreasing learning rate.

  Args:
    loss: loss tensor to minimize.
    global_step: A tf.Variable of type int holding the global training step.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    training_op: An op that performs weight updates on the model.
    learning_rate: An op that decays learning rate, if that option is set in
        `hparams`.
  """
  if hparams.exponentially_decay_learning_rate:
    learning_rate = tf.train.exponential_decay(hparams.initial_learning_rate,
                                               global_step,
                                               hparams.decay_steps,
                                               hparams.decay_rate,
                                               staircase=True,
                                               name='learning_rate')
  else:
    learning_rate = tf.Variable(hparams.initial_learning_rate, trainable=False)
  opt = tf.train.AdagradOptimizer(learning_rate)
  params = tf.trainable_variables()
  gradients = tf.gradients(loss, params)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams.clip_norm)
  training_op = opt.apply_gradients(
      zip(clipped_gradients, params),
      global_step=global_step)

  return training_op, learning_rate


def eval_accuracy(predictions, labels):
  """Evaluates the accuracy of the predictions.

  Checks how often the prediciton with the highest weight is correct on average.

  Args:
    predictions: Output tensor of a linear layer of shape
      [batch * batch_sequence_length, one_hot_length].
    labels: tensor of ints between 0 and one_hot_length-1 of shape
      [batch * batch_sequence_length].

  Returns:
    The precision of the highest weighted predicted class.
  """
  correct_predictions = tf.nn.in_top_k(predictions, labels, 1)
  return tf.reduce_mean(tf.to_float(correct_predictions))


def one_hot_encoder(melody, min_note, max_note):
  """Converts a melody into a list of input features and a list of labels.

  This encoder converts each melody note to a one-hot vector (a list of floats
  that are all 0.0 and 1.0 at the index equal to the encoded value). The one-hot
  length is `max_note` - `min_note` + 2. NO_EVENT gets 0th position. NOTE_OFF
  gets 1st position. Pitches get pitch + 2.

  Two tensors are created: model inputs and model labels. Inputs are one-hot
  vectors. Each label is the note index (an int) of the one_hot that comes next.
  The vector of labels is equal to the vector of inputs (without one-hot
  encoding) shifted left by 1 and padded with a NO_EVENT or NOTE_OFF.

  The intput and label sequence lengths are padded with NO_EVENT to a multiple
  of `melody.steps_per_bar` to make them end at the end of a bar. Final bars
  with only a single event that is a NOTE_OFF are truncated rather than padded.

  Args:
    melody: A MonophonicMelody object to encode.
    min_note: Minimum pitch (inclusive) that the output notes will take on.
    max_note: Maximum pitch (exclusive) that the output notes will take on.

  Returns:
    sequence_example: A SequenceExample proto containing inputs and labels
    sequences.
  """
  note_range = max_note - min_note
  one_hot_length = note_range + melodies_lib.NUM_SPECIAL_EVENTS
  note_indices = [
      note + melodies_lib.NUM_SPECIAL_EVENTS if note < 0 else
      note - min_note + melodies_lib.NUM_SPECIAL_EVENTS for note in melody
  ]
  inputs = np.zeros((len(note_indices), one_hot_length), dtype=float)
  inputs[np.arange(len(note_indices)), note_indices] = 1.0
  labels = (note_indices[1:] +
            [melodies_lib.NO_EVENT + melodies_lib.NUM_SPECIAL_EVENTS])

  # Pad to the end of the measure.
  if len(inputs) % melody.steps_per_bar == 1:
    # Last event is always note off. If that is the only event in the last bar,
    # remove it.
    inputs = inputs[:-1]
    labels = labels[:-1]
  elif len(inputs) % melody.steps_per_bar > 1:
    # Pad out to end of the bar.
    pad_len = melody.steps_per_bar - len(inputs) % melody.steps_per_bar
    padding = np.zeros((pad_len, one_hot_length), dtype=float)
    padding[:, 0] = 1.0
    inputs = np.concatenate((inputs, padding), axis=0)
    labels += [0] * pad_len

  input_features = [
      tf.train.Feature(float_list=tf.train.FloatList(value=input_))
      for input_ in inputs
  ]
  label_features = [
      tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
      for label in labels
  ]
  feature_list = {
      'inputs': tf.train.FeatureList(feature=input_features),
      'labels': tf.train.FeatureList(feature=label_features)
  }
  feature_lists = tf.train.FeatureLists(feature_list=feature_list)
  return tf.train.SequenceExample(feature_lists=feature_lists)
