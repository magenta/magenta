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
r"""Trains a basic RNN language model for melody next-note prediction.

This model provides baselines for the application of language modeling to melody
generation. This code also serves as a working example for implementing a
language model in TensorFlow.

How to run training:
  EXPERIMENT_RUN_DIR=<your experiment run directory>
  TRAIN_SET=<your training set>
  HPARAMS=<hyperparameter string>
  NUM_TRAINING_STEPS=<# steps>
  bazel build learning/brain/research/magenta/models/melody:basic_rnn_train
  bazel-bin/learning/brain/research/magenta/models/melody/basic_rnn_train \
    --experiment_run_dir=$EXPERIMENT_RUN_DIR \
    --sequence_example_file=$TRAIN_SET \
    --hparams=$HPARAMS \
    --num_training_steps=$NUM_TRAINING_STEPS

There are two approaches to building a recurrent model in TensorFlow:
1) Dynamic looping - implemented in tf.nn.dynamic_rnn:
   Use a dynamic loop to unroll the recurrent model at graph eval time.
   A while_loop control flow op is used under the hood, which has a circular
   connection to itself in the graph. Only one instance of the graph inside the
   loop is constructed.
2) Chunking - implemented in tf.nn.state_saving_rnn:
   Unroll a chunk of the recurrent model at graph construction time using a
   Python loop. Training examples are trained in chunks on the unrolled portion
   of the loop.

Dynamic looping is implemented here.

================================================================================
The data
================================================================================

The training data is stored as tf.SequenceExample protos. Each training sample
is a sequence of notes given as one-hot vectors and a sequence of labels. In
language modeling the label is just the next correct item in the sequence, so
the labels are the input sequence shifted backward by one step. The last label
is a rest.

0 = no-event
1 = note-off event
2 = note-on event for pitch 48
3 = note-on event for pitch 49
...
37 = note-on event for pitch 83

A no-event continues the previous state, whether thats continuing to hold a note
on or continuing a rest.

The 'inputs' represent the current note, which is stored as a one-hot vector,
and 'labels' represent the next note, which is stored as an int. For example,
the first 9 16th notes of Twinkle Twinkle Little Star would be encoded:

14, 0, 14, 0, 21, 0, 21, 0, 23

So if batch_size = 1, and num_unroll = 8, batch.sequences['inputs'] would be
the tensor:

[[[0.0, 0.0, ... 1.0 (14th index), ... 0.0, 0.0],
  [1.0, 0.0, ... 0.0, 0.0],
  [0.0, 0.0, ... 1.0 (14th index), ... 0.0, 0.0],
  [1.0, 0.0, ... 0.0, 0.0],
  [0.0, 0.0, ... 1.0 (21st index), ... 0.0, 0.0],
  [1.0, 0.0, ... 0.0, 0.0],
  [0.0, 0.0, ... 1.0 (21st index), ... 0.0, 0.0],
  [1.0, 0.0, ... 0.0, 0.0]]]

And batch.sequences['labels'] would be the tensor:

[[0, 14, 0, 21, 0, 21, 0, 23]]

The first dimension of the tensors is the batch_size, and since
batch_size = 1 in this example, the batch only contains one sequence.

Heres a brief description of each method:

================================================================================
Dynamic looping method
================================================================================

The data reading pipeline is implemented with a tf.PaddingFIFOQueue.
Most of the complicated data reading code has been collected into a single
function, basic_rnn_ops.dynamic_rnn_batch(), which returns a batch queue:

  (inputs, labels, lengths) = basic_rnn_ops.dynamic_rnn_batch(*args)

The recurrent model is constructed with tf.nn.dynamic_rnn(). This code is inside
basic_rnn_ops.dynamic_rnn():

  hidden, final_state = tf.nn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=lengths,
      initial_state=initial_state)

================================================================================
"""

import collections
import logging
import os
import re
import sys
import time

# internal imports
import tensorflow as tf

import basic_rnn_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('eval', False,
                            'If true, this process evaluates the model and '
                            'does not update weights.')
tf.app.flags.DEFINE_string('sequence_example_file', '',
                           'Path to TFRecord file containing '
                           'tf.SequenceExample records for training.')
tf.app.flags.DEFINE_string('experiment_run_dir', '/tmp/basic_rnn/run1',
                           'Path to directory where output from this run will '
                           'be saved. A run\'s output consists of training '
                           'data - model checkpoints and summaries - and if '
                           'an eval job is run eval data which are summaries. '
                           'Multiple runs are typically stored in a parent '
                           'directory for the experiment, where an experiment '
                           'is a particular combination of hyperparameters. '
                           'Training and eval directories are recursively '
                           'created by their respective train and eval jobs. '
                           'Point TensorBoard to the experiment to see all runs.')
tf.app.flags.DEFINE_string('hparams', '{}',
                           'String representation of Python dictionary '
                           'containing hyperparameter to value mapping. This '
                           'mapping is merged with the default '
                           'hyperparameters.')
tf.app.flags.DEFINE_integer('num_training_steps', 10000,
                            'The the number of training steps to take in this '
                            'training session.')
tf.app.flags.DEFINE_integer('summary_frequency', 10,
                            'A summary statement will be logged every '
                            '`summary_frequency` steps of training.')
tf.app.flags.DEFINE_integer('steps_to_average', 20,
                            'Accuracy averaged over the last '
                            '`steps_to_average` steps is reported.')


def make_graph(sequence_example_file='', hparams_string='{}', is_eval_mode=False):
  """Construct the model and return the graph.

  Constructs the TensorFlow graph. Hyperparameters
  are given in the hparams flag as a string representation of a Python
  dictionary.
  For example: '{"batch_size":64,"rnn_layer_sizes":[100,100]}'

  Args:
    sequence_example_file: String path to tfrecord file containing training
        samples.
    hparams_string: String literal of a Python dictionary, where keys are
        hyperparameter names and values replace default values.
    is_eval_mode: If True, training related ops are not build.

  Returns:
    tf.Graph instance which contains the TF ops.

  Raises:
    ValueError: If sequence_example_file does not match any files.
  """
  file_list = [sequence_example_file]
  logging.info('Dataset files: %s', file_list)

  with tf.Graph().as_default() as graph:
    hparams = basic_rnn_ops.default_hparams()
    hparams = hparams.parse(hparams_string)
    logging.info('hparams = %s', hparams.values())

    with tf.variable_scope('rnn_model'):
      # Define the type of RNN cell to use.
      cell = basic_rnn_ops.make_cell(hparams)

      # There are two ways to construct a variable length RNN in TensorFlow:
      # dynamic_rnn, and state_saving_rnn. The code below demonstrates how to
      # construct an end-to-end samples on disk to labels and logits pipeline
      # for dynamic_rnn.

      # Construct dynamic_rnn reader and inference.

      # Get a batch queue.
      (melody_sequence,
       melody_labels,
       lengths) = basic_rnn_ops.dynamic_rnn_batch(file_list, hparams)

      # Make inference graph. That is, inputs to logits.
      # Note: long sequences need a lot of memory on GPU because all forward
      # pass activations are needed to compute backprop. Additionally
      # multiple steps are computed simultaneously (the parts of each step
      # which don't depend on other steps). The `parallel_iterations`
      # and `swap_memory` arguments given here trade lower GPU memory
      # footprint for speed decrease.
      logits, _, _ = basic_rnn_ops.dynamic_rnn_inference(
          melody_sequence, lengths, cell, hparams, zero_initial_state=True,
          parallel_iterations=1, swap_memory=True)

      # The first hparams.skip_first_n_losses steps of the logits tensor is
      # removed. Those first steps are given to the model as a primer during
      # generation. The model does not get penalized for incorrect
      # predictions in those first steps so the loss does not include those
      # logits.
      truncated_logits = logits[:, hparams.skip_first_n_losses:, :]

      # Reshape logits from [batch_size, sequence_length, one_hot_length] to
      # [batch_size * sequence_length, one_hot_length].
      flat_logits = tf.reshape(truncated_logits,
                               [-1, hparams.one_hot_length])

      # Reshape labels from [batch_size, num_unroll] to
      # [batch_size * sequence_length]. Also truncate first steps to match
      # truncated_logits.
      flat_labels = tf.reshape(
          melody_labels[:, hparams.skip_first_n_losses:], [-1])

      # Compute loss and gradients for training, and accuracy for evaluation.
      cross_entropy, log_perplexity = basic_rnn_ops.log_perplexity_loss(
          flat_logits, flat_labels)
      accuracy = basic_rnn_ops.eval_accuracy(flat_logits, flat_labels)

      global_step = tf.Variable(0, name='global_step', trainable=False)

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('cross_entropy', cross_entropy)
      tf.add_to_collection('log_perplexity', log_perplexity)
      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('global_step', global_step)

      # Compute weight updates, and updates to learning rate and global step.
      if not is_eval_mode:
        training_op, learning_rate = basic_rnn_ops.train_op(
            cross_entropy, global_step, hparams)
        tf.add_to_collection('training_op', training_op)
        tf.add_to_collection('learning_rate', learning_rate)

  return graph


def training_loop(graph, train_dir, num_training_steps=10000,
                  summary_frequency=10, steps_to_average=20):
  """A generator which runs training steps at each output.

  Args:
    graph: A tf.Graph object containing the model.
    train_dir: A string path to the directory to write training checkpoints and
        summary events.
    num_training_steps: Generator terminates after this many steps.
    summary_frequency: How many training iterations to run per generator
        iteration.
    steps_to_average: Average accuracy has a moving window. This is the size of
        that window.

  Yields:
    A dict of training metrics, and runs summary_frequency training steps
    between each yield.
  """
  cross_entropy = graph.get_collection('cross_entropy')[0]
  log_perplexity = graph.get_collection('log_perplexity')[0]
  accuracy = graph.get_collection('accuracy')[0]
  global_step = graph.get_collection('global_step')[0]
  learning_rate = graph.get_collection('learning_rate')[0]
  training_op = graph.get_collection('training_op')[0]

  checkpoint_file = os.path.join(train_dir, 'basic_rnn.ckpt')

  with graph.as_default():
    summary_op = tf.merge_summary([
        tf.scalar_summary('cross_entropy_loss', cross_entropy),
        tf.scalar_summary('log_perplexity', log_perplexity),
        tf.scalar_summary('learning_rate', learning_rate),
        tf.scalar_summary('accuracy', accuracy),
        tf.scalar_summary('global_step', global_step)])

    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()

  # Run training loop.
  session = tf.Session(graph=graph)
  summary_writer = tf.train.SummaryWriter(train_dir, session.graph)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coord)

  step = 0
  gs = 0

  logging.info('Starting training loop')
  try:
    accuracies = collections.deque(maxlen=steps_to_average)
    session.run(init_op)
    while gs < num_training_steps:
      step += 1
      ce, lp, a, gs, lr, serialized_summaries, _ = session.run(
          [cross_entropy, log_perplexity, accuracy, global_step, learning_rate,
           summary_op, training_op])
      summary_writer.add_summary(serialized_summaries, global_step=gs)

      accuracies.append(a)
      if step % summary_frequency == 0:
        saved_path = saver.save(session, checkpoint_file, global_step=gs)
        logging.info('Wrote checkpoint to %s', saved_path)
        summary_writer.flush()
        avg_accuracy = sum(accuracies) / len(accuracies)
        logging.info('Global Step: %s - Loss: %.3f - '
                     'Log-perplexity: %.3f - Step Accuracy: %.2f - '
                     'Avg Accuracy (last %d summaries): %.2f - '
                     'Learning Rate: %f', '{:,}'.format(gs),
                     ce, lp, a, steps_to_average, avg_accuracy, lr)
        yield {'step': step, 'global_step': gs, 'loss': ce,
               'log_perplexity': lp, 'accuracy': a,
               'average_accuracy': avg_accuracy, 'learning_rate': lr}
    saver.save(session, train_dir, global_step=gs)
  except tf.errors.OutOfRangeError as e:
    logging.warn('Got error reported to coordinator: %s', e)
  finally:
    try:
      coord.request_stop()
      summary_writer.close()
    except RuntimeError as e:
      logging.warn('Got runtime error: %s', e)


def eval_loop(graph, eval_dir, train_dir, num_training_steps=10000,
              summary_frequency=10):
  """A generator which runs evaluation steps at each output.

  Args:
    graph: A tf.Graph object containing the model.
    eval_dir: A string path to the directory to write eval summary events.
    train_dir: A string path to the directory to search for checkpoints to eval.
    num_training_steps: Generator terminates after this many steps.
    summary_frequency: How many training iterations to run per generator
        iteration.

  Yields:
    A dict of training metrics, and runs summary_frequency training steps
    between each yield. If no checkpoints are found, None is yielded.
  """
  cross_entropy = graph.get_collection('cross_entropy')[0]
  log_perplexity = graph.get_collection('log_perplexity')[0]
  accuracy = graph.get_collection('accuracy')[0]
  global_step = graph.get_collection('global_step')[0]

  with graph.as_default():
    summary_op = tf.merge_summary([
        tf.scalar_summary('cross_entropy_loss', cross_entropy),
        tf.scalar_summary('log_perplexity', log_perplexity),
        tf.scalar_summary('accuracy', accuracy)])

    saver = tf.train.Saver()
  session = tf.Session(graph=graph)
  summary_writer = tf.train.SummaryWriter(eval_dir, session.graph)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coord)

  gs = 0

  logging.info('Starting eval loop')
  try:
    while gs < num_training_steps:
      checkpoint_path = tf.train.latest_checkpoint(train_dir)
      if not checkpoint_path:
        logging.info('Waiting for checkpoint file in directory %s',
                     train_dir)
        yield
        continue

      saver.restore(session, checkpoint_path)

      ce, lp, a, gs, serialized_summaries = session.run(
          [cross_entropy, log_perplexity, accuracy, global_step, summary_op])

      logging.info('Global Step: %s - Loss: %.3f - Log-perplexity: %.3f - '
                   'Step Accuracy: %.2f', gs, ce, lp, a)

      summary_writer.add_summary(serialized_summaries, global_step=gs)
      summary_writer.flush()

      yield {'loss': ce, 'log_perplexity': lp, 'accuracy': a,
             'global_step': gs}
  except tf.errors.OutOfRangeError as e:
    logging.warn('Got error reported to coordinator: %s', e)
  finally:
    coord.request_stop()
    summary_writer.close()

  coord.join(threads)


def wait_until(time_sec):
  while time.time() < time_sec:
    pass


def main(_):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  graph = make_graph(sequence_example_file=FLAGS.sequence_example_file,
                     hparams_string=FLAGS.hparams, is_eval_mode=FLAGS.eval)

  train_dir = os.path.join(FLAGS.experiment_run_dir, 'train')
  eval_dir = os.path.join(FLAGS.experiment_run_dir, 'eval')
  logging.info('Train dir: %s', train_dir)
  logging.info('Eval dir: %s', eval_dir)

  if FLAGS.eval:
    if not os.path.exists(eval_dir):
      os.makedirs(eval_dir)
  else:
    if not os.path.exists(train_dir):
      os.makedirs(train_dir)

  if FLAGS.eval:
    last_time = time.time()
    for _ in eval_loop(graph, eval_dir, train_dir,
                       num_training_steps=FLAGS.num_training_steps):
      wait_until(last_time + 10.0)
      last_time = time.time()
  else:
    for _ in training_loop(graph, train_dir,
                           num_training_steps=FLAGS.num_training_steps):
      pass


if __name__ == '__main__':
  tf.app.run()

