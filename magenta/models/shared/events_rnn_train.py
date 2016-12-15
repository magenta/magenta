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
"""Train and evaluate an event sequence RNN model."""

import time

# internal imports
import tensorflow as tf


def run_training(graph, train_dir, num_training_steps=None,
                 summary_frequency=10, should_stop_func=None):
  """Runs the training loop.

  Args:
    graph: A tf.Graph object containing the model.
    train_dir: The path to the directory where checkpoints and summary events
        will be written to.
    num_training_steps: The number of steps to train for before exiting.
    summary_frequency: The number of steps between each summary. A summary is
        when graph values from the last step are logged to the console.
    should_stop_func: An optional external function to act as a stop signal.
        The function is expected to return a Boolean specifying whether or not
        the training should stop.
  """
  global_step = graph.get_collection('global_step')[0]
  learning_rate = graph.get_collection('learning_rate')[0]
  loss = graph.get_collection('loss')[0]
  perplexity = graph.get_collection('perplexity')[0]
  accuracy = graph.get_collection('accuracy')[0]
  train_op = graph.get_collection('train_op')[0]

  sv = tf.train.Supervisor(graph=graph, logdir=train_dir, save_model_secs=30,
                           global_step=global_step)

  with sv.managed_session() as sess:
    global_step_ = sess.run(global_step)
    if num_training_steps and global_step_ >= num_training_steps:
      tf.logging.info('This checkpoint\'s global_step value is already %d, '
                      'which is greater or equal to the specified '
                      'num_training_steps value of %d. Exiting training.',
                      global_step_, num_training_steps)
      return
    tf.logging.info('Starting training loop...')
    while not num_training_steps or global_step_ < num_training_steps:
      if sv.should_stop():
        break
      if should_stop_func and should_stop_func():
        break
      if (global_step_ + 1) % summary_frequency == 0:
        (global_step_, learning_rate_, loss_, perplexity_, accuracy_,
         _) = sess.run([global_step, learning_rate, loss, perplexity, accuracy,
                        train_op])
        tf.logging.info('Global Step: %d - '
                        'Learning Rate: %.5f - '
                        'Loss: %.3f - '
                        'Perplexity: %.3f - '
                        'Accuracy: %.3f',
                        global_step_, learning_rate_, loss_, perplexity_,
                        accuracy_)
      else:
        global_step_, _ = sess.run([global_step, train_op])
    sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
    tf.logging.info('Training complete.')


def run_eval(graph, train_dir, eval_dir, num_training_steps=None,
             summary_frequency=10, should_stop_func=None):
  """Runs the training loop.

  Args:
    graph: A tf.Graph object containing the model.
    train_dir: The path to the directory where checkpoints will be loaded
        from for evaluation.
    eval_dir: The path to the directory where the evaluation summary events
        will be written to.
    num_training_steps: When the `global_step` from latest checkpoint loaded
        from for `train_dir` has reached `num_training_steps`, the evaluation
        loop will be stopped.
    summary_frequency: The number of seconds between each summary. A summary is
        when evaluation data is logged to the console and evaluation
        summary events are written to `eval_dir`.
    should_stop_func: An optional external function to act as a stop signal.
        When given the loss and global step, the function is expected to return
        a Boolean specifying whether or not the eval should stop.
  """
  global_step = graph.get_collection('global_step')[0]
  loss = graph.get_collection('loss')[0]
  perplexity = graph.get_collection('perplexity')[0]
  accuracy = graph.get_collection('accuracy')[0]
  summary_op = graph.get_collection('summary_op')[0]

  with graph.as_default():
    saver = tf.train.Saver()
    with tf.Session() as sess:
      summary_writer = tf.train.SummaryWriter(eval_dir, sess.graph)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      global_step_ = 0
      last_global_step = None
      tf.logging.info('Starting eval loop...')
      try:
        while not num_training_steps or global_step_ < num_training_steps:
          checkpoint_path = tf.train.latest_checkpoint(train_dir)
          if not checkpoint_path:
            tf.logging.info('Waiting for checkpoint file in directory %s.',
                            train_dir)
          else:
            saver.restore(sess, checkpoint_path)

            global_step_, loss_, perplexity_, accuracy_, summary_op_ = sess.run(
                [global_step, loss, perplexity, accuracy, summary_op])

            tf.logging.info('Global Step: %d - '
                            'Loss: %.3f - '
                            'Perplexity: %.3f - '
                            'Accuracy: %.3f',
                            global_step_, loss_, perplexity_, accuracy_)
            if should_stop_func and should_stop_func(loss_, global_step_):
              break

            if global_step_ != last_global_step:
              summary_writer.add_summary(summary_op_, global_step=global_step_)
              summary_writer.flush()
              last_global_step = global_step_

          time.sleep(summary_frequency)

      except tf.errors.OutOfRangeError as e:
        tf.logging.warn('Got error reported to coordinator: %s', e)
      finally:
        coord.request_stop()
        summary_writer.close()

      coord.join(threads)
