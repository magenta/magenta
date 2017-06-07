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

# internal imports
import tensorflow as tf


def run_training(graph, train_dir, num_training_steps=None,
                 summary_frequency=10, save_checkpoint_secs=60,
                 checkpoints_to_keep=10):
  """Runs the training loop.

  Args:
    graph: A tf.Graph object containing the model.
    train_dir: The path to the directory where checkpoints and summary events
        will be written to.
    num_training_steps: The number of steps to train for before exiting.
    summary_frequency: The number of steps between each summary. A summary is
        when graph values from the last step are logged to the console and
        written to disk.
    save_checkpoint_secs: The frequency at which to save checkpoints, in
        seconds.
    checkpoints_to_keep: The number of most recent checkpoints to keep in
       `train_dir`. Keeps all if set to 0.
  """
  with graph.as_default():
    global_step = tf.train.get_or_create_global_step()
    loss = tf.get_collection('loss')[0]
    perplexity = tf.get_collection('metrics/perplexity')[0]
    accuracy = tf.get_collection('metrics/accuracy')[0]
    train_op = tf.get_collection('train_op')[0]

    logging_dict = {
        'Global Step': global_step,
        'Loss': loss,
        'Perplexity': perplexity,
        'Accuracy': accuracy
    }
    hooks = [
        tf.train.NanTensorHook(loss),
        tf.train.LoggingTensorHook(
            logging_dict, every_n_iter=summary_frequency),
        tf.train.StepCounterHook(
            output_dir=train_dir, every_n_steps=summary_frequency)
    ]
    if num_training_steps:
      hooks.append(tf.train.StopAtStepHook(num_training_steps))

    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(max_to_keep=checkpoints_to_keep))

    tf.logging.info('Starting training loop...')
    tf.contrib.training.train(
        train_op=train_op,
        logdir=train_dir,
        scaffold=scaffold,
        hooks=hooks,
        save_checkpoint_secs=save_checkpoint_secs,
        save_summaries_steps=summary_frequency)
    tf.logging.info('Training complete.')


# TODO(adarob): Limit to a single epoch each evaluation step.
def run_eval(graph, train_dir, eval_dir, num_batches, timeout_secs=300):
  """Runs the training loop.

  Args:
    graph: A tf.Graph object containing the model.
    train_dir: The path to the directory where checkpoints will be loaded
        from for evaluation.
    eval_dir: The path to the directory where the evaluation summary events
        will be written to.
    num_batches: The number of full batches to use for each evaluation step.
    timeout_secs: The number of seconds after which to stop waiting for a new
        checkpoint.
  """
  with graph.as_default():
    global_step = tf.train.get_or_create_global_step()
    loss = tf.get_collection('loss')[0]
    perplexity = tf.get_collection('metrics/perplexity')[0]
    accuracy = tf.get_collection('metrics/accuracy')[0]
    eval_ops = tf.get_collection('eval_ops')

    logging_dict = {
        'Global Step': global_step,
        'Loss': loss,
        'Perplexity': perplexity,
        'Accuracy': accuracy
    }
    hooks = [
        EvalLoggingTensorHook(logging_dict, every_n_iter=num_batches),
        tf.contrib.training.StopAfterNEvalsHook(num_batches),
        tf.contrib.training.SummaryAtEndHook(eval_dir),
    ]

    tf.contrib.training.evaluate_repeatedly(
        train_dir,
        eval_ops=eval_ops,
        hooks=hooks,
        eval_interval_secs=60,
        timeout=timeout_secs)


class EvalLoggingTensorHook(tf.train.LoggingTensorHook):
  """A revised version of LoggingTensorHook to use during evaluation.

  This version supports being reset and increments `_iter_count` before run
  instead of after run.
  """

  def begin(self):
    # Reset timer.
    self._timer.update_last_triggered_step(0)
    super(EvalLoggingTensorHook, self).begin()

  def before_run(self, run_context):
    self._iter_count += 1
    return super(EvalLoggingTensorHook, self).before_run(run_context)

  def after_run(self, run_context, run_values):
    super(EvalLoggingTensorHook, self).after_run(run_context, run_values)
    self._iter_count -= 1
