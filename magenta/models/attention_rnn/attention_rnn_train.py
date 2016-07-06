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
"""Train and evaluate the attention RNN model."""

import logging
import attention_rnn_encoder_decoder
import attention_rnn_graph
import os
import sys
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('eval', False,
                            'If true, this process evaluates the model and '
                            'does not update weights.')
tf.app.flags.DEFINE_string('sequence_example_file', '',
                           'Path to TFRecord file containing '
                           'tf.SequenceExample records for training or '
                           'evaluation.')
tf.app.flags.DEFINE_string('run_dir', '/tmp/attention_rnn/run1',
                           'Path to the directory where checkpoints and '
                           'summary events will be saved during training and '
                           'evaluation, or loaded from during generation. '
                           'Separate subdirectories for training events and '
                           'eval events will be created within `run_dir`. '
                           'Multiple runs can be stored within the '
                           'parent directory of `run_dir`. Point TensorBoard '
                           'to the parent directory of `run_dir` to see all '
                           'your runs.')
tf.app.flags.DEFINE_string('hparams', '{}',
                           'String representation of Python dictionary '
                           'containing hyperparameter to value mapping. This '
                           'mapping is merged with the default '
                           'hyperparameters.')
tf.app.flags.DEFINE_integer('num_training_steps', 10000,
                            'The the number of global training steps your '
                            'model should take before exiting training. '
                            'During evaluation, the eval loop will run until '
                            'the `global_step` parameter of the model being '
                            'evaluated has reached `num_training_steps`.')
tf.app.flags.DEFINE_integer('summary_frequency', 10,
                            'A summary statement will be logged every '
                            '`summary_frequency` steps during training or '
                            'every `summary_frequency` seconds during '
                            'evaluation.')


def run_training(graph, train_dir, num_training_steps=10000,
                 summary_frequency=10):
  """Runs the training loop.

  Args:
    graph: A tf.Graph object containing the model.
    train_dir: The path to the directory where checkpoints and summary events
        will be written to.
    num_training_steps: The number of steps to train for before exiting.
    summary_frequency: The number of steps between each summary. A summary is
        when graph values from the last step are logged to the console.
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
    _global_step = sess.run(global_step)
    if _global_step >= num_training_steps:
      logging.info('This checkpoint\'s global_step value is already %d, '
                   'which is greater or equal to the specified '
                   'num_training_steps value of %d. Exiting training.',
                   _global_step, num_training_steps)
      return
    logging.info('Starting training loop...')
    while _global_step < num_training_steps:
      if sv.should_stop():
        break
      if (_global_step + 1) % summary_frequency == 0:
        (_global_step, _learning_rate, _loss, _perplexity, _accuracy, _
         ) = sess.run([global_step, learning_rate, loss, perplexity,
                       accuracy, train_op])
        logging.info('Global Step: %d - '
                     'Learning Rate: %.5f - '
                     'Loss: %.3f - '
                     'Perplexity: %.3f - '
                     'Accuracy: %.3f',
                     _global_step, _learning_rate, _loss, _perplexity,
                     _accuracy)
      else:
        _global_step, _ = sess.run([global_step, train_op])
    sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
    logging.info('Training complete.')


def run_eval(graph, train_dir, eval_dir, num_training_steps=10000,
             summary_frequency=10):
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
      _global_step = 0
      last_global_step = None
      logging.info('Starting eval loop...')
      try:
        while _global_step < num_training_steps:
          checkpoint_path = tf.train.latest_checkpoint(train_dir)
          if not checkpoint_path:
            logging.info('Waiting for checkpoint file in directory %s.',
                         train_dir)
          else:
            saver.restore(sess, checkpoint_path)

            _global_step, _loss, _perplexity, _accuracy, _summary_op = sess.run(
                [global_step, loss, perplexity, accuracy, summary_op])

            logging.info('Global Step: %d - '
                         'Loss: %.3f - '
                         'Perplexity: %.3f - '
                         'Accuracy: %.3f',
                         _global_step, _loss, _perplexity, _accuracy)

            if _global_step != last_global_step:
              summary_writer.add_summary(_summary_op, global_step=_global_step)
              summary_writer.flush()
              last_global_step = _global_step

          time.sleep(summary_frequency)

      except tf.errors.OutOfRangeError as e:
        logging.warn('Got error reported to coordinator: %s', e)
      finally:
        coord.request_stop()
        summary_writer.close()

      coord.join(threads)


def main(_):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  melody_encoder_decoder = attention_rnn_encoder_decoder.MelodyEncoderDecoder()

  mode = 'eval' if FLAGS.eval else 'train'
  graph = attention_rnn_graph.build_graph(mode,
                                         FLAGS.hparams,
                                         melody_encoder_decoder.input_size,
                                         melody_encoder_decoder.num_classes,
                                         FLAGS.sequence_example_file)

  train_dir = os.path.join(FLAGS.run_dir, 'train')
  if not os.path.exists(train_dir):
    os.makedirs(train_dir)
  logging.info('Train dir: %s', train_dir)

  if FLAGS.eval:
    eval_dir = os.path.join(FLAGS.run_dir, 'eval')
    if not os.path.exists(eval_dir):
      os.makedirs(eval_dir)
    logging.info('Eval dir: %s', eval_dir)
    run_eval(graph, train_dir, eval_dir, FLAGS.num_training_steps,
             FLAGS.summary_frequency)

  else:
    run_training(graph, train_dir, FLAGS.num_training_steps,
                 FLAGS.summary_frequency)


if __name__ == '__main__':
  tf.app.run()
