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
"""Tests for basic_rnn."""

import os
import tensorflow as tf

from magenta.models.basic_rnn import basic_rnn_train

FLAGS = tf.app.flags.FLAGS


def StringsContainSubstrings(strings, substrings):
  """Returns true if every substring is contained in at least one string.

  Looks for any of the substrings in each string. Only requires each substring
  be found once.

  Args:
    strings: List of strings to search for substrings in.
    substrings: List of strings to find in `strings`.

  Returns:
    True if every substring is contained in at least one string.
  """
  needles = set(substrings)
  for s in strings:
    if not needles:
      break
    for substr in needles:
      if substr in s:
        needles.remove(substr)
        break
  if not needles:
    return True
  return False


def StringsExcludeSubstrings(strings, substrings):
  """Returns true if all strings do not contain any substring.

  Args:
    strings: List of strings to search for substrings in.
    substrings: List of strings to find in `strings`.

  Returns:
    True if every substring is not contained in any string.
  """
  for s in strings:
    for substr in substrings:
      if substr in s:
        return False
  return True


class BasicRNNTest(tf.test.TestCase):

  def setUp(self):
    self.sequence_example_file = os.path.join(
        tf.resource_loader.get_data_files_path(), 'testdata',
        'melodies.sample.tfrecord')
    self.train_dir = '/tmp/train_dir'
    self.eval_dir = '/tmp/eval_dir'

  def testDynamicRnnTrainingLoop(self):
    graph = basic_rnn_train.make_graph(
        sequence_example_file=self.sequence_example_file)
    metrics = list(basic_rnn_train.training_loop(
        graph, self.train_dir, num_training_steps=5, summary_frequency=1))

    for metric in metrics:
      self.assertTrue(metric['loss'] >= 0)
      self.assertTrue(metric['accuracy'] >= 0)

  def testEvalLoop(self):
    train_graph = basic_rnn_train.make_graph(
        sequence_example_file=self.sequence_example_file)
    list(basic_rnn_train.training_loop(
        train_graph, self.eval_dir, num_training_steps=5, summary_frequency=1))

    eval_graph = basic_rnn_train.make_graph(
        sequence_example_file=self.sequence_example_file)
    metric = basic_rnn_train.eval_loop(
        eval_graph, self.eval_dir, self.eval_dir,
        num_training_steps=5, summary_frequency=1).next()
    self.assertTrue('loss' in metric)
    self.assertTrue('log_perplexity' in metric)
    self.assertTrue('accuracy' in metric)

  def testDynamicRnnGraphCorrectness(self):
    graph = basic_rnn_train.make_graph(
        sequence_example_file=self.sequence_example_file)
    op_names = [op.name for op in graph.get_operations()]

    self.assertTrue(StringsContainSubstrings(
        op_names,
        ['RNN/MultiRNNCell/Cell0/LSTMCell',
         'RNN/while/MultiRNNCell/Cell0/LSTMCell']))
    self.assertTrue(StringsExcludeSubstrings(
        op_names,
        ['RNN/MultiRNNCell_1/Cell0/LSTMCell', 'InputQueueingStateSaver']))

  def testRnnLayerSize(self):
    hparams = '{"rnn_layer_sizes":[100, 100]}'
    graph = basic_rnn_train.make_graph(
        sequence_example_file=self.sequence_example_file,
        hparams_string=hparams)
    op_names = [op.name for op in graph.get_operations()]
    self.assertTrue(StringsContainSubstrings(
        op_names,
        ['RNN/MultiRNNCell/Cell0/LSTMCell', 'RNN/MultiRNNCell/Cell1/LSTMCell']))


if __name__ == '__main__':
  tf.test.main()