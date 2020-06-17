# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for events_rnn_graph."""

import tempfile

from magenta.contrib import training as contrib_training
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_model
import note_seq
from note_seq import testing_lib
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class EventSequenceRNNGraphTest(tf.test.TestCase):

  def setUp(self):
    self._sequence_file = tempfile.NamedTemporaryFile(
        prefix='EventSequenceRNNGraphTest')

    self.config = events_rnn_model.EventSequenceRnnConfig(
        None,
        note_seq.OneHotEventSequenceEncoderDecoder(
            testing_lib.TrivialOneHotEncoding(12)),
        contrib_training.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            clip_norm=5,
            learning_rate=0.01))

  def testBuildTrainGraph(self):
    with tf.Graph().as_default():
      events_rnn_graph.get_build_graph_fn(
          'train', self.config,
          sequence_example_file_paths=[self._sequence_file.name])()

  def testBuildEvalGraph(self):
    with tf.Graph().as_default():
      events_rnn_graph.get_build_graph_fn(
          'eval', self.config,
          sequence_example_file_paths=[self._sequence_file.name])()

  def testBuildGenerateGraph(self):
    with tf.Graph().as_default():
      events_rnn_graph.get_build_graph_fn('generate', self.config)()

  def testBuildGraphWithAttention(self):
    self.config.hparams.attn_length = 10
    with tf.Graph().as_default():
      events_rnn_graph.get_build_graph_fn(
          'train', self.config,
          sequence_example_file_paths=[self._sequence_file.name])()


if __name__ == '__main__':
  tf.test.main()
