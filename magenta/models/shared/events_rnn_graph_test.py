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
"""Tests for events_rnn_graph."""

import tempfile

# internal imports
import tensorflow as tf
import magenta

from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_model


class EventSequenceRNNGraphTest(tf.test.TestCase):

  def setUp(self):
    self._sequence_file = tempfile.NamedTemporaryFile(
        prefix='EventSequenceRNNGraphTest')

    self.config = events_rnn_model.EventSequenceRnnConfig(
        None,
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.testing_lib.TrivialOneHotEncoding(12)),
        tf.contrib.training.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            clip_norm=5,
            learning_rate=0.01))

  def testBuildTrainGraph(self):
    g = events_rnn_graph.build_graph(
        'train', self.config,
        sequence_example_file_paths=[self._sequence_file.name])
    self.assertTrue(isinstance(g, tf.Graph))

  def testBuildEvalGraph(self):
    g = events_rnn_graph.build_graph(
        'eval', self.config,
        sequence_example_file_paths=[self._sequence_file.name])
    self.assertTrue(isinstance(g, tf.Graph))

  def testBuildGenerateGraph(self):
    g = events_rnn_graph.build_graph('generate', self.config)
    self.assertTrue(isinstance(g, tf.Graph))

  def testBuildGraphWithAttention(self):
    self.config.hparams.attn_length = 10
    g = events_rnn_graph.build_graph(
        'train', self.config,
        sequence_example_file_paths=[self._sequence_file.name])
    self.assertTrue(isinstance(g, tf.Graph))


if __name__ == '__main__':
  tf.test.main()
