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
"""Provides function to build the attention RNN model's graph."""

# internal imports
import magenta

from magenta.models.shared import melody_rnn_graph


def default_hparams():
  return magenta.common.HParams(
      batch_size=128,
      rnn_layer_sizes=[128, 128],
      dropout_keep_prob=0.5,
      skip_first_n_losses=0,
      attn_length=40,
      clip_norm=3,
      initial_learning_rate=0.001,
      decay_steps=1000,
      decay_rate=0.97)


def build_graph(mode, hparams_string, encoder_decoder,
                sequence_example_file=None):
  """Builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    hparams_string: A string literal of a Python dictionary, where keys are
        hyperparameter names and values replace default values. For example:
        '{"batch_size":64,"rnn_layer_sizes":[128,128]}'
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
  hparams = default_hparams()
  hparams = hparams.parse(hparams_string)
  return melody_rnn_graph.build_graph(mode, hparams, encoder_decoder,
                                      sequence_example_file)
