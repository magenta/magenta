# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Performance RNN model."""

# internal imports

import tensorflow as tf
import magenta

from magenta.models.performance_rnn import performance_encoder_decoder
from magenta.models.shared import events_rnn_model


class PerformanceRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN performance generation models."""

  def generate_performance(
      self, num_steps, primer_sequence, temperature=1.0, beam_size=1,
      branch_factor=1, steps_per_iteration=1):
    """Generate a performance track from a primer performance track.

    Args:
      num_steps: The integer length in steps of the final track, after
          generation. Includes the primer.
      primer_sequence: The primer sequence, a Performance object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes tracks more
         random, less than 1.0 makes tracks less random.
      beam_size: An integer, beam size to use when generating tracks via
          beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of steps to take per beam search
          iteration.

    Returns:
      The generated Performance object (which begins with the provided primer
      track).
    """
    return self._generate_events(num_steps, primer_sequence, temperature,
                                 beam_size, branch_factor, steps_per_iteration)

  def performance_log_likelihood(self, sequence):
    """Evaluate the log likelihood of a performance.

    Args:
      sequence: The Performance object for which to evaluate the log likelihood.

    Returns:
      The log likelihood of `sequence` under this model.
    """
    return self._evaluate_log_likelihood([sequence])[0]


class PerformanceRnnConfig(events_rnn_model.EventSequenceRnnConfig):
  """Stores a configuration for a Performance RNN.

  Attributes:
    num_velocity_bins: Number of velocity bins to use. If 0, don't use velocity
        at all.
  """

  def __init__(self, details, encoder_decoder, hparams, num_velocity_bins=0):
    super(PerformanceRnnConfig, self).__init__(
        details, encoder_decoder, hparams)
    self.num_velocity_bins = num_velocity_bins


default_configs = {
    'performance': PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='performance',
            description='Performance RNN'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            performance_encoder_decoder.PerformanceOneHotEncoding()),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001)),

    'performance_with_dynamics': PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='performance_with_dynamics',
            description='Performance RNN with dynamics'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            performance_encoder_decoder.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32),
}
