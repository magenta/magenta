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

"""Polyphonic RNN model."""

from magenta.contrib import training as contrib_training
from magenta.models.polyphony_rnn import polyphony_encoder_decoder
from magenta.models.shared import events_rnn_model
import note_seq
from note_seq.protobuf import generator_pb2


class PolyphonyRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN polyphonic sequence generation models."""

  def generate_polyphonic_sequence(
      self, num_steps, primer_sequence, temperature=1.0, beam_size=1,
      branch_factor=1, steps_per_iteration=1, modify_events_callback=None):
    """Generate a polyphonic track from a primer polyphonic track.

    Args:
      num_steps: The integer length in steps of the final track, after
          generation. Includes the primer.
      primer_sequence: The primer sequence, a PolyphonicSequence object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes tracks more
         random, less than 1.0 makes tracks less random.
      beam_size: An integer, beam size to use when generating tracks via
          beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of steps to take per beam search
          iteration.
      modify_events_callback: An optional callback for modifying the event list.
          Can be used to inject events rather than having them generated. If not
          None, will be called with 3 arguments after every event: the current
          EventSequenceEncoderDecoder, a list of current EventSequences, and a
          list of current encoded event inputs.
    Returns:
      The generated PolyphonicSequence object (which begins with the provided
      primer track).
    """
    return self._generate_events(num_steps, primer_sequence, temperature,
                                 beam_size, branch_factor, steps_per_iteration,
                                 modify_events_callback=modify_events_callback)

  def polyphonic_sequence_log_likelihood(self, sequence):
    """Evaluate the log likelihood of a polyphonic sequence.

    Args:
      sequence: The PolyphonicSequence object for which to evaluate the log
          likelihood.

    Returns:
      The log likelihood of `sequence` under this model.
    """
    return self._evaluate_log_likelihood([sequence])[0]


default_configs = {
    'polyphony':
        events_rnn_model.EventSequenceRnnConfig(
            generator_pb2.GeneratorDetails(
                id='polyphony', description='Polyphonic RNN'),
            note_seq.OneHotEventSequenceEncoderDecoder(
                polyphony_encoder_decoder.PolyphonyOneHotEncoding()),
            contrib_training.HParams(
                batch_size=64,
                rnn_layer_sizes=[256, 256, 256],
                dropout_keep_prob=0.5,
                clip_norm=5,
                learning_rate=0.001)),
}
