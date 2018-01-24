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
"""RNN-NADE model."""

# internal imports
import tensorflow as tf

import magenta
from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_graph
from magenta.models.shared import events_rnn_model
import magenta.music as mm


class PianorollRnnNadeModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN-NADE sequence generation models."""

  def _build_graph_for_generation(self):
    return pianoroll_rnn_nade_graph.get_build_graph_fn(
        'generate', self._config)()

  def _generate_step_for_batch(self, pianoroll_sequences, inputs, initial_state,
                               temperature):
    """Extends a batch of event sequences by a single step each.

    This method modifies the event sequences in place.

    Args:
      pianoroll_sequences: A list of PianorollSequences. The list of event
          sequences should have length equal to `self._batch_size()`.
      inputs: A Python list of model inputs, with length equal to
          `self._batch_size()`.
      initial_state: A numpy array containing the initial RNN-NADE state, where
          `initial_state.shape[0]` is equal to `self._batch_size()`.
      temperature: Unused.

    Returns:
      final_state: The final RNN-NADE state, the same size as `initial_state`.
      loglik: The log-likelihood of the sampled value for each event
          sequence, a 1-D numpy array of length
          `self._batch_size()`. If `inputs` is a full-length inputs batch, the
          log-likelihood of each entire sequence up to and including the
          generated step will be computed and returned.
    """
    assert len(pianoroll_sequences) == self._batch_size()

    graph_inputs = self._session.graph.get_collection('inputs')[0]
    graph_initial_state = tuple(
        self._session.graph.get_collection('initial_state'))
    graph_final_state = tuple(
        self._session.graph.get_collection('final_state'))
    graph_sample = self._session.graph.get_collection('sample')[0]
    graph_log_prob = self._session.graph.get_collection('log_prob')[0]

    sample, loglik, final_state = self._session.run(
        [graph_sample, graph_log_prob, graph_final_state],
        {
            graph_inputs: inputs,
            graph_initial_state: initial_state,
        })

    self._config.encoder_decoder.extend_event_sequences(
        pianoroll_sequences, sample)

    return final_state, loglik[:, 0]

  def generate_pianoroll_sequence(
      self, num_steps, primer_sequence, beam_size=1, branch_factor=1,
      steps_per_iteration=1):
    """Generate a pianoroll track from a primer pianoroll track.

    Args:
      num_steps: The integer length in steps of the final track, after
          generation. Includes the primer.
      primer_sequence: The primer sequence, a PianorollSequence object.
      beam_size: An integer, beam size to use when generating tracks via
          beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: The number of steps to take per beam search
          iteration.
    Returns:
      The generated PianorollSequence object (which begins with the provided
      primer track).
    """
    return self._generate_events(
        num_steps=num_steps, primer_events=primer_sequence, temperature=None,
        beam_size=beam_size, branch_factor=branch_factor,
        steps_per_iteration=steps_per_iteration)


default_configs = {
    'rnn-nade': events_rnn_model.EventSequenceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='rnn-nade',
            description='RNN-NADE'),
        mm.PianorollEncoderDecoder(),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[128, 128, 128],
            nade_hidden_units=128,
            dropout_keep_prob=0.5,
            clip_norm=5,
            learning_rate=0.001)),
    'rnn-nade_attn': events_rnn_model.EventSequenceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='rnn-nade_attn',
            description='RNN-NADE with attention.'),
        mm.PianorollEncoderDecoder(),
        tf.contrib.training.HParams(
            batch_size=48,
            rnn_layer_sizes=[128, 128],
            attn_length=32,
            nade_hidden_units=128,
            dropout_keep_prob=0.5,
            clip_norm=5,
            learning_rate=0.001)),
}
