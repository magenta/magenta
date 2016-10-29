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
"""Melody RNN model."""

import copy
import heapq

# internal imports

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

import magenta
from magenta.models.melody_rnn import melody_rnn_graph
import magenta.music as mm

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = 0


class MelodyRnnModelException(Exception):
  pass


class MelodyRnnModel(mm.BaseModel):
  """Class for RNN melody generation models.

  Currently this class only supports generation, of both melodies and note
  sequences (containing melodies). Support for model training will be added
  at a later time.
  """

  def __init__(self, config):
    """Initialize the MelodyRnnModel.

    Args:
      config: A MelodyRnnConfig containing the MelodyEncoderDecoder and HParams
        to use.
    """
    super(MelodyRnnModel, self).__init__()
    self._config = config

    # Override hparams for generation.
    # TODO(fjord): once this class supports training, make this step conditional
    # on the usage mode.
    self._config.hparams.dropout_keep_prob = 1.0
    self._config.hparams.batch_size = 1

  def _build_graph_for_generation(self):
    return melody_rnn_graph.build_graph('generate', self._config)

  def _generate_step(self, melodies, inputs, initial_state, temperature):
    """Extends a list of melodies by a single step each."""
    graph_inputs = self._session.graph.get_collection('inputs')[0]
    graph_initial_state = self._session.graph.get_collection('initial_state')[0]
    graph_final_state = self._session.graph.get_collection('final_state')[0]
    graph_softmax = self._session.graph.get_collection('softmax')[0]
    graph_temperature = self._session.graph.get_collection('temperature')

    feed_dict = {graph_inputs: inputs, graph_initial_state: initial_state}
    # For backwards compatibility, we only try to pass temperature if the
    # placeholder exists in the graph.
    if graph_temperature:
      feed_dict[graph_temperature[0]] = temperature
    final_state, softmax = self._session.run(
        [graph_final_state, graph_softmax], feed_dict)
    indices = self._config.encoder_decoder.extend_event_sequences(melodies,
                                                                  softmax)

    return melodies, final_state, softmax[range(len(melodies)), -1, indices]

  def _generate_branches(self, melodies, loglik, branch_factor, num_steps,
                         inputs, initial_state, temperature):
    """Performs a single iteration of branch generation for beam search."""
    all_melodies = []
    all_final_state = np.empty((0, initial_state.shape[1]))
    all_loglik = np.empty(0)

    for _ in range(branch_factor):
      melodies_copy = copy.deepcopy(melodies)
      loglik_copy = copy.deepcopy(loglik)
      for _ in range(num_steps):
        melodies_copy, final_state, softmax = self._generate_step(
            melodies_copy, inputs, initial_state, temperature)
        loglik_copy += np.log(softmax)
      all_melodies += melodies_copy
      all_final_state = np.append(all_final_state, final_state, axis=0)
      all_loglik = np.append(all_loglik, loglik_copy, axis=0)

    return all_melodies, all_final_state, all_loglik

  def _prune_branches(self, melodies, final_state, loglik, k):
    """Prune all but `k` melodies."""
    indices = heapq.nlargest(k, range(len(melodies)), key=lambda i: loglik[i])

    melodies = [melodies[i] for i in indices]
    final_state = final_state[indices, :]
    loglik = loglik[indices]

    return melodies, final_state, loglik

  def _beam_search(self, melody, num_steps, temperature, beam_size,
                   branch_factor, steps_per_iteration):
    """Generates a melody using beam search.

    Initially, the beam is filled with `beam_size` copies of the initial
    melody.

    Each iteration, the beam is pruned to contain only the `beam_size` melodies
    with highest likelihood. Then `branch_factor` new melodies are generated
    for each melody in the beam. These new melodies are formed by extending
    each melody in the beam by `steps_per_iteration` steps. So between a
    branching and a pruning phase, there will be `beam_size` * `branch_factor`
    active melodies.

    Prior to the first "real" iteration, an initial branch generation will take
    place. This is for two reasons:

    1) The RNN model needs to be "primed" with the initial melody.
    2) The desired total number of steps `num_steps` might not be a multiple of
       `steps_per_iteration`, so the initial branching generates melody steps
       such that all subsequent iterations can generate `steps_per_iteration`
       steps.

    After the final iteration, the single melody in the beam with highest
    likelihood will be returned.

    Args:
      melody: The initial melody.
      num_steps: The integer length in steps of the final melody, after
          generation.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes melodies more
         random, less than 1.0 makes melodies less random.
      beam_size: The integer beam size to use.
      branch_factor: The integer branch factor to use.
      steps_per_iteration: The integer number of melody steps to take per
          iteration.

    Returns:
      The highest-likelihood melody as computed by the beam search.
    """
    melodies = [copy.deepcopy(melody) for _ in range(beam_size)]
    graph_initial_state = self._session.graph.get_collection('initial_state')[0]
    loglik = np.zeros(beam_size)

    # Choose the number of steps for the first iteration such that subsequent
    # iterations can all take the same number of steps.
    first_iteration_num_steps = (num_steps - 1) % steps_per_iteration + 1

    inputs = self._config.encoder_decoder.get_inputs_batch(
        melodies, full_length=True)
    initial_state = self._session.run(graph_initial_state)
    melodies, final_state, loglik = self._generate_branches(
        melodies, loglik, branch_factor, first_iteration_num_steps, inputs,
        initial_state, temperature)

    num_iterations = (num_steps -
                      first_iteration_num_steps) / steps_per_iteration

    for _ in range(num_iterations):
      melodies, final_state, loglik = self._prune_branches(
          melodies, final_state, loglik, k=beam_size)
      inputs = self._config.encoder_decoder.get_inputs_batch(melodies)
      melodies, final_state, loglik = self._generate_branches(
          melodies, loglik, branch_factor, steps_per_iteration, inputs,
          final_state, temperature)

    # Prune to a single melody.
    melodies, final_state, loglik = self._prune_branches(
        melodies, final_state, loglik, k=1)

    tf.logging.info('Beam search yields melody with log-likelihood: %f ',
                    loglik[0])

    return melodies[0]

  def generate_melody(self, num_steps, primer_melody, temperature=1.0,
                      beam_size=1, branch_factor=1, steps_per_iteration=1):
    """Generate a melody from a primer melody.

    Args:
      num_steps: The integer length in steps of the final melody, after
          generation. Includes the primer.
      primer_melody: The primer melody, a Melody object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes melodies more
         random, less than 1.0 makes melodies less random.
      beam_size: An integer, beam size to use when generating melodies via beam
          search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of melody steps to take per beam
          search iteration.

    Returns:
      The generated Melody object (which begins with the provided primer
          melody).

    Raises:
      MelodyRnnModelException: If the primer melody has zero
          length or is not shorter than num_steps.
    """
    if not primer_melody:
      raise MelodyRnnModelException(
          'primer melody must have non-zero length')
    if len(primer_melody) >= num_steps:
      raise MelodyRnnModelException(
          'primer melody must be shorter than `num_steps`')

    if beam_size != self._config.hparams.batch_size:
      raise MelodyRnnModelException(
          'currently beam search only supports using batch size as beam size')

    melody = copy.deepcopy(primer_melody)

    transpose_amount = melody.squash(
        self._config.min_note,
        self._config.max_note,
        self._config.transpose_to_key)

    if num_steps > len(melody):
      melody = self._beam_search(melody, num_steps - len(melody), temperature,
                                 beam_size, branch_factor, steps_per_iteration)

    melody.transpose(-transpose_amount)

    return melody


class MelodyRnnConfig(object):
  """Stores a configuration for a MelodyRnn.

  You can change `min_note` and `max_note` to increase/decrease the melody
  range. Since melodies are transposed into this range to be run through
  the model and then transposed back into their original range after the
  melodies have been extended, the location of the range is somewhat
  arbitrary, but the size of the range determines the possible size of the
  generated melodies range. `transpose_to_key` should be set to the key
  that if melodies were transposed into that key, they would best sit
  between `min_note` and `max_note` with having as few notes outside that
  range.

  Attributes:
    details: The GeneratorDetails message describing the config.
    encoder_decoder: The EventSequenceEncoderDecoder object to use.
    hparams: The HParams containing hyperparameters to use.
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.
    transpose_to_key: The key that encoded melodies will be transposed into, or
        None if it should not be transposed.
  """

  def __init__(self, details, encoder_decoder, hparams,
               min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE,
               transpose_to_key=DEFAULT_TRANSPOSE_TO_KEY):
    if min_note < mm.MIN_MIDI_PITCH:
      raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > mm.MAX_MIDI_PITCH + 1:
      raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note - min_note < mm.NOTES_PER_OCTAVE:
      raise ValueError('max_note - min_note must be >= 12. min_note is %d. '
                       'max_note is %d. max_note - min_note is %d.' %
                       (min_note, max_note, max_note - min_note))
    if (transpose_to_key is not None and
        (transpose_to_key < 0 or transpose_to_key > mm.NOTES_PER_OCTAVE - 1)):
      raise ValueError('transpose_to_key must be >= 0 and <= 11. '
                       'transpose_to_key is %d.' % transpose_to_key)

    self.details = details
    self.encoder_decoder = encoder_decoder
    self.hparams = hparams
    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key


# Default configurations.
default_configs = {
    'basic_rnn': MelodyRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='basic_rnn',
            description='Melody RNN with one-hot encoding.'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.MelodyOneHotEncoding(
                min_note=DEFAULT_MIN_NOTE,
                max_note=DEFAULT_MAX_NOTE)),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            clip_norm=5,
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.85)),
    'lookback_rnn': MelodyRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='lookback_rnn',
            description='Melody RNN with lookback encoding.'),
        magenta.music.LookbackEventSequenceEncoderDecoder(
            magenta.music.MelodyOneHotEncoding(
                min_note=DEFAULT_MIN_NOTE,
                max_note=DEFAULT_MAX_NOTE)),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            clip_norm=5,
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.95)),
    'attention_rnn': MelodyRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='attention_rnn',
            description='Melody RNN with lookback encoding and attention.'),
        magenta.music.KeyMelodyEncoderDecoder(
            min_note=DEFAULT_MIN_NOTE,
            max_note=DEFAULT_MAX_NOTE),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            attn_length=40,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.97))
}
