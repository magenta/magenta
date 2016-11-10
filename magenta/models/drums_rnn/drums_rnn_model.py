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
"""Drums RNN model."""

import copy
import heapq

# internal imports

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

import magenta
import magenta.music as mm

from magenta.models.drums_rnn import drums_rnn_graph


class DrumsRnnModelException(Exception):
  pass


class DrumsRnnModel(mm.BaseModel):
  """Class for RNN drum track generation models.

  Currently this class only supports generation, of both drum tracks and note
  sequences (containing drum tracks). Support for model training will be added
  at a later time.
  """

  def __init__(self, config):
    """Initialize the DrumsRnnModel.

    Args:
      config: A DrumsRnnConfig containing the encoder/decoder and HParams
        to use.
    """
    super(DrumsRnnModel, self).__init__()
    self._config = config

    # Override hparams for generation.
    # TODO(fjord): once this class supports training, make this step conditional
    # on the usage mode.
    self._config.hparams.dropout_keep_prob = 1.0
    self._config.hparams.batch_size = 1

  def _build_graph_for_generation(self):
    return drums_rnn_graph.build_graph('generate', self._config)

  def _generate_step_for_batch(self, drum_tracks, inputs, initial_state,
                               temperature):
    """Extends a batch of drum tracks by a single step each.

    This method modifies the drum tracks in place.

    Args:
      drum_tracks: A list of DrumTrack objects. The list should have length
          equal to `self._config.hparams.batch_size`.
      inputs: A Python list of model inputs, with length equal to
          `self._config.hparams.batch_size`.
      initial_state: A numpy array containing the initial RNN state, where
          `initial_state.shape[0]` is equal to
          `self._config.hparams.batch_size`.
      temperature: The softmax temperature.

    Returns:
      final_state: The final RNN state, a numpy array the same size as
          `initial_state`.
      softmax: The chosen softmax value for each drum track, a 1-D numpy array
          of length `self._config.hparams.batch_size`.
    """
    assert len(drum_tracks) == self._config.hparams.batch_size

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
    indices = self._config.encoder_decoder.extend_event_sequences(drum_tracks,
                                                                  softmax)

    return final_state, softmax[range(len(drum_tracks)), -1, indices]

  def _generate_step(self, drum_tracks, inputs, initial_state, temperature):
    """Extends a list of drum tracks by a single step each.

    This method modifies the drum tracks in place.

    Args:
      drum_tracks: A list of DrumTrack objects.
      inputs: A Python list of model inputs, with length equal to the number of
          drum tracks.
      initial_state: A numpy array containing the initial RNN states, where
          `initial_state.shape[0]` is equal to the number of drum tracks.
      temperature: The softmax temperature.

    Returns:
      final_state: The final RNN state, a numpy array the same size as
          `initial_state`.
      softmax: The chosen softmax value for each drum track, a 1-D numpy array
          the same length as `drum_tracks`.
    """
    batch_size = self._config.hparams.batch_size
    num_full_batches = len(drum_tracks) / batch_size

    final_state = np.empty((len(drum_tracks), initial_state.shape[1]))
    softmax = np.empty(len(drum_tracks))

    offset = 0
    for _ in range(num_full_batches):
      # Generate a single step for one batch of drum tracks.
      batch_indices = range(offset, offset + batch_size)
      batch_final_state, batch_softmax = self._generate_step_for_batch(
          [drum_tracks[i] for i in batch_indices],
          [inputs[i] for i in batch_indices],
          initial_state[batch_indices, :],
          temperature)
      final_state[batch_indices, :] = batch_final_state
      softmax[batch_indices] = batch_softmax
      offset += batch_size

    if offset < len(drum_tracks):
      # There's an extra non-full batch. Pad it with a bunch of copies of the
      # final drum track.
      num_extra = len(drum_tracks) - offset
      pad_size = batch_size - num_extra
      batch_indices = range(offset, len(drum_tracks))
      batch_final_state, batch_softmax = self._generate_step_for_batch(
          [drum_tracks[i] for i in batch_indices] +
          [copy.deepcopy(drum_tracks[-1]) for _ in range(pad_size)],
          [inputs[i] for i in batch_indices] + inputs[-1] * pad_size,
          np.append(initial_state[batch_indices, :],
                    np.tile(inputs[-1, :], (pad_size, 1)),
                    axis=0),
          temperature)
      final_state[batch_indices] = batch_final_state[0:num_extra, :]
      softmax[batch_indices] = batch_softmax[0:num_extra]

    return final_state, softmax

  def _generate_branches(self, drum_tracks, loglik, branch_factor, num_steps,
                         inputs, initial_state, temperature):
    """Performs a single iteration of branch generation for beam search.

    This method generates `branch_factor` branches for each drum track in
    `drum_tracks`, where each branch extends the drum track by `num_steps`
    steps.

    Args:
      drum_tracks: A list of DrumTrack objects.
      loglik: A 1-D numpy array of drum track log-likelihoods, the same size as
          `drum_tracks`.
      branch_factor: The integer branch factor to use.
      num_steps: The integer number of steps to take per branch.
      inputs: A Python list of model inputs, with length equal to the number of
          drum tracks.
      initial_state: A numpy array containing the initial RNN states, where
          `initial_state.shape[0]` is equal to the number of drum tracks.
      temperature: The softmax temperature.

    Returns:
      all_drum_tracks: A list of Melody objects, with `branch_factor` times as
          many drum tracks as the initial list of drum tracks.
      all_final_state: A numpy array of final RNN states, where
          `final_state.shape[0]` is equal to the length of `all_drum_tracks`.
      all_loglik: A 1-D numpy array of log-likelihoods, with length equal to the
          length of `all_drum_tracks`.
    """
    all_drum_tracks = [copy.deepcopy(drum_track)
                       for drum_track in drum_tracks * branch_factor]
    all_inputs = inputs * branch_factor
    all_final_state = np.tile(initial_state, (branch_factor, 1))
    all_loglik = np.tile(loglik, (branch_factor,))

    for _ in range(num_steps):
      all_final_state, all_softmax = self._generate_step(
          all_drum_tracks, all_inputs, all_final_state, temperature)
      all_loglik += np.log(all_softmax)

    return all_drum_tracks, all_final_state, all_loglik

  def _prune_branches(self, drum_tracks, final_state, loglik, k):
    """Prune all but `k` drum tracks.

    This method prunes all but the `k` drum_tracks with highest log-likelihood.

    Args:
      drum_tracks: A list of DrumTrack objects.
      final_state: A numpy array containing the final RNN states, where
          `final_state.shape[0]` is equal to the number of drum tracks.
      loglik: A 1-D numpy array of log-likelihoods, the same size as
          `drum_tracks`.
      k: The number of drum tracks to keep after pruning.

    Returns:
      drum_tracks: The pruned list of DrumTrack objects, of length `k`.
      final_state: The pruned numpy array of final RNN states, where
          `final_state.shape[0]` is equal to `k`.
      loglik: The pruned log-likelihoods, a 1-D numpy array of length `k`.
    """
    indices = heapq.nlargest(k, range(len(drum_tracks)),
                             key=lambda i: loglik[i])

    drum_tracks = [drum_tracks[i] for i in indices]
    final_state = final_state[indices, :]
    loglik = loglik[indices]

    return drum_tracks, final_state, loglik

  def _beam_search(self, drums, num_steps, temperature, beam_size,
                   branch_factor, steps_per_iteration):
    """Generates a drum track using beam search.

    Initially, the beam is filled with `beam_size` copies of the initial drum
    track.

    Each iteration, the beam is pruned to contain only the `beam_size` drum
    tracks with highest likelihood. Then `branch_factor` new drum tracks are
    generated for each drum track in the beam. These new drum tracks are formed
    by extending each drum track in the beam by `steps_per_iteration` steps. So
    between a branching and a pruning phase, there will be `beam_size` *
    `branch_factor` active drum tracks.

    Prior to the first "real" iteration, an initial branch generation will take
    place. This is for two reasons:

    1) The RNN model needs to be "primed" with the initial drum track.
    2) The desired total number of steps `num_steps` might not be a multiple of
       `steps_per_iteration`, so the initial branching generates steps such that
       all subsequent iterations can generate `steps_per_iteration` steps.

    After the final iteration, the single drum track in the beam with highest
    likelihood will be returned.

    Args:
      drums: The initial drum track.
      num_steps: The integer length in steps of the final drum track, after
          generation.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes drum tracks more
         random, less than 1.0 makes drum tracks less random.
      beam_size: The integer beam size to use.
      branch_factor: The integer branch factor to use.
      steps_per_iteration: The integer number of steps to take per iteration.

    Returns:
      The highest-likelihood drum track as computed by the beam search.
    """
    drum_tracks = [copy.deepcopy(drums) for _ in range(beam_size)]
    graph_initial_state = self._session.graph.get_collection('initial_state')[0]
    loglik = np.zeros(beam_size)

    # Choose the number of steps for the first iteration such that subsequent
    # iterations can all take the same number of steps.
    first_iteration_num_steps = (num_steps - 1) % steps_per_iteration + 1

    inputs = self._config.encoder_decoder.get_inputs_batch(
        drum_tracks, full_length=True)
    initial_state = np.tile(
        self._session.run(graph_initial_state), (beam_size, 1))
    drum_tracks, final_state, loglik = self._generate_branches(
        drum_tracks, loglik, branch_factor, first_iteration_num_steps, inputs,
        initial_state, temperature)

    num_iterations = (num_steps -
                      first_iteration_num_steps) / steps_per_iteration

    for _ in range(num_iterations):
      drum_tracks, final_state, loglik = self._prune_branches(
          drum_tracks, final_state, loglik, k=beam_size)
      inputs = self._config.encoder_decoder.get_inputs_batch(drum_tracks)
      drum_tracks, final_state, loglik = self._generate_branches(
          drum_tracks, loglik, branch_factor, steps_per_iteration, inputs,
          final_state, temperature)

    # Prune to a single drum track.
    drum_tracks, final_state, loglik = self._prune_branches(
        drum_tracks, final_state, loglik, k=1)

    tf.logging.info('Beam search yields melody with log-likelihood: %f ',
                    loglik[0])

    return drum_tracks[0]

  def generate_drum_track(self, num_steps, primer_drums, temperature=1.0,
                          beam_size=1, branch_factor=1, steps_per_iteration=1):
    """Generate a drum track from a primer drum track.

    Args:
      num_steps: The integer length in steps of the final drum track, after
          generation. Includes the primer.
      primer_drums: The primer drum track, a DrumTrack object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes drum tracks more
         random, less than 1.0 makes drum tracks less random.
      beam_size: An integer, beam size to use when generating drum tracks via
          beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of steps to take per beam search
          iteration.

    Returns:
      The generated DrumTrack object (which begins with the provided primer drum
          track).

    Raises:
      DrumsRnnModelException: If the primer drum track has zero length or is not
          shorter than num_steps.
    """
    if not primer_drums:
      raise DrumsRnnModelException(
          'primer drum track must have non-zero length')
    if len(primer_drums) >= num_steps:
      raise DrumsRnnModelException(
          'primer drum track must be shorter than `num_steps`')

    drums = copy.deepcopy(primer_drums)

    if num_steps > len(drums):
      drums = self._beam_search(drums, num_steps - len(drums), temperature,
                                beam_size, branch_factor, steps_per_iteration)

    return drums


class DrumsRnnConfig(object):
  """Stores a configuration for a DrumsRnn.

  Attributes:
    details: The GeneratorDetails message describing the config.
    encoder_decoder: The EventSequenceEncoderDecoder object to use.
    hparams: The HParams containing hyperparameters to use.
  """

  def __init__(self, details, encoder_decoder, hparams):
    self.details = details
    self.encoder_decoder = encoder_decoder
    self.hparams = hparams


# Default configurations.
default_configs = {
    'simple': DrumsRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='simple',
            description='Drums RNN with 2-state encoding.'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.DrumNoDrumOneHotEncoding()),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            clip_norm=5,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.95)),
    'beats': DrumsRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='beats',
            description='Drums RNN with binary counters.'),
        magenta.music.LookbackEventSequenceEncoderDecoder(
            magenta.music.MultiDrumOneHotEncoding(),
            lookback_distances=[],
            binary_counter_bits=6),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[256, 256, 256],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            attn_length=32,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.95)),
    'beats_with_lookback': DrumsRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='beats_and_lookback',
            description='Drums RNN with binary counters and lookback.'),
        magenta.music.LookbackEventSequenceEncoderDecoder(
            magenta.music.MultiDrumOneHotEncoding(),
            binary_counter_bits=6),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[256, 256, 256],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            attn_length=32,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.95))
}
