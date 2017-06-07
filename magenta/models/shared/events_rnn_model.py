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
"""Event sequence RNN model."""

import copy
import heapq

# internal imports

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

from magenta.common import state_util
from magenta.models.shared import events_rnn_graph
import magenta.music as mm


class EventSequenceRnnModelException(Exception):
  pass


class EventSequenceRnnModel(mm.BaseModel):
  """Class for RNN event sequence generation models.

  Currently this class only supports generation, of both event sequences and
  note sequences (via event sequences). Support for model training will be added
  at a later time.
  """

  def __init__(self, config):
    """Initialize the EventSequenceRnnModel.

    Args:
      config: An EventSequenceRnnConfig containing the encoder/decoder and
        HParams to use.
    """
    super(EventSequenceRnnModel, self).__init__()
    self._config = config

  def _build_graph_for_generation(self):
    return events_rnn_graph.build_graph('generate', self._config)

  def _batch_size(self):
    """Extracts the batch size from the graph."""
    return self._session.graph.get_collection('inputs')[0].shape[0].value

  def _generate_step_for_batch(self, event_sequences, inputs, initial_state,
                               temperature):
    """Extends a batch of event sequences by a single step each.

    This method modifies the event sequences in place.

    Args:
      event_sequences: A list of event sequences, each of which is a Python
          list-like object. The list of event sequences should have length equal
          to `self._batch_size()`. These are extended by this method.
      inputs: A Python list of model inputs, with length equal to
          `self._batch_size()`.
      initial_state: A numpy array containing the initial RNN state, where
          `initial_state.shape[0]` is equal to `self._batch_size()`.
      temperature: The softmax temperature.

    Returns:
      final_state: The final RNN state, a numpy array the same size as
          `initial_state`.
      loglik: The log-likelihood of the chosen softmax value for each event
          sequence, a 1-D numpy array of length
          `self._batch_size()`. If `inputs` is a full-length inputs batch, the
          log-likelihood of each entire sequence up to and including the
          generated step will be computed and returned.
    """
    assert len(event_sequences) == self._batch_size()

    graph_inputs = self._session.graph.get_collection('inputs')[0]
    graph_initial_state = self._session.graph.get_collection('initial_state')
    graph_final_state = self._session.graph.get_collection('final_state')
    graph_softmax = self._session.graph.get_collection('softmax')[0]
    graph_temperature = self._session.graph.get_collection('temperature')

    feed_dict = {graph_inputs: inputs,
                 tuple(graph_initial_state): initial_state}
    # For backwards compatibility, we only try to pass temperature if the
    # placeholder exists in the graph.
    if graph_temperature:
      feed_dict[graph_temperature[0]] = temperature
    final_state, softmax = self._session.run(
        [graph_final_state, graph_softmax], feed_dict)

    if softmax.shape[1] > 1:
      # The inputs batch is longer than a single step, so we also want to
      # compute the log-likelihood of the event sequences up until the step
      # we're generating.
      loglik = self._config.encoder_decoder.evaluate_log_likelihood(
          event_sequences, softmax[:, :-1, :])
    else:
      loglik = np.zeros(len(event_sequences))

    indices = self._config.encoder_decoder.extend_event_sequences(
        event_sequences, softmax)
    p = softmax[range(len(event_sequences)), -1, indices]

    return final_state, loglik + np.log(p)

  def _generate_step(self, event_sequences, inputs, initial_states,
                     temperature):
    """Extends a list of event sequences by a single step each.

    This method modifies the event sequences in place.

    Args:
      event_sequences: A list of event sequence objects, which are extended by
          this method.
      inputs: A Python list of model inputs, with length equal to the number of
          event sequences.
      initial_states: A collection of structures for the initial RNN states,
          one for each event sequence.
      temperature: The softmax temperature.

    Returns:
      final_states: The final RNN states, a list the same size as
          `initial_states`.
      loglik: The log-likelihood of the chosen softmax value for each event
          sequence, a 1-D numpy array of length
          `self._batch_size()`. If `inputs` is a full-length inputs batch, the
          log-likelihood of each entire sequence up to and including the
          generated step will be computed and returned.
    """
    # Split the sequences to extend into batches matching the model batch size.
    batch_size = self._batch_size()
    num_seqs = len(event_sequences)
    num_batches = int(np.ceil(num_seqs / float(batch_size)))

    final_states = []
    loglik = np.empty(num_seqs)

    # Add padding to fill the final batch.
    pad_amt = -len(event_sequences) % batch_size
    padded_event_sequences = event_sequences + [
        copy.deepcopy(event_sequences[-1]) for _ in range(pad_amt)]
    padded_inputs = inputs + [inputs[-1]] * pad_amt
    padded_initial_states = initial_states + [initial_states[-1]] * pad_amt

    for b in range(num_batches):
      i, j = b * batch_size, (b + 1) * batch_size
      pad_amt = max(0, j - num_seqs)
      # Generate a single step for one batch of event sequences.
      batch_final_state, batch_loglik = self._generate_step_for_batch(
          padded_event_sequences[i:j],
          padded_inputs[i:j],
          state_util.batch(padded_initial_states[i:j], batch_size),
          temperature)
      final_states += state_util.unbatch(
          batch_final_state, batch_size)[:j - i - pad_amt]
      loglik[i:j - pad_amt] = batch_loglik[:j - i - pad_amt]

    return final_states, loglik

  def _generate_branches(self, event_sequences, loglik, branch_factor,
                         num_steps, inputs, initial_states, temperature):
    """Performs a single iteration of branch generation for beam search.

    This method generates `branch_factor` branches for each event sequence in
    `event_sequences`, where each branch extends the event sequence by
    `num_steps` steps.

    Args:
      event_sequences: A list of event sequence objects.
      loglik: A 1-D numpy array of event sequence log-likelihoods, the same size
          as `event_sequences`.
      branch_factor: The integer branch factor to use.
      num_steps: The integer number of steps to take per branch.
      inputs: A Python list of model inputs, with length equal to the number of
          event sequences.
      initial_states: A collection of structures for the initial RNN states,
          one for each event sequence.
      temperature: The softmax temperature.

    Returns:
      all_event_sequences: A list of event sequences, with `branch_factor` times
          as many event sequences as the initial list.
      all_final_state: A list of structures for the initial RNN states, with a
          length equal to the length of `all_event_sequences`.
      all_loglik: A 1-D numpy array of event sequence log-likelihoods, with
          length equal to the length of `all_event_sequences`.
    """
    all_event_sequences = [copy.deepcopy(events)
                           for events in event_sequences * branch_factor]
    all_inputs = inputs * branch_factor
    all_final_state = initial_states * branch_factor
    all_loglik = np.tile(loglik, (branch_factor,))

    for _ in range(num_steps):
      all_final_state, all_step_loglik = self._generate_step(
          all_event_sequences, all_inputs, all_final_state, temperature)
      all_loglik += all_step_loglik

    return all_event_sequences, all_final_state, all_loglik

  def _prune_branches(self, event_sequences, final_states, loglik, k):
    """Prune all but `k` event sequences.

    This method prunes all but the `k` event sequences with highest log-
    likelihood.

    Args:
      event_sequences: A list of event sequence objects.
      final_states: A collection of structures for the final RNN states,
          one for each event sequence.
      loglik: A 1-D numpy array of log-likelihoods, the same size as
          `event_sequences`.
      k: The number of event sequences to keep after pruning.

    Returns:
      event_sequences: The pruned list of event sequences, of length `k`.
      final_states: The pruned list of structures for the final RNN states, of
          length `k`.
      loglik: The pruned event sequence log-likelihoods, a 1-D numpy array of
          length `k`.
    """
    indices = heapq.nlargest(k, range(len(event_sequences)),
                             key=lambda i: loglik[i])

    event_sequences = [event_sequences[i] for i in indices]
    final_states = [final_states[i] for i in indices]
    loglik = loglik[indices]

    return event_sequences, final_states, loglik

  def _beam_search(self, events, num_steps, temperature, beam_size,
                   branch_factor, steps_per_iteration, control_events=None,
                   modify_events_callback=None):
    """Generates an event sequence using beam search.

    Initially, the beam is filled with `beam_size` copies of the initial event
    sequence.

    Each iteration, the beam is pruned to contain only the `beam_size` event
    sequences with highest likelihood. Then `branch_factor` new event sequences
    are generated for each sequence in the beam. These new sequences are formed
    by extending each sequence in the beam by `steps_per_iteration` steps. So
    between a branching and a pruning phase, there will be `beam_size` *
    `branch_factor` active event sequences.

    Prior to the first "real" iteration, an initial branch generation will take
    place. This is for two reasons:

    1) The RNN model needs to be "primed" with the initial event sequence.
    2) The desired total number of steps `num_steps` might not be a multiple of
       `steps_per_iteration`, so the initial branching generates steps such that
       all subsequent iterations can generate `steps_per_iteration` steps.

    After the final iteration, the single event sequence in the beam with
    highest likelihood will be returned.

    Args:
      events: The initial event sequence, a Python list-like object.
      num_steps: The integer length in steps of the final event sequence, after
          generation.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes events more
         random, less than 1.0 makes events less random.
      beam_size: The integer beam size to use.
      branch_factor: The integer branch factor to use.
      steps_per_iteration: The integer number of steps to take per iteration.
      control_events: A sequence of control events upon which to condition the
          generation. If not None, the encoder/decoder should be a
          ConditionalEventSequenceEncoderDecoder, and the control events will be
          used along with the target sequence to generate model inputs.
      modify_events_callback: An optional callback for modifying the event list.
          Can be used to inject events rather than having them generated. If not
          None, will be called with 3 arguments after every event: the current
          EventSequenceEncoderDecoder, a list of current EventSequences, and a
          list of current encoded event inputs.

    Returns:
      The highest-likelihood event sequence as computed by the beam search.
    """
    event_sequences = [copy.deepcopy(events) for _ in range(beam_size)]
    graph_initial_state = self._session.graph.get_collection('initial_state')
    loglik = np.zeros(beam_size)

    # Choose the number of steps for the first iteration such that subsequent
    # iterations can all take the same number of steps.
    first_iteration_num_steps = (num_steps - 1) % steps_per_iteration + 1

    if control_events is not None:
      # We are conditioning on a control sequence.
      inputs = self._config.encoder_decoder.get_inputs_batch(
          control_events, event_sequences, full_length=True)
    else:
      inputs = self._config.encoder_decoder.get_inputs_batch(
          event_sequences, full_length=True)

    if modify_events_callback:
      modify_events_callback(
          self._config.encoder_decoder, event_sequences, inputs)

    zero_state = state_util.unbatch(self._session.run(graph_initial_state))[0]
    initial_states = [zero_state] * beam_size
    event_sequences, final_state, loglik = self._generate_branches(
        event_sequences, loglik, branch_factor, first_iteration_num_steps,
        inputs, initial_states, temperature)

    num_iterations = (num_steps -
                      first_iteration_num_steps) / steps_per_iteration

    for _ in range(num_iterations):
      event_sequences, final_state, loglik = self._prune_branches(
          event_sequences, final_state, loglik, k=beam_size)
      if control_events is not None:
        # We are conditioning on a control sequence.
        inputs = self._config.encoder_decoder.get_inputs_batch(
            control_events, event_sequences)
      else:
        inputs = self._config.encoder_decoder.get_inputs_batch(event_sequences)

      if modify_events_callback:
        modify_events_callback(
            self._config.encoder_decoder, event_sequences, inputs)

      event_sequences, final_state, loglik = self._generate_branches(
          event_sequences, loglik, branch_factor, steps_per_iteration, inputs,
          final_state, temperature)

    # Prune to a single sequence.
    event_sequences, final_state, loglik = self._prune_branches(
        event_sequences, final_state, loglik, k=1)

    tf.logging.info('Beam search yields sequence with log-likelihood: %f ',
                    loglik[0])

    return event_sequences[0]

  def _generate_events(self, num_steps, primer_events, temperature=1.0,
                       beam_size=1, branch_factor=1, steps_per_iteration=1,
                       control_events=None, modify_events_callback=None):
    """Generate an event sequence from a primer sequence.

    Args:
      num_steps: The integer length in steps of the final event sequence, after
          generation. Includes the primer.
      primer_events: The primer event sequence, a Python list-like object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes events more
         random, less than 1.0 makes events less random.
      beam_size: An integer, beam size to use when generating event sequences
          via beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of steps to take per beam search
          iteration.
      control_events: A sequence of control events upon which to condition the
          generation. If not None, the encoder/decoder should be a
          ConditionalEventSequenceEncoderDecoder, and the control events will be
          used along with the target sequence to generate model inputs.
      modify_events_callback: An optional callback for modifying the event list.
          Can be used to inject events rather than having them generated. If not
          None, will be called with 3 arguments after every event: the current
          EventSequenceEncoderDecoder, a list of current EventSequences, and a
          list of current encoded event inputs.

    Returns:
      The generated event sequence (which begins with the provided primer).

    Raises:
      EventSequenceRnnModelException: If the primer sequence has zero length or
          is not shorter than num_steps.
    """
    if (control_events is not None and
        not isinstance(self._config.encoder_decoder,
                       mm.ConditionalEventSequenceEncoderDecoder)):
      raise EventSequenceRnnModelException(
          'control sequence provided but encoder/decoder is not a '
          'ConditionalEventSequenceEncoderDecoder')

    if not primer_events:
      raise EventSequenceRnnModelException(
          'primer sequence must have non-zero length')
    if len(primer_events) >= num_steps:
      raise EventSequenceRnnModelException(
          'primer sequence must be shorter than `num_steps`')
    if control_events is not None and len(control_events) < num_steps:
      raise EventSequenceRnnModelException(
          'control sequence must be at least `num_steps`')

    events = primer_events
    if num_steps > len(primer_events):
      events = self._beam_search(events, num_steps - len(events), temperature,
                                 beam_size, branch_factor, steps_per_iteration,
                                 control_events, modify_events_callback)
    return events

  def _evaluate_batch_log_likelihood(self, event_sequences, inputs,
                                     initial_state):
    """Evaluates the log likelihood of a batch of event sequences.

    Args:
      event_sequences: A list of event sequences, each of which is a Python
          list-like object. The list of event sequences should have length equal
          to `self._batch_size()`.
      inputs: A Python list of model inputs, with length equal to
          `self._batch_size()`.
      initial_state: A numpy array containing the initial RNN state, where
          `initial_state.shape[0]` is equal to `self._batch_size()`.

    Returns:
      A Python list containing the log likelihood of each sequence in
      `event_sequences`.
    """
    graph_inputs = self._session.graph.get_collection('inputs')[0]
    graph_initial_state = self._session.graph.get_collection('initial_state')
    graph_softmax = self._session.graph.get_collection('softmax')[0]
    graph_temperature = self._session.graph.get_collection('temperature')

    feed_dict = {graph_inputs: inputs,
                 tuple(graph_initial_state): initial_state}
    # For backwards compatibility, we only try to pass temperature if the
    # placeholder exists in the graph.
    if graph_temperature:
      feed_dict[graph_temperature[0]] = 1.0
    softmax = self._session.run(graph_softmax, feed_dict)

    return self._config.encoder_decoder.evaluate_log_likelihood(
        event_sequences, softmax)

  def _evaluate_log_likelihood(self, event_sequences, control_events=None):
    """Evaluate log likelihood for a list of event sequences of the same length.

    Args:
      event_sequences: A list of event sequences for which to evaluate the log
          likelihood.
      control_events: A sequence of control events upon which to condition the
          event sequences. If not None, the encoder/decoder should be a
          ConditionalEventSequenceEncoderDecoder, and the log likelihood of each
          event sequence will be computed conditional on the control sequence.

    Returns:
      The log likelihood of each sequence in `event_sequences`.

    Raises:
      EventSequenceRnnModelException: If the event sequences are not all the
          same length, or if the control sequence is shorter than the event
          sequences.
    """
    num_steps = len(event_sequences[0])
    for events in event_sequences[1:]:
      if len(events) != num_steps:
        raise EventSequenceRnnModelException(
            'log likelihood evaluation requires all event sequences to have '
            'the same length')
    if control_events is not None and len(control_events) < num_steps:
      raise EventSequenceRnnModelException(
          'control sequence must be at least as long as the event sequences')

    batch_size = self._batch_size()
    num_full_batches = len(event_sequences) / batch_size

    loglik = np.empty(len(event_sequences))

    # Since we're computing log-likelihood and not generating, the inputs batch
    # doesn't need to include the final event in each sequence.
    if control_events is not None:
      # We are conditioning on a control sequence.
      inputs = self._config.encoder_decoder.get_inputs_batch(
          control_events, [events[:-1] for events in event_sequences],
          full_length=True)
    else:
      inputs = self._config.encoder_decoder.get_inputs_batch(
          [events[:-1] for events in event_sequences], full_length=True)

    graph_initial_state = self._session.graph.get_collection('initial_state')
    initial_state = [
        self._session.run(graph_initial_state)] * len(event_sequences)
    offset = 0
    for _ in range(num_full_batches):
      # Evaluate a single step for one batch of event sequences.
      batch_indices = range(offset, offset + batch_size)
      batch_loglik = self._evaluate_batch_log_likelihood(
          [event_sequences[i] for i in batch_indices],
          [inputs[i] for i in batch_indices],
          initial_state[batch_indices])
      loglik[batch_indices] = batch_loglik
      offset += batch_size

    if offset < len(event_sequences):
      # There's an extra non-full batch. Pad it with a bunch of copies of the
      # final sequence.
      num_extra = len(event_sequences) - offset
      pad_size = batch_size - num_extra
      batch_indices = range(offset, len(event_sequences))
      batch_loglik = self._evaluate_batch_log_likelihood(
          [event_sequences[i] for i in batch_indices] + [
              copy.deepcopy(event_sequences[-1]) for _ in range(pad_size)],
          [inputs[i] for i in batch_indices] + inputs[-1] * pad_size,
          np.append(initial_state[batch_indices],
                    np.tile(inputs[-1, :], (pad_size, 1)),
                    axis=0))
      loglik[batch_indices] = batch_loglik[0:num_extra]

    return loglik


class EventSequenceRnnConfig(object):
  """Stores a configuration for an event sequence RNN.

  Attributes:
    details: The GeneratorDetails message describing the config.
    encoder_decoder: The EventSequenceEncoderDecoder or
        ConditionalEventSequenceEncoderDecoder object to use.
    hparams: The HParams containing hyperparameters to use.
    steps_per_quarter: The integer number of quantized time steps per quarter
        note to use.
  """

  def __init__(self, details, encoder_decoder, hparams, steps_per_quarter=4):
    self.details = details
    self.encoder_decoder = encoder_decoder
    self.hparams = hparams
    self.steps_per_quarter = steps_per_quarter
