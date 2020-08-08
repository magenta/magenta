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

"""Event sequence RNN model."""

import collections
import copy
import functools

from magenta.common import beam_search
from magenta.common import state_util
from magenta.contrib import training as contrib_training
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import model
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf

# Model state when generating event sequences, consisting of the next inputs to
# feed the model, the current RNN state, the current control sequence (if
# applicable), and state for the current control sequence (if applicable).
ModelState = collections.namedtuple(
    'ModelState', ['inputs', 'rnn_state', 'control_events', 'control_state'])


class EventSequenceRnnModelError(Exception):
  pass


def _extend_control_events_default(control_events, events, state):
  """Default function for extending control event sequence.

  This function extends a control event sequence by duplicating the final event
  in the sequence.  The control event sequence will be extended to have length
  one longer than the generated event sequence.

  Args:
    control_events: The control event sequence to extend.
    events: The list of generated events.
    state: State maintained while generating, unused.

  Returns:
    The resulting state after extending the control sequence (in this case the
    state will be returned unmodified).
  """
  while len(control_events) <= len(events):
    control_events.append(control_events[-1])
  return state


class EventSequenceRnnModel(model.BaseModel):
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
    events_rnn_graph.get_build_graph_fn('generate', self._config)()

  def _batch_size(self):
    """Extracts the batch size from the graph."""
    return int(self._session.graph.get_collection('inputs')[0].shape[0])

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

    if isinstance(softmax, list):
      if softmax[0].shape[1] > 1:
        softmaxes = []
        for beam in range(softmax[0].shape[0]):
          beam_softmaxes = []
          for event in range(softmax[0].shape[1] - 1):
            beam_softmaxes.append(
                [softmax[s][beam, event] for s in range(len(softmax))])
          softmaxes.append(beam_softmaxes)
        loglik = self._config.encoder_decoder.evaluate_log_likelihood(
            event_sequences, softmaxes)
      else:
        loglik = np.zeros(len(event_sequences))
    else:
      if softmax.shape[1] > 1:
        # The inputs batch is longer than a single step, so we also want to
        # compute the log-likelihood of the event sequences up until the step
        # we're generating.
        loglik = self._config.encoder_decoder.evaluate_log_likelihood(
            event_sequences, softmax[:, :-1, :])
      else:
        loglik = np.zeros(len(event_sequences))

    indices = np.array(self._config.encoder_decoder.extend_event_sequences(
        event_sequences, softmax))
    if isinstance(softmax, list):
      p = 1.0
      for i in range(len(softmax)):
        p *= softmax[i][range(len(event_sequences)), -1, indices[:, i]]
    else:
      p = softmax[range(len(event_sequences)), -1, indices]

    return final_state, loglik + np.log(p)

  def _generate_step(self, event_sequences, model_states, logliks, temperature,
                     extend_control_events_callback=None,
                     modify_events_callback=None):
    """Extends a list of event sequences by a single step each.

    This method modifies the event sequences in place. It also returns the
    modified event sequences and updated model states and log-likelihoods.

    Args:
      event_sequences: A list of event sequence objects, which are extended by
          this method.
      model_states: A list of model states, each of which contains model inputs
          and initial RNN states.
      logliks: A list containing the current log-likelihood for each event
          sequence.
      temperature: The softmax temperature.
      extend_control_events_callback: A function that takes three arguments: a
          current control event sequence, a current generated event sequence,
          and the control state. The function should a) extend the control event
          sequence to be one longer than the generated event sequence (or do
          nothing if it is already at least this long), and b) return the
          resulting control state.
      modify_events_callback: An optional callback for modifying the event list.
          Can be used to inject events rather than having them generated. If not
          None, will be called with 3 arguments after every event: the current
          EventSequenceEncoderDecoder, a list of current EventSequences, and a
          list of current encoded event inputs.

    Returns:
      event_sequences: A list of extended event sequences. These are modified in
          place but also returned.
      final_states: A list of resulting model states, containing model inputs
          for the next step along with RNN states for each event sequence.
      logliks: A list containing the updated log-likelihood for each event
          sequence.
    """
    # Split the sequences to extend into batches matching the model batch size.
    batch_size = self._batch_size()
    num_seqs = len(event_sequences)
    num_batches = int(np.ceil(num_seqs / float(batch_size)))

    # Extract inputs and RNN states from the model states.
    inputs = [model_state.inputs for model_state in model_states]
    initial_states = [model_state.rnn_state for model_state in model_states]

    # Also extract control sequences and states.
    control_sequences = [
        model_state.control_events for model_state in model_states]
    control_states = [
        model_state.control_state for model_state in model_states]

    final_states = []
    logliks = np.array(logliks, dtype=np.float32)

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
      logliks[i:j - pad_amt] += batch_loglik[:j - i - pad_amt]

    # Construct inputs for next step.
    if extend_control_events_callback is not None:
      # We are conditioning on control sequences.
      for idx in range(len(control_sequences)):
        # Extend each control sequence to ensure that it is longer than the
        # corresponding event sequence.
        control_states[idx] = extend_control_events_callback(
            control_sequences[idx], event_sequences[idx], control_states[idx])
      next_inputs = self._config.encoder_decoder.get_inputs_batch(
          control_sequences, event_sequences)
    else:
      next_inputs = self._config.encoder_decoder.get_inputs_batch(
          event_sequences)

    if modify_events_callback:
      # Modify event sequences and inputs for next step.
      modify_events_callback(
          self._config.encoder_decoder, event_sequences, next_inputs)

    model_states = [ModelState(inputs=inputs, rnn_state=final_state,
                               control_events=control_events,
                               control_state=control_state)
                    for inputs, final_state, control_events, control_state
                    in zip(next_inputs, final_states,
                           control_sequences, control_states)]

    return event_sequences, model_states, logliks

  def _generate_events(self, num_steps, primer_events, temperature=1.0,
                       beam_size=1, branch_factor=1, steps_per_iteration=1,
                       control_events=None, control_state=None,
                       extend_control_events_callback=(
                           _extend_control_events_default),
                       modify_events_callback=None):
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
          used along with the target sequence to generate model inputs. In some
          cases, the control event sequence cannot be fully-determined as later
          control events depend on earlier generated events; use the
          `extend_control_events_callback` argument to provide a function that
          extends the control event sequence.
      control_state: Initial state used by `extend_control_events_callback`.
      extend_control_events_callback: A function that takes three arguments: a
          current control event sequence, a current generated event sequence,
          and the control state. The function should a) extend the control event
          sequence to be one longer than the generated event sequence (or do
          nothing if it is already at least this long), and b) return the
          resulting control state.
      modify_events_callback: An optional callback for modifying the event list.
          Can be used to inject events rather than having them generated. If not
          None, will be called with 3 arguments after every event: the current
          EventSequenceEncoderDecoder, a list of current EventSequences, and a
          list of current encoded event inputs.

    Returns:
      The generated event sequence (which begins with the provided primer).

    Raises:
      EventSequenceRnnModelError: If the primer sequence has zero length or
          is not shorter than num_steps.
    """
    if (control_events is not None and
        not isinstance(self._config.encoder_decoder,
                       note_seq.ConditionalEventSequenceEncoderDecoder)):
      raise EventSequenceRnnModelError(
          'control sequence provided but encoder/decoder is not a '
          'ConditionalEventSequenceEncoderDecoder')
    if control_events is not None and extend_control_events_callback is None:
      raise EventSequenceRnnModelError(
          'must provide callback for extending control sequence (or use'
          'default)')

    if not primer_events:
      raise EventSequenceRnnModelError(
          'primer sequence must have non-zero length')
    if len(primer_events) >= num_steps:
      raise EventSequenceRnnModelError(
          'primer sequence must be shorter than `num_steps`')

    if len(primer_events) >= num_steps:
      # Sequence is already long enough, no need to generate.
      return primer_events

    event_sequences = [copy.deepcopy(primer_events)]

    # Construct inputs for first step after primer.
    if control_events is not None:
      # We are conditioning on a control sequence. Make sure it is longer than
      # the primer sequence.
      control_state = extend_control_events_callback(
          control_events, primer_events, control_state)
      inputs = self._config.encoder_decoder.get_inputs_batch(
          [control_events], event_sequences, full_length=True)
    else:
      inputs = self._config.encoder_decoder.get_inputs_batch(
          event_sequences, full_length=True)

    if modify_events_callback:
      # Modify event sequences and inputs for first step after primer.
      modify_events_callback(
          self._config.encoder_decoder, event_sequences, inputs)

    graph_initial_state = self._session.graph.get_collection('initial_state')
    initial_states = state_util.unbatch(self._session.run(graph_initial_state))

    # Beam search will maintain a state for each sequence consisting of the next
    # inputs to feed the model, and the current RNN state. We start out with the
    # initial full inputs batch and the zero state.
    initial_state = ModelState(
        inputs=inputs[0], rnn_state=initial_states[0],
        control_events=control_events, control_state=control_state)

    generate_step_fn = functools.partial(
        self._generate_step,
        temperature=temperature,
        extend_control_events_callback=
        extend_control_events_callback if control_events is not None else None,
        modify_events_callback=modify_events_callback)

    events, _, loglik = beam_search(
        initial_sequence=event_sequences[0],
        initial_state=initial_state,
        generate_step_fn=generate_step_fn,
        num_steps=num_steps - len(primer_events),
        beam_size=beam_size,
        branch_factor=branch_factor,
        steps_per_iteration=steps_per_iteration)

    tf.logging.info('Beam search yields sequence with log-likelihood: %f ',
                    loglik)

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
      EventSequenceRnnModelError: If the event sequences are not all the
          same length, or if the control sequence is shorter than the event
          sequences.
    """
    num_steps = len(event_sequences[0])
    for events in event_sequences[1:]:
      if len(events) != num_steps:
        raise EventSequenceRnnModelError(
            'log likelihood evaluation requires all event sequences to have '
            'the same length')
    if control_events is not None and len(control_events) < num_steps:
      raise EventSequenceRnnModelError(
          'control sequence must be at least as long as the event sequences')

    batch_size = self._batch_size()
    num_full_batches = len(event_sequences) // batch_size

    loglik = np.empty(len(event_sequences))

    # Since we're computing log-likelihood and not generating, the inputs batch
    # doesn't need to include the final event in each sequence.
    if control_events is not None:
      # We are conditioning on a control sequence.
      inputs = self._config.encoder_decoder.get_inputs_batch(
          [control_events] * len(event_sequences),
          [events[:-1] for events in event_sequences],
          full_length=True)
    else:
      inputs = self._config.encoder_decoder.get_inputs_batch(
          [events[:-1] for events in event_sequences], full_length=True)

    graph_initial_state = self._session.graph.get_collection('initial_state')
    initial_state = self._session.run(graph_initial_state)
    offset = 0
    for _ in range(num_full_batches):
      # Evaluate a single step for one batch of event sequences.
      batch_indices = range(offset, offset + batch_size)
      batch_loglik = self._evaluate_batch_log_likelihood(
          [event_sequences[i] for i in batch_indices],
          [inputs[i] for i in batch_indices],
          [initial_state] * len(batch_indices))
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
          np.append([initial_state] * len(batch_indices),
                    np.tile(inputs[-1, :], (pad_size, 1)),
                    axis=0))
      loglik[batch_indices] = batch_loglik[0:num_extra]

    return loglik


class EventSequenceRnnConfig(object):
  """Stores a configuration for an event sequence RNN.

  Only one of `steps_per_quarter` or `steps_per_second` will be applicable for
  any particular model.

  Attributes:
    details: The GeneratorDetails message describing the config.
    encoder_decoder: The EventSequenceEncoderDecoder or
        ConditionalEventSequenceEncoderDecoder object to use.
    hparams: The HParams containing hyperparameters to use. Will be merged with
        default hyperparameter values.
    steps_per_quarter: The integer number of quantized time steps per quarter
        note to use.
    steps_per_second: The integer number of quantized time steps per second to
        use.
  """

  def __init__(self, details, encoder_decoder, hparams,
               steps_per_quarter=4, steps_per_second=100):
    hparams_dict = {
        'batch_size': 64,
        'rnn_layer_sizes': [128, 128],
        'dropout_keep_prob': 1.0,
        'attn_length': 0,
        'clip_norm': 3,
        'learning_rate': 0.001,
        'residual_connections': False,
        'use_cudnn': False
    }
    hparams_dict.update(hparams.values())

    self.details = details
    self.encoder_decoder = encoder_decoder
    self.hparams = contrib_training.HParams(**hparams_dict)
    self.steps_per_quarter = steps_per_quarter
    self.steps_per_second = steps_per_second
