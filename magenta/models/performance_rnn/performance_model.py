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

import collections
import functools

# internal imports

import tensorflow as tf
import magenta

from magenta.models.shared import events_rnn_model
from magenta.music.performance_lib import PerformanceEvent


# State for constructing a time-varying control sequence. Keeps track of the
# current event position and time step in the generated performance, to allow
# the control sequence to vary with clock time.
PerformanceControlState = collections.namedtuple(
    'PerformanceControlState', ['current_perf_index', 'current_perf_step'])


class PerformanceRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN performance generation models."""

  def generate_performance(
      self, num_steps, primer_sequence, temperature=1.0, beam_size=1,
      branch_factor=1, steps_per_iteration=1, control_signal_fns=None,
      disable_conditioning_fn=None):
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
      control_signal_fns: A list of functions that map time step to desired
          control value, or None if not using control signals.
      disable_conditioning_fn: A function that maps time step to whether or not
          conditioning should be disabled, or None if there is no conditioning
          or conditioning is not optional.

    Returns:
      The generated Performance object (which begins with the provided primer
      track).
    """
    if control_signal_fns:
      control_event = tuple(f(0) for f in control_signal_fns)
      if disable_conditioning_fn is not None:
        control_event = (disable_conditioning_fn(0), control_event)
      control_events = [control_event]
      control_state = PerformanceControlState(
          current_perf_index=0, current_perf_step=0)
      extend_control_events_callback = functools.partial(
          _extend_control_events, control_signal_fns, disable_conditioning_fn)
    else:
      control_events = None
      control_state = None
      extend_control_events_callback = None

    return self._generate_events(
        num_steps, primer_sequence, temperature, beam_size, branch_factor,
        steps_per_iteration, control_events=control_events,
        control_state=control_state,
        extend_control_events_callback=extend_control_events_callback)

  def performance_log_likelihood(self, sequence, control_values,
                                 disable_conditioning):
    """Evaluate the log likelihood of a performance.

    Args:
      sequence: The Performance object for which to evaluate the log likelihood.
      control_values: List of (single) values for all control signal.
      disable_conditioning: Whether or not to disable optional conditioning. If
          True, disable conditioning. If False, do not disable. None when no
          conditioning or it is not optional.


    Returns:
      The log likelihood of `sequence` under this model.
    """
    if control_values:
      control_event = tuple(control_values)
      if disable_conditioning is not None:
        control_event = (disable_conditioning, control_event)
      control_events = [control_event] * len(sequence)
    else:
      control_events = None

    return self._evaluate_log_likelihood(
        [sequence], control_events=control_events)[0]


def _extend_control_events(control_signal_fns, disable_conditioning_fn,
                           control_events, performance, control_state):
  """Extend a performance control sequence.

  Extends `control_events` -- a sequence of control signal value tuples -- to be
  one event longer than `performance`, so the next event of `performance` can be
  conditionally generated.

  This function is meant to be used as the `extend_control_events_callback`
  in the `_generate_events` method of `EventSequenceRnnModel`.

  Args:
    control_signal_fns: A list of functions that map time step to desired
        control value, or None if not using control signals.
    disable_conditioning_fn: A function that maps time step to whether or not
        conditioning should be disabled, or None if there is no conditioning or
        conditioning is not optional.
    control_events: The control sequence to extend.
    performance: The Performance being generated.
    control_state: A PerformanceControlState tuple containing the current
        position in `performance`. We maintain this so as not to have to
        recompute the total performance length (in steps) every time we want to
        extend the control sequence.

  Returns:
    The PerformanceControlState after extending the control sequence one step
    past the end of the generated performance.
  """
  idx = control_state.current_perf_index
  step = control_state.current_perf_step

  while idx < len(performance):
    if performance[idx].event_type == PerformanceEvent.TIME_SHIFT:
      step += performance[idx].event_value
    idx += 1

    control_event = tuple(f(step) for f in control_signal_fns)
    if disable_conditioning_fn is not None:
      control_event = (disable_conditioning_fn(step), control_event)
    control_events.append(control_event)

  return PerformanceControlState(
      current_perf_index=idx, current_perf_step=step)


class PerformanceRnnConfig(events_rnn_model.EventSequenceRnnConfig):
  """Stores a configuration for a Performance RNN.

  Attributes:
    num_velocity_bins: Number of velocity bins to use. If 0, don't use velocity
        at all.
    control_signals: List of PerformanceControlSignal objects to use for
        conditioning, or None if not conditioning on anything.
    optional_conditioning: If True, conditioning can be disabled by setting a
        flag as part of the conditioning input.
  """

  def __init__(self, details, encoder_decoder, hparams, num_velocity_bins=0,
               control_signals=None, optional_conditioning=False):
    if control_signals is not None:
      control_encoder = magenta.music.MultipleEventSequenceEncoder(
          [control.encoder for control in control_signals])
      if optional_conditioning:
        control_encoder = magenta.music.OptionalEventSequenceEncoder(
            control_encoder)
      encoder_decoder = magenta.music.ConditionalEventSequenceEncoderDecoder(
          control_encoder, encoder_decoder)

    super(PerformanceRnnConfig, self).__init__(
        details, encoder_decoder, hparams)
    self.num_velocity_bins = num_velocity_bins
    self.control_signals = control_signals
    self.optional_conditioning = optional_conditioning


default_configs = {
    'performance': PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='performance',
            description='Performance RNN'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.PerformanceOneHotEncoding()),
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
            magenta.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32),

    'performance_with_dynamics_and_modulo_encoding': PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='performance_with_dynamics_and_modulo_encoding',
            description='Performance RNN with dynamics and modulo encoding'),
        magenta.music.ModuloPerformanceEventSequenceEncoderDecoder(
            num_velocity_bins=32),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32),

    'density_conditioned_performance_with_dynamics': PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='density_conditioned_performance_with_dynamics',
            description='Note-density-conditioned Performance RNN + dynamics'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        control_signals=[
            magenta.music.NoteDensityPerformanceControlSignal(
                window_size_seconds=3.0,
                density_bin_ranges=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
        ]),

    'pitch_conditioned_performance_with_dynamics': PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='pitch_conditioned_performance_with_dynamics',
            description='Pitch-histogram-conditioned Performance RNN'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        control_signals=[
            magenta.music.PitchHistogramPerformanceControlSignal(
                window_size_seconds=5.0)
        ]),

    'multiconditioned_performance_with_dynamics': PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='multiconditioned_performance_with_dynamics',
            description='Density- and pitch-conditioned Performance RNN'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        control_signals=[
            magenta.music.NoteDensityPerformanceControlSignal(
                window_size_seconds=3.0,
                density_bin_ranges=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
            magenta.music.PitchHistogramPerformanceControlSignal(
                window_size_seconds=5.0)
        ]),

    'optional_multiconditioned_performance_with_dynamics': PerformanceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='optional_multiconditioned_performance_with_dynamics',
            description='Optionally multiconditioned Performance RNN'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        control_signals=[
            magenta.music.NoteDensityPerformanceControlSignal(
                window_size_seconds=3.0,
                density_bin_ranges=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
            magenta.music.PitchHistogramPerformanceControlSignal(
                window_size_seconds=5.0)
        ],
        optional_conditioning=True)
}
