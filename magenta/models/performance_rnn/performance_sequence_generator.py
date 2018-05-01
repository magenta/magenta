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
"""Performance RNN generation code as a SequenceGenerator interface."""

from __future__ import division

import ast
from functools import partial
import math

# internal imports
import tensorflow as tf

from magenta.models.performance_rnn import performance_model
import magenta.music as mm
from magenta.music import performance_controls

# This model can leave hanging notes. To avoid cacophony we turn off any note
# after 5 seconds.
MAX_NOTE_DURATION_SECONDS = 5.0

# Default number of notes per second used to determine number of RNN generation
# steps.
DEFAULT_NOTE_DENSITY = performance_controls.DEFAULT_NOTE_DENSITY


class PerformanceRnnSequenceGenerator(mm.BaseSequenceGenerator):
  """Performance RNN generation code as a SequenceGenerator interface."""

  def __init__(self, model, details,
               steps_per_second=mm.DEFAULT_STEPS_PER_SECOND,
               num_velocity_bins=0,
               control_signals=None, optional_conditioning=False,
               max_note_duration=MAX_NOTE_DURATION_SECONDS,
               fill_generate_section=True, checkpoint=None, bundle=None):
    """Creates a PerformanceRnnSequenceGenerator.

    Args:
      model: Instance of PerformanceRnnModel.
      details: A generator_pb2.GeneratorDetails for this generator.
      steps_per_second: Number of quantized steps per second.
      num_velocity_bins: Number of quantized velocity bins. If 0, don't use
          velocity.
      control_signals: A list of PerformanceControlSignal objects.
      optional_conditioning: If True, conditioning can be disabled dynamically.
      max_note_duration: The maximum note duration in seconds to allow during
          generation. This model often forgets to release notes; specifying a
          maximum duration can force it to do so.
      fill_generate_section: If True, the model will generate RNN steps until
          the entire generate section has been filled. If False, the model will
          estimate the number of RNN steps needed and then generate that many
          events, even if the generate section isn't completely filled.
      checkpoint: Where to search for the most recent model checkpoint. Mutually
          exclusive with `bundle`.
      bundle: A GeneratorBundle object that includes both the model checkpoint
          and metagraph. Mutually exclusive with `checkpoint`.
    """
    super(PerformanceRnnSequenceGenerator, self).__init__(
        model, details, checkpoint, bundle)
    self.steps_per_second = steps_per_second
    self.num_velocity_bins = num_velocity_bins
    self.control_signals = control_signals
    self.optional_conditioning = optional_conditioning
    self.max_note_duration = max_note_duration
    self.fill_generate_section = fill_generate_section

  def _generate(self, input_sequence, generator_options):
    if len(generator_options.input_sections) > 1:
      raise mm.SequenceGeneratorException(
          'This model supports at most one input_sections message, but got %s' %
          len(generator_options.input_sections))
    if len(generator_options.generate_sections) != 1:
      raise mm.SequenceGeneratorException(
          'This model supports only 1 generate_sections message, but got %s' %
          len(generator_options.generate_sections))

    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      primer_sequence = mm.trim_note_sequence(
          input_sequence, input_section.start_time, input_section.end_time)
      input_start_step = mm.quantize_to_step(
          input_section.start_time, self.steps_per_second, quantize_cutoff=0.0)
    else:
      primer_sequence = input_sequence
      input_start_step = 0

    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    if last_end_time > generate_section.start_time:
      raise mm.SequenceGeneratorException(
          'Got GenerateSection request for section that is before or equal to '
          'the end of the NoteSequence. This model can only extend sequences. '
          'Requested start time: %s, Final note end time: %s' %
          (generate_section.start_time, last_end_time))

    # Quantize the priming sequence.
    quantized_primer_sequence = mm.quantize_note_sequence_absolute(
        primer_sequence, self.steps_per_second)

    extracted_perfs, _ = mm.extract_performances(
        quantized_primer_sequence, start_step=input_start_step,
        num_velocity_bins=self.num_velocity_bins)
    assert len(extracted_perfs) <= 1

    generate_start_step = mm.quantize_to_step(
        generate_section.start_time, self.steps_per_second, quantize_cutoff=0.0)
    # Note that when quantizing end_step, we set quantize_cutoff to 1.0 so it
    # always rounds down. This avoids generating a sequence that ends at 5.0
    # seconds when the requested end time is 4.99.
    generate_end_step = mm.quantize_to_step(
        generate_section.end_time, self.steps_per_second, quantize_cutoff=1.0)

    if extracted_perfs and extracted_perfs[0]:
      performance = extracted_perfs[0]
    else:
      # If no track could be extracted, create an empty track that starts at the
      # requested generate_start_step.
      performance = mm.Performance(
          steps_per_second=(
              quantized_primer_sequence.quantization_info.steps_per_second),
          start_step=generate_start_step,
          num_velocity_bins=self.num_velocity_bins)

    # Ensure that the track extends up to the step we want to start generating.
    performance.set_length(generate_start_step - performance.start_step)

    # Extract generation arguments from generator options.
    arg_types = {
        'disable_conditioning': lambda arg: ast.literal_eval(arg.string_value),
        'temperature': lambda arg: arg.float_value,
        'beam_size': lambda arg: arg.int_value,
        'branch_factor': lambda arg: arg.int_value,
        'steps_per_iteration': lambda arg: arg.int_value
    }
    if self.control_signals:
      for control in self.control_signals:
        arg_types[control.name] = lambda arg: ast.literal_eval(arg.string_value)

    args = dict((name, value_fn(generator_options.args[name]))
                for name, value_fn in arg_types.items()
                if name in generator_options.args)

    # Make sure control signals are present and convert to lists if necessary.
    if self.control_signals:
      for control in self.control_signals:
        if control.name not in args:
          tf.logging.warning(
              'Control value not specified, using default: %s = %s',
              control.name, control.default_value)
          args[control.name] = [control.default_value]
        elif control.validate(args[control.name]):
          args[control.name] = [args[control.name]]
        else:
          if not isinstance(args[control.name], list) or not all(
              control.validate(value) for value in args[control.name]):
            tf.logging.fatal(
                'Invalid control value: %s = %s',
                control.name, args[control.name])

    # Make sure disable conditioning flag is present when conditioning is
    # optional and convert to list if necessary.
    if self.optional_conditioning:
      if 'disable_conditioning' not in args:
        args['disable_conditioning'] = [False]
      elif isinstance(args['disable_conditioning'], bool):
        args['disable_conditioning'] = [args['disable_conditioning']]
      else:
        if not isinstance(args['disable_conditioning'], list) or not all(
            isinstance(value, bool) for value in args['disable_conditioning']):
          tf.logging.fatal(
              'Invalid disable_conditioning value: %s',
              args['disable_conditioning'])

    total_steps = performance.num_steps + (
        generate_end_step - generate_start_step)

    mean_note_density = (
        sum(args['notes_per_second']) / len(args['notes_per_second'])
        if 'notes_per_second' in args else DEFAULT_NOTE_DENSITY)

    # Set up functions that map generation step to control signal values and
    # disable conditioning flag.
    if self.control_signals:
      control_signal_fns = []
      for control in self.control_signals:
        control_signal_fns.append(partial(
            _step_to_value,
            num_steps=total_steps,
            values=args[control.name]))
        del args[control.name]
      args['control_signal_fns'] = control_signal_fns
    if self.optional_conditioning:
      args['disable_conditioning_fn'] = partial(
          _step_to_value,
          num_steps=total_steps,
          values=args['disable_conditioning'])
      del args['disable_conditioning']

    if not performance:
      # Primer is empty; let's just start with silence.
      performance.set_length(min(performance.max_shift_steps, total_steps))

    while performance.num_steps < total_steps:
      # Assume the average specified (or default) note density and 4 RNN steps
      # per note. Can't know for sure until generation is finished because the
      # number of notes per quantized step is variable.
      note_density = max(1.0, mean_note_density)
      steps_to_gen = total_steps - performance.num_steps
      rnn_steps_to_gen = int(math.ceil(
          4.0 * note_density * steps_to_gen / self.steps_per_second))
      tf.logging.info(
          'Need to generate %d more steps for this sequence, will try asking '
          'for %d RNN steps' % (steps_to_gen, rnn_steps_to_gen))
      performance = self._model.generate_performance(
          len(performance) + rnn_steps_to_gen, performance, **args)

      if not self.fill_generate_section:
        # In the interest of speed just go through this loop once, which may not
        # entirely fill the generate section.
        break

    performance.set_length(total_steps)

    generated_sequence = performance.to_sequence(
        max_note_duration=self.max_note_duration)

    assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return generated_sequence


def _step_to_value(step, num_steps, values):
  """Map step in performance to desired control signal value."""
  num_segments = len(values)
  index = min(step * num_segments // num_steps, num_segments - 1)
  return values[index]


def get_generator_map():
  """Returns a map from the generator ID to a SequenceGenerator class creator.

  Binds the `config` argument so that the arguments match the
  BaseSequenceGenerator class constructor.

  Returns:
    Map from the generator ID to its SequenceGenerator class creator with a
    bound `config` argument.
  """
  def create_sequence_generator(config, **kwargs):
    return PerformanceRnnSequenceGenerator(
        performance_model.PerformanceRnnModel(config), config.details,
        steps_per_second=config.steps_per_second,
        num_velocity_bins=config.num_velocity_bins,
        control_signals=config.control_signals,
        optional_conditioning=config.optional_conditioning,
        fill_generate_section=False,
        **kwargs)

  return {key: partial(create_sequence_generator, config)
          for (key, config) in performance_model.default_configs.items()}
