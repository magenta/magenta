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
"""Polyphonic RNN generation code as a SequenceGenerator interface."""

from functools import partial

# internal imports

import tensorflow as tf

from magenta.models.polyphonic_rnn import polyphony_lib
from magenta.models.polyphonic_rnn import polyphony_model

import magenta.music as mm


class PolyphonicRnnSequenceGenerator(mm.BaseSequenceGenerator):
  """Polyphonic RNN generation code as a SequenceGenerator interface."""

  def __init__(self, model, details, steps_per_quarter=4, checkpoint=None,
               bundle=None):
    """Creates a PolyphonicRnnSequenceGenerator.

    Args:
      model: Instance of PolyphonicRnnModel.
      details: A generator_pb2.GeneratorDetails for this generator.
      steps_per_quarter: What precision to use when quantizing the sequence. How
          many steps per quarter note.
      checkpoint: Where to search for the most recent model checkpoint. Mutually
          exclusive with `bundle`.
      bundle: A GeneratorBundle object that includes both the model checkpoint
          and metagraph. Mutually exclusive with `checkpoint`.
    """
    super(PolyphonicRnnSequenceGenerator, self).__init__(
        model, details, checkpoint, bundle)
    self._steps_per_quarter = steps_per_quarter

  def _seconds_to_steps(self, seconds, qpm):
    """Converts seconds to steps.

    Uses the generator's steps_per_quarter setting and the specified qpm.

    Args:
      seconds: number of seconds.
      qpm: current qpm.

    Returns:
      Number of steps the seconds represent.
    """
    return int(seconds * (qpm / 60.0) * self._steps_per_quarter)

  def _generate(self, input_sequence, generator_options):
    if len(generator_options.input_sections) > 1:
      raise mm.SequenceGeneratorException(
          'This model supports at most one input_sections message, but got %s' %
          len(generator_options.input_sections))
    if len(generator_options.generate_sections) != 1:
      raise mm.SequenceGeneratorException(
          'This model supports only 1 generate_sections message, but got %s' %
          len(generator_options.generate_sections))

    # This sequence will be quantized later, so it is guaranteed to have only 1
    # tempo.
    qpm = mm.DEFAULT_QUARTERS_PER_MINUTE
    if input_sequence.tempos:
      qpm = input_sequence.tempos[0].qpm

    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      primer_sequence = mm.extract_subsequence(
          input_sequence, input_section.start_time, input_section.end_time)
      input_start_step = self._seconds_to_steps(
          input_section.start_time, qpm)
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
    quantized_primer_sequence = mm.quantize_note_sequence(
        primer_sequence, self._steps_per_quarter)

    extracted_seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_primer_sequence, search_start_step=input_start_step)
    assert len(extracted_seqs) <= 1

    start_step = self._seconds_to_steps(
        generate_section.start_time, qpm)
    end_step = self._seconds_to_steps(generate_section.end_time, qpm)

    if extracted_seqs and extracted_seqs[0]:
      poly_seq = extracted_seqs[0]
    else:
      # If no track could be extracted, create an empty track that starts 1 step
      # before the request start_step. This will result in 1 step of silence
      # when the track is extended below.
      poly_seq = polyphony_lib.PolyphonicSequence(
          steps_per_quarter=(
              quantized_primer_sequence.quantization_info.steps_per_quarter),
          start_step=start_step)

    # Ensure that the track extends up to the step we want to start generating.
    poly_seq.set_length(start_step - poly_seq.start_step)
    poly_seq.trim_trailing_end_and_step_end_events()

    # Extract generation arguments from generator options.
    arg_types = {
        'temperature': lambda arg: arg.float_value,
        'beam_size': lambda arg: arg.int_value,
        'branch_factor': lambda arg: arg.int_value,
        'steps_per_iteration': lambda arg: arg.int_value
    }
    args = dict((name, value_fn(generator_options.args[name]))
                for name, value_fn in arg_types.items()
                if name in generator_options.args)

    total_steps = end_step - start_step
    while poly_seq.num_steps < total_steps:
      # Assume it takes ~5 rnn steps to generate one quantized step.
      # Can't know for sure until generation is finished because the number of
      # notes per quantized step is variable.
      steps_to_gen = total_steps - poly_seq.num_steps
      rnn_steps_to_gen = 5 * steps_to_gen
      tf.logging.info(
          'Need to generate %d more steps for this sequence, will try asking '
          'for %d RNN steps' % (steps_to_gen, rnn_steps_to_gen))
      poly_seq = self._model.generate_polyphonic_sequence(
          len(poly_seq) + rnn_steps_to_gen, poly_seq, **args)
    poly_seq.set_length(total_steps)

    generated_sequence = poly_seq.to_sequence(qpm=qpm)
    assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return generated_sequence


def get_generator_map():
  """Returns a map from the generator ID to its SequenceGenerator class.

  Binds the `config` argument so that the constructor matches the
  BaseSequenceGenerator class.
  Returns:
    Map from the generator ID to its SequenceGenerator class with a bound
    `config` argument.
  """
  return {key: partial(PolyphonicRnnSequenceGenerator,
                       polyphony_model.PolyphonicRnnModel(config),
                       config.details)
          for (key, config) in polyphony_model.default_configs.items()}
