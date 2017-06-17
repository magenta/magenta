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
"""RNN-NADE generation code as a SequenceGenerator interface."""

from functools import partial

# internal imports

from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_model
import magenta.music as mm


class PianorollRnnNadeSequenceGenerator(mm.BaseSequenceGenerator):
  """RNN-NADE generation code as a SequenceGenerator interface."""

  def __init__(self, model, details, steps_per_quarter=4, checkpoint=None,
               bundle=None):
    """Creates a PianorollRnnNadeSequenceGenerator.

    Args:
      model: Instance of PianorollRnnNadeModel.
      details: A generator_pb2.GeneratorDetails for this generator.
      steps_per_quarter: What precision to use when quantizing the sequence. How
          many steps per quarter note.
      checkpoint: Where to search for the most recent model checkpoint. Mutually
          exclusive with `bundle`.
      bundle: A GeneratorBundle object that includes both the model checkpoint
          and metagraph. Mutually exclusive with `checkpoint`.
    """
    super(PianorollRnnNadeSequenceGenerator, self).__init__(
        model, details, checkpoint, bundle)
    self.steps_per_quarter = steps_per_quarter

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

    steps_per_second = mm.steps_per_quarter_to_steps_per_second(
        self.steps_per_quarter, qpm)

    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      primer_sequence = mm.trim_note_sequence(
          input_sequence, input_section.start_time, input_section.end_time)
      input_start_step = mm.quantize_to_step(
          input_section.start_time, steps_per_second, quantize_cutoff=0)
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
        primer_sequence, self.steps_per_quarter)

    extracted_seqs, _ = mm.extract_pianoroll_sequences(
        quantized_primer_sequence, start_step=input_start_step)
    assert len(extracted_seqs) <= 1

    generate_start_step = mm.quantize_to_step(
        generate_section.start_time, steps_per_second, quantize_cutoff=0)
    # Note that when quantizing end_step, we set quantize_cutoff to 1.0 so it
    # always rounds down. This avoids generating a sequence that ends at 5.0
    # seconds when the requested end time is 4.99.
    generate_end_step = mm.quantize_to_step(
        generate_section.end_time, steps_per_second, quantize_cutoff=1.0)

    if extracted_seqs and extracted_seqs[0]:
      pianoroll_seq = extracted_seqs[0]
    else:
      raise ValueError('No priming pianoroll could be extracted.')

    # Ensure that the track extends up to the step we want to start generating.
    pianoroll_seq.set_length(generate_start_step - pianoroll_seq.start_step)

    # Extract generation arguments from generator options.
    arg_types = {
        'beam_size': lambda arg: arg.int_value,
        'branch_factor': lambda arg: arg.int_value,
    }
    args = dict((name, value_fn(generator_options.args[name]))
                for name, value_fn in arg_types.items()
                if name in generator_options.args)

    total_steps = pianoroll_seq.num_steps + (
        generate_end_step - generate_start_step)

    pianoroll_seq = self._model.generate_pianoroll_sequence(
        total_steps, pianoroll_seq, **args)
    pianoroll_seq.set_length(total_steps)

    generated_sequence = pianoroll_seq.to_sequence(qpm=qpm)
    assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return generated_sequence


def get_generator_map():
  """Returns a map from the generator ID to a SequenceGenerator class creator.

  Binds the `config` argument so that the arguments match the
  BaseSequenceGenerator class constructor.

  Returns:
    Map from the generator ID to its SequenceGenerator class creator with a
    bound `config` argument.
  """
  def create_sequence_generator(config, **kwargs):
    return PianorollRnnNadeSequenceGenerator(
        pianoroll_rnn_nade_model.PianorollRnnNadeModel(config), config.details,
        steps_per_quarter=config.steps_per_quarter, **kwargs)

  return {key: partial(create_sequence_generator, config)
          for (key, config) in pianoroll_rnn_nade_model.default_configs.items()}
