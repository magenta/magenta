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
"""Melody RNN generation code as a SequenceGenerator interface."""

import copy
from functools import partial
import heapq
import random


# internal imports
import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin

import tensorflow as tf
import magenta

from magenta.models.melody_rnn import melody_rnn_config
from magenta.models.melody_rnn import melody_rnn_graph


class MelodyRnnSequenceGeneratorException(Exception):
  pass


class MelodyRnnSequenceGenerator(magenta.music.BaseSequenceGenerator):
  """Shared Melody RNN generation code as a SequenceGenerator interface."""

  def __init__(self, config, steps_per_quarter=4, checkpoint=None, bundle=None):
    """Creates a MelodyRnnSequenceGenerator.

    Args:
      config: A MelodyRnnConfig containing the GeneratorDetails,
          MelodyEncoderDecoder, and HParams to use.
      steps_per_quarter: What precision to use when quantizing the melody. How
          many steps per quarter note.
      checkpoint: Where to search for the most recent model checkpoint. Mutually
          exclusive with `bundle`.
      bundle: A GeneratorBundle object that includes both the model checkpoint
          and metagraph. Mutually exclusive with `checkpoint`.
    """
    super(MelodyRnnSequenceGenerator, self).__init__(
        config.details, checkpoint, bundle)
    self._session = None
    self._steps_per_quarter = steps_per_quarter

    self._config = config
    # Override hparams for generation.
    self._config.hparams.dropout_keep_prob = 1.0
    self._config.hparams.batch_size = 1

  def _initialize_with_checkpoint(self, checkpoint_file):
    graph = melody_rnn_graph.build_graph('generate', self._config)
    with graph.as_default():
      saver = tf.train.Saver()
      self._session = tf.Session()
      tf.logging.info('Checkpoint used: %s', checkpoint_file)
      saver.restore(self._session, checkpoint_file)

  def _initialize_with_checkpoint_and_metagraph(self, checkpoint_filename,
                                                metagraph_filename):
    with tf.Graph().as_default():
      self._session = tf.Session()
      new_saver = tf.train.import_meta_graph(metagraph_filename)
      new_saver.restore(self._session, checkpoint_filename)

  def _write_checkpoint_with_metagraph(self, checkpoint_filename):
    with self._session.graph.as_default():
      saver = tf.train.Saver(sharded=False)
      saver.save(self._session, checkpoint_filename, meta_graph_suffix='meta',
                 write_meta_graph=True)

  def _close(self):
    self._session.close()
    self._session = None

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
      raise magenta.music.SequenceGeneratorException(
          'This model supports at most one input_sections message, but got %s' %
          len(generator_options.input_sections))
    if len(generator_options.generate_sections) != 1:
      raise magenta.music.SequenceGeneratorException(
          'This model supports only 1 generate_sections message, but got %s' %
          len(generator_options.generate_sections))

    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      primer_sequence = magenta.music.extract_subsequence(
          input_sequence, input_section.start_time, input_section.end_time)
    else:
      primer_sequence = input_sequence

    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    if last_end_time > generate_section.start_time:
      raise magenta.music.SequenceGeneratorException(
          'Got GenerateSection request for section that is before the end of '
          'the NoteSequence. This model can only extend sequences. '
          'Requested start time: %s, Final note end time: %s' %
          (generate_section.start_time, last_end_time))

    # Quantize the priming sequence.
    quantized_sequence = magenta.music.QuantizedSequence()
    quantized_sequence.from_note_sequence(
        primer_sequence, self._steps_per_quarter)
    # Setting gap_bars to infinite ensures that the entire input will be used.
    extracted_melodies, _ = magenta.music.extract_melodies(
        quantized_sequence, min_bars=0, min_unique_pitches=1,
        gap_bars=float('inf'), ignore_polyphonic_notes=True)
    assert len(extracted_melodies) <= 1

    qpm = (primer_sequence.tempos[0].qpm
           if primer_sequence and primer_sequence.tempos
           else magenta.music.DEFAULT_QUARTERS_PER_MINUTE)
    start_step = self._seconds_to_steps(
        generate_section.start_time, qpm)
    end_step = self._seconds_to_steps(generate_section.end_time, qpm)

    if extracted_melodies and extracted_melodies[0]:
      melody = extracted_melodies[0]
    else:
      tf.logging.warn('No melodies were extracted from the priming sequence. '
                      'Melodies will be generated from scratch.')
      melody = magenta.music.Melody(
          [random.randint(self._config.encoder_decoder.min_note,
                          self._config.encoder_decoder.max_note)],
          start_step=start_step)
      start_step += 1

    # Ensure that the melody extends up to the step we want to start generating.
    melody.set_length(start_step - melody.start_step)

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

    generated_melody = self.generate_melody(
        end_step - melody.start_step, melody, **args)
    generated_sequence = generated_melody.to_sequence(qpm=qpm)
    assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return generated_sequence

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
      MelodyRnnSequenceGeneratorException: If the primer melody has zero
          length or is not shorter than num_steps.
    """
    if not primer_melody:
      raise MelodyRnnSequenceGeneratorException(
          'primer melody must have non-zero length')
    if len(primer_melody) >= num_steps:
      raise MelodyRnnSequenceGeneratorException(
          'primer melody must be shorter than `num_steps`')

    if beam_size != self._config.hparams.batch_size:
      raise MelodyRnnSequenceGeneratorException(
          'currently beam search only supports using batch size as beam size')

    melody = copy.deepcopy(primer_melody)

    transpose_amount = melody.squash(
        self._config.encoder_decoder.min_note,
        self._config.encoder_decoder.max_note,
        self._config.encoder_decoder.transpose_to_key)

    if num_steps > len(melody):
      melody = self._beam_search(melody, num_steps - len(melody), temperature,
                                 beam_size, branch_factor, steps_per_iteration)

    melody.transpose(-transpose_amount)

    return melody


def get_generator_map():
  """Returns a map from the generator ID to its SequenceGenerator class.

  Binds the `config` argument so that the constructor matches the
  BaseSequenceGenerator class.

  Returns:
    Map from the generator ID to its SequenceGenerator class with a bound
    `config` argument.
  """
  return {key: partial(MelodyRnnSequenceGenerator, config)
          for (key, config) in melody_rnn_config.default_configs.items()}
