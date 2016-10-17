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
import random

# internal imports
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
    if len(generator_options.generate_sections) != 1:
      raise magenta.music.SequenceGeneratorException(
          'This model supports only 1 generate_sections message, but got %s' %
          len(generator_options.generate_sections))

    generate_section = generator_options.generate_sections[0]
    primer_sequence = input_sequence

    notes_by_end_time = sorted(primer_sequence.notes, key=lambda n: n.end_time)
    last_end_time = notes_by_end_time[-1].end_time if notes_by_end_time else 0
    if last_end_time > generate_section.start_time_seconds:
      raise magenta.music.SequenceGeneratorException(
          'Got GenerateSection request for section that is before the end of '
          'the NoteSequence. This model can only extend sequences. '
          'Requested start time: %s, Final note end time: %s' %
          (generate_section.start_time_seconds, notes_by_end_time[-1].end_time))

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
        generate_section.start_time_seconds, qpm)
    end_step = self._seconds_to_steps(generate_section.end_time_seconds, qpm)

    if extracted_melodies and extracted_melodies[0]:
      melody = extracted_melodies[0]
    else:
      tf.logging.warn('No melodies were extracted from the priming sequence. '
                      'Melodies will be generated from scratch.')
      melody = magenta.music.Melody([
          random.randint(self._config.encoder_decoder.min_note,
                         self._config.encoder_decoder.max_note)])
      start_step += 1

    # Ensure that the melody extends up to the step we want to start generating.
    melody.set_length(start_step)

    generated_melody = self.generate_melody(end_step, melody)
    generated_sequence = generated_melody.to_sequence(qpm=qpm)
    assert generated_sequence.total_time <= generate_section.end_time_seconds
    return generated_sequence

  def generate_melody(self, num_steps, primer_melody):
    """Generate a melody from a primer melody.

    Args:
      num_steps: An integer number of steps to generate. This is the total
          number of steps to generate, including the primer melody.
      primer_melody: The primer melody, a Melody object.

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
          'primer melody must be shorter than num_steps')

    melody = copy.deepcopy(primer_melody)

    encoder_decoder = self._config.encoder_decoder
    transpose_amount = melody.squash(
        encoder_decoder.min_note,
        encoder_decoder.max_note,
        encoder_decoder.transpose_to_key)

    inputs = self._session.graph.get_collection('inputs')[0]
    initial_state = self._session.graph.get_collection('initial_state')[0]
    final_state = self._session.graph.get_collection('final_state')[0]
    softmax = self._session.graph.get_collection('softmax')[0]

    final_state_ = None
    for i in range(num_steps - (len(melody) + melody.start_step)):
      if i == 0:
        inputs_ = encoder_decoder.get_inputs_batch([melody], full_length=True)
        initial_state_ = self._session.run(initial_state)
      else:
        inputs_ = encoder_decoder.get_inputs_batch([melody])
        initial_state_ = final_state_

      feed_dict = {inputs: inputs_, initial_state: initial_state_}
      final_state_, softmax_ = self._session.run(
          [final_state, softmax], feed_dict)
      encoder_decoder.extend_event_sequences([melody], softmax_)

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
