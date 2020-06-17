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

"""Melody RNN model."""

import copy

from magenta.contrib import training as contrib_training
from magenta.models.shared import events_rnn_model
import note_seq
from note_seq.protobuf import generator_pb2

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = 0


class MelodyRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN melody generation models."""

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
    """
    melody = copy.deepcopy(primer_melody)

    transpose_amount = melody.squash(
        self._config.min_note,
        self._config.max_note,
        self._config.transpose_to_key)

    melody = self._generate_events(num_steps, melody, temperature, beam_size,
                                   branch_factor, steps_per_iteration)

    melody.transpose(-transpose_amount)

    return melody

  def melody_log_likelihood(self, melody):
    """Evaluate the log likelihood of a melody under the model.

    Args:
      melody: The Melody object for which to evaluate the log likelihood.

    Returns:
      The log likelihood of `melody` under this model.
    """
    melody_copy = copy.deepcopy(melody)

    melody_copy.squash(
        self._config.min_note,
        self._config.max_note,
        self._config.transpose_to_key)

    return self._evaluate_log_likelihood([melody_copy])[0]


class MelodyRnnConfig(events_rnn_model.EventSequenceRnnConfig):
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
    super(MelodyRnnConfig, self).__init__(details, encoder_decoder, hparams)

    if min_note < note_seq.MIN_MIDI_PITCH:
      raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > note_seq.MAX_MIDI_PITCH + 1:
      raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note - min_note < note_seq.NOTES_PER_OCTAVE:
      raise ValueError('max_note - min_note must be >= 12. min_note is %d. '
                       'max_note is %d. max_note - min_note is %d.' %
                       (min_note, max_note, max_note - min_note))
    if (transpose_to_key is not None and
        (transpose_to_key < 0 or
         transpose_to_key > note_seq.NOTES_PER_OCTAVE - 1)):
      raise ValueError('transpose_to_key must be >= 0 and <= 11. '
                       'transpose_to_key is %d.' % transpose_to_key)

    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key


# Default configurations.
default_configs = {
    'basic_rnn':
        MelodyRnnConfig(
            generator_pb2.GeneratorDetails(
                id='basic_rnn',
                description='Melody RNN with one-hot encoding.'),
            note_seq.OneHotEventSequenceEncoderDecoder(
                note_seq.MelodyOneHotEncoding(
                    min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE)),
            contrib_training.HParams(
                batch_size=128,
                rnn_layer_sizes=[128, 128],
                dropout_keep_prob=0.5,
                clip_norm=5,
                learning_rate=0.001)),
    'mono_rnn':
        MelodyRnnConfig(
            generator_pb2.GeneratorDetails(
                id='mono_rnn',
                description='Monophonic RNN with one-hot encoding.'),
            note_seq.OneHotEventSequenceEncoderDecoder(
                note_seq.MelodyOneHotEncoding(min_note=0, max_note=128)),
            contrib_training.HParams(
                batch_size=128,
                rnn_layer_sizes=[128, 128],
                dropout_keep_prob=0.5,
                clip_norm=5,
                learning_rate=0.001),
            min_note=0,
            max_note=128,
            transpose_to_key=None),
    'lookback_rnn':
        MelodyRnnConfig(
            generator_pb2.GeneratorDetails(
                id='lookback_rnn',
                description='Melody RNN with lookback encoding.'),
            note_seq.LookbackEventSequenceEncoderDecoder(
                note_seq.MelodyOneHotEncoding(
                    min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE)),
            contrib_training.HParams(
                batch_size=128,
                rnn_layer_sizes=[128, 128],
                dropout_keep_prob=0.5,
                clip_norm=5,
                learning_rate=0.001)),
    'attention_rnn':
        MelodyRnnConfig(
            generator_pb2.GeneratorDetails(
                id='attention_rnn',
                description='Melody RNN with lookback encoding and attention.'),
            note_seq.KeyMelodyEncoderDecoder(
                min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE),
            contrib_training.HParams(
                batch_size=128,
                rnn_layer_sizes=[128, 128],
                dropout_keep_prob=0.5,
                attn_length=40,
                clip_norm=3,
                learning_rate=0.001))
}
