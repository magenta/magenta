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

# internal imports
import tensorflow as tf

import magenta
from magenta.models.shared import events_rnn_model
import magenta.music as mm


class DrumsRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN drum track generation models."""

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
    """
    return self._generate_events(num_steps, primer_drums, temperature,
                                 beam_size, branch_factor, steps_per_iteration)

  def drum_track_log_likelihood(self, drums):
    """Evaluate the log likelihood of a drum track under the model.

    Args:
      drums: The DrumTrack object for which to evaluate the log likelihood.

    Returns:
      The log likelihood of `drums` under this model.
    """
    return self._evaluate_log_likelihood([drums])[0]


# Default configurations.
default_configs = {
    'one_drum': events_rnn_model.EventSequenceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='one_drum',
            description='Drums RNN with 2-state encoding.'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.MultiDrumOneHotEncoding([
                [39] +  # use hand clap as default when decoding
                list(range(mm.MIN_MIDI_PITCH, 39)) +
                list(range(39, mm.MAX_MIDI_PITCH + 1))])),
        tf.contrib.training.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            clip_norm=5,
            learning_rate=0.001),
        steps_per_quarter=2),

    'drum_kit': events_rnn_model.EventSequenceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='drum_kit',
            description='Drums RNN with multiple drums and binary counters.'),
        magenta.music.LookbackEventSequenceEncoderDecoder(
            magenta.music.MultiDrumOneHotEncoding(),
            lookback_distances=[],
            binary_counter_bits=6),
        tf.contrib.training.HParams(
            batch_size=128,
            rnn_layer_sizes=[256, 256, 256],
            dropout_keep_prob=0.5,
            attn_length=32,
            clip_norm=3,
            learning_rate=0.001))
}
