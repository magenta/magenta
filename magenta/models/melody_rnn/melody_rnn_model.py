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
"""Melody RNN model."""

# internal imports
import magenta

from magenta.models.melody_rnn import melody_rnn_sequence_generator


class MelodyRnnModel(object):
  """Class for RNN melody generation models.

  Currently this class only supports generation, of both melodies and note
  sequences (containing melodies). Support for model training will be added
  at a later time.
  """

  def __init__(self, config, checkpoint=None, bundle_filename=None,
               steps_per_quarter=4):
    """Initialize the MelodyRnnModel.

    Args:
      config: A MelodyRnnConfig containing the MelodyEncoderDecoder and HParams
        use.
      checkpoint: Where to search for the most recent model checkpoint.
      bundle_filename: The filename of a generator_pb2.GeneratorBundle object
          that includes both the model checkpoint and metagraph.
      steps_per_quarter: What precision to use when quantizing the melody. How
          many steps per quarter note.
    """
    if bundle_filename is not None:
      bundle = magenta.music.read_bundle_file(bundle_filename)
    else:
      bundle = None
    self._generator = melody_rnn_sequence_generator(
        config, checkpoint, bundle, steps_per_quarter)

  def generate_melody(self, num_steps, primer_melody):
    """Uses the model to generate a melody from a primer melody.

    Args:
      num_steps: An integer number of steps to generate. This is the total
          number of steps to generate, including the primer melody.
      primer_melody: The primer melody, a Melody object.

    Returns:
      The generated Melody object (which begins with the provided primer
          melody).
    """
    with self._generator:
      return self._generator.generate_melody(num_steps, primer_melody)

  def generate_sequence(self, input_sequence, generator_options):
    """Generates a sequence from the model based on sequence and options.

    Args:
      input_sequence: An input NoteSequence to base the generation on.
      generator_options: A GeneratorOptions proto with options to use for
          generation.

    Returns:
      The generated NoteSequence proto.
    """
    with self._generator:
      return self._generator.generate(input_sequence, generator_options)
