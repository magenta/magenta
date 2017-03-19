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
"""Python functions which run only within a Jupyter notebook."""

# internal imports
import IPython

from magenta.music import midi_synth

_DEFAULT_SAMPLE_RATE = 44100


def play_sequence(sequence,
                  synth=midi_synth.synthesize,
                  sample_rate=_DEFAULT_SAMPLE_RATE,
                  **synth_args):
  """Creates an interactive player for a synthesized note sequence.

  This function should only be called from a Jupyter notebook.

  Args:
    sequence: A music_pb2.NoteSequence to synthesize and play.
    synth: A synthesis function that takes a sequence and sample rate as input.
    sample_rate: The sample rate at which to synthesize.
    **synth_args: Additional keyword arguments to pass to the synth function.
  """
  array_of_floats = synth(sequence, sample_rate=sample_rate, **synth_args)
  IPython.display.display(IPython.display.Audio(array_of_floats,
                                                rate=sample_rate))
