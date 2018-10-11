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
"""MIDI audio synthesis."""

# internal imports
import numpy as np

from magenta.music import midi_io


def synthesize(sequence, sample_rate, wave=np.sin):
  """Synthesizes audio from a music_pb2.NoteSequence using a waveform.

  This uses the pretty_midi `synthesize` method. Sound quality will be lower
  than using `fluidsynth` with a good SoundFont.

  Args:
    sequence: A music_pb2.NoteSequence to synthesize.
    sample_rate: An integer audio sampling rate in Hz.
    wave: Function that returns a periodic waveform.

  Returns:
    A 1-D numpy float array containing the synthesized waveform.
  """
  midi = midi_io.sequence_proto_to_pretty_midi(sequence)
  return midi.synthesize(fs=sample_rate, wave=wave)


def fluidsynth(sequence, sample_rate, sf2_path=None):
  """Synthesizes audio from a music_pb2.NoteSequence using FluidSynth.

  This uses the pretty_midi `fluidsynth` method. In order to use this synth,
  you must have FluidSynth and pyFluidSynth installed.

  Args:
    sequence: A music_pb2.NoteSequence to synthesize.
    sample_rate: An integer audio sampling rate in Hz.
    sf2_path: A string path to a SoundFont. If None, uses the TimGM6mb.sf2 file
        included with pretty_midi.

  Returns:
    A 1-D numpy float array containing the synthesized waveform.
  """
  midi = midi_io.sequence_proto_to_pretty_midi(sequence)
  return midi.fluidsynth(fs=sample_rate, sf2_path=sf2_path)
