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

# Lint as: python3
"""Drum hit mappings."""

from magenta.models.onsets_frames_transcription import constants
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf

# Names for pitches in the Groove MIDI Dataset and Expanded Groove MIDI Dataset.
GROOVE_PITCH_NAMES = {
    36: 'Kick',
    38: 'Snare_Head',
    40: 'Snare_Rim',
    37: 'Snare_X-Stick',
    48: 'Tom1',
    50: 'Tom1_Rim',
    45: 'Tom2',
    47: 'Tom2_Rim',
    43: 'Tom3_Head',
    58: 'Tom3_Rim',
    46: 'HHOpen_Bow',
    26: 'HHOpen_Edge',
    42: 'HHClosed_Bow',
    22: 'HHClosed_Edge',
    44: 'HHPedal',
    49: 'Crash1_Bow',
    55: 'Crash1_Edge',
    57: 'Crash2_Bow',
    52: 'Crash2_Edge',
    51: 'Ride_Bow',
    59: 'Ride_Edge',
    53: 'Ride_Bell',
    # New
    39: 'Clap',
    54: 'Tambourine',
    56: 'Cowbell',
    70: 'Maracas',
    64: 'Low_Conga',
    75: 'Claves',
}

# Drum hit mappings.
# Each mapping is an array of arrays containing pitches.  Each internal array
# represents a pitch class. The first element of each pitch class is the base
# pitch to which all other pitches in the class will be mapped.
HIT_MAPS = {
    '3-hit': [
        # kick drum
        [36],

        # hi-hats (including cymbals, maracas, cowbell)
        [42, 22, 44, 46, 26, 49, 52, 55, 56, 57, 51, 53, 59, 54, 70],

        # snare drum (including toms)
        [38, 37, 40, 48, 50, 45, 47, 43, 58, 39, 64, 75],
    ],
    '8-hit': [
        # Kick
        [36],

        # Snare,X-stick, handclap
        [38, 40, 37, 39],

        # Toms + (Low_Conga extra)
        [48, 50, 45, 47, 43, 58, 64],

        # HH + Tambourine + (Maracas extra)
        [46, 26, 42, 22, 44, 54, 70],

        # Ride
        [51, 59],

        # Ride bell+ cow bell
        [53, 56],

        # Crashes
        [49, 55, 57, 52],

        # Clave / Sticks
        [75],
    ],
}


def map_pianoroll(pianoroll,
                  mapping_name,
                  reduce_mode,
                  min_pitch=constants.MIN_MIDI_PITCH):
  """Return a mapped pianoroll.

  The given mapping is a list of pitch classes, each with a base pitch. The
  pianoroll is a tensor of prediction of the form frame X pitch. All pitches are
  mapped to the base pitches in the provided mapping, and all other pitches are
  zeroed out.

  Args:
    pianoroll: A tensor of onset predictions of the form frame X pitch.
    mapping_name: Which mapping from HIT_MAPS to use.
    reduce_mode: If 'any', treats values as booleans and uses reduce_any. 'any'
      is appropriate for mapping note pianorolls. If 'max', treats values as
      floats and uses reduce_max. 'max' is appropriate for mapping velocity
      pianorolls.
    min_pitch: Used to offset MIDI pitches for the pianoroll.

  Returns:
    mapped_onset_predictions: The mapped onset_predictions.
  """
  mapping = []
  for m in HIT_MAPS[mapping_name]:
    mapping.append([p - min_pitch for p in m])

  mapped_pitches = {pitches[0]: pitches for pitches in mapping}
  mapped_predictions = []
  for pitch in range(pianoroll.shape[1]):
    if pitch in mapped_pitches:
      if reduce_mode == 'any':
        mapped_predictions.append(
            tf.cast(
                tf.math.reduce_any(
                    tf.cast(
                        tf.gather(pianoroll, mapped_pitches[pitch], axis=1),
                        tf.bool),
                    axis=1), pianoroll.dtype))
      elif reduce_mode == 'max':
        mapped_predictions.append(
            tf.math.reduce_max(
                tf.gather(pianoroll, mapped_pitches[pitch], axis=1), axis=1))
      else:
        raise ValueError('Unknown reduce_mode: {}'.format(reduce_mode))
    else:
      mapped_predictions.append(tf.zeros_like(pianoroll[:, pitch]))
  return tf.stack(mapped_predictions, axis=1)


def map_sequences(sequence_str, mapping_name):
  """Map the NoteSequence for drums."""
  mapping = HIT_MAPS[mapping_name]
  drums_sequence = music_pb2.NoteSequence.FromString(sequence_str)
  simple_mapping = {}
  for pitch_class in mapping:
    for pitch in pitch_class:
      simple_mapping[pitch] = pitch_class[0]

  for note in drums_sequence.notes:
    if note.pitch not in simple_mapping:
      tf.logging.warn('Could not find mapping for pitch %d', note.pitch)
    else:
      note.pitch = simple_mapping[note.pitch]
  return drums_sequence.SerializeToString()
