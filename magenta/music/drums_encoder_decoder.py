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
"""Classes for converting between drum tracks and models inputs/outputs."""

# internal imports
from magenta.music import encoder_decoder


# Default list of 9 drum types, where each type is represented by a list of
# MIDI pitches for drum sounds belonging to that type. This default list
# attempts to map all GM1 and GM2 drums onto a much smaller standard drum kit
# based on drum sound and function.
DEFAULT_DRUM_TYPE_PITCHES = [
    # bass drum
    [36, 35],

    # snare drum
    [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],

    # closed hi-hat
    [42, 44, 54, 68, 69, 70, 71, 73, 78, 80],

    # open hi-hat
    [46, 67, 72, 74, 79, 81],

    # low tom
    [45, 29, 41, 61, 64, 84],

    # mid tom
    [48, 47, 60, 63, 77, 86, 87],

    # high tom
    [50, 30, 43, 62, 76, 83],

    # crash cymbal
    [49, 55, 57, 58],

    # ride cymbal
    [51, 52, 53, 59, 82]
]


class DrumsEncodingException(Exception):
  pass


class MultiDrumOneHotEncoding(encoder_decoder.OneHotEncoding):
  """Encodes drum events as binary where each bit is a different drum type.

  Each event consists of multiple simultaneous drum "pitches". This encoding
  converts each pitch to a drum type, e.g. bass drum, hi-hat, etc. Each drum
  type is mapped to a single bit of a binary integer representation, where the
  bit has value 0 if the drum type is not present, and 1 if it is present.

  If multiple "pitches" corresponding to the same drum type (e.g. two different
  ride cymbals) are present, the encoding is the same as if only of of them were
  present.
  """

  def __init__(self, drum_type_pitches=None, ignore_unknown_drums=True):
    """Initializes the MultiDrumOneHotEncoding.

    Args:
      drum_type_pitches: A Python list of the MIDI pitch values for each drum
          type. If None, `DEFAULT_DRUM_TYPE_PITCHES` will be used.
      ignore_unknown_drums: If True, unknown drum pitches will not be encoded.
          If False, a DrumsEncodingException will be raised when unknown drum
          pitches are encountered.
    """
    if drum_type_pitches is None:
      drum_type_pitches = DEFAULT_DRUM_TYPE_PITCHES
    self._drum_map = dict(enumerate(drum_type_pitches))
    self._inverse_drum_map = dict((pitch, index)
                                  for index, pitches in self._drum_map.items()
                                  for pitch in pitches)
    self._ignore_unknown_drums = ignore_unknown_drums

  @property
  def num_classes(self):
    return 2 ** len(self._drum_map)

  @property
  def default_event(self):
    return frozenset()

  def encode_event(self, event):
    drum_type_indices = set()
    for pitch in event:
      if pitch in self._inverse_drum_map:
        drum_type_indices.add(self._inverse_drum_map[pitch])
      elif not self._ignore_unknown_drums:
        raise DrumsEncodingException('unknown drum pitch: %d' % pitch)
    return sum(2 ** i for i in drum_type_indices)

  def decode_event(self, index):
    bits = reversed(str(bin(index)))
    # Use the first "pitch" for each drum type.
    return frozenset(self._drum_map[i][0]
                     for i, b in enumerate(bits)
                     if b == '1')
