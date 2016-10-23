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
"""Classes for converting between chord progressions and models inputs/outputs.

MajorMinorChordOneHotEncoding is an encoding.OneHotEncoding that specifies a
one-hot encoding for ChordProgression events, i.e. chord symbol strings. This
encoding has 25 classes, all 12 major and minor triads plus "no chord".

ChordsEncoderDecoder is an encoding.EventSequenceEncoderDecoder that translates
between chord progressions and model data.

Use ChordsEncoderDecoder.encode or ChordsEncoderDecoder.transpose_and_encode to
convert a ChordProgression object to a tf.train.SequenceExample of inputs and
labels. These SequenceExamples are fed into the model during training and
evaluation.

During generation, use ChordsEncoderDecoder.get_inputs_batch to convert a list
of chord progressions into an inputs batch which can be fed into the model to
predict what the next chord should be for each progression. Then use
ChordsEncoderDecoder.extend_event_sequences to extend each of those chord
progressions with an event sampled from the softmax output by the model.

OneHotChordsEncoderDecoder and LookbackChordsEncoderDecoder are subclasses of
ChordsEncoderDecoder that use the corresponding encodings.
"""

import abc

from magenta.music import chord_symbols_lib
from magenta.music import constants
from magenta.music import encoding

NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
NO_CHORD = constants.NO_CHORD


class ChordEncodingException(Exception):
  pass


class MajorMinorChordOneHotEncoding(encoding.OneHotEncoding):
  """Encodes chords as root + major/minor, with zero index for "no chord".

  Encodes chords as follows:
    0:     "no chord"
    1-12:  chords with a major triad, where 1 is C major, 2 is C# major, etc.
    13-24: chords with a minor triad, where 13 is C minor, 14 is C# minor, etc.
  """

  # Mapping from pitch class index to name.  Eventually this should be defined
  # more globally, but right now only `decode_chord` needs it.
  _PITCH_CLASS_MAPPING = ['C', 'C#', 'D', 'E-', 'E', 'F',
                          'F#', 'G', 'A-', 'A', 'B-', 'B']

  def __init__(self, chord_symbol_functions=
               chord_symbols_lib.ChordSymbolFunctions.get()):
    """Initialize the MajorMinorEncoderDecoder object.

    Args:
      chord_symbol_functions: ChordSymbolFunctions object with which to perform
          the actual transposition of chord symbol strings.
    """
    self._chord_symbol_functions = chord_symbol_functions

  @property
  def num_classes(self):
    return 2 * NOTES_PER_OCTAVE + 1

  @property
  def default_event(self):
    return NO_CHORD

  def encode_event(self, event):
    if event == NO_CHORD:
      return 0

    root = self._chord_symbol_functions.chord_symbol_root(event)
    quality = self._chord_symbol_functions.chord_symbol_quality(event)

    if quality == chord_symbols_lib.CHORD_QUALITY_MAJOR:
      return root + 1
    elif quality == chord_symbols_lib.CHORD_QUALITY_MINOR:
      return root + NOTES_PER_OCTAVE + 1
    else:
      raise ChordEncodingException('chord is neither major nor minor: %s'
                                   % event)

  def decode_event(self, index):
    if index == 0:
      return NO_CHORD
    elif index - 1 < 12:
      # major
      return self._PITCH_CLASS_MAPPING[index - 1]
    else:
      # minor
      return self._PITCH_CLASS_MAPPING[index - NOTES_PER_OCTAVE - 1] + 'm'


class ChordsEncoderDecoder(encoding.EventSequenceEncoderDecoder):
  """An abstract class for translating between chords and model data."""
  __metaclass__ = abc.ABCMeta

  def transpose_and_encode(self, chords, transpose_amount):
    """Returns a SequenceExample for the given chord progression.

    Args:
      chords: A ChordProgression object.
      transpose_amount: The number of half steps to transpose the chords.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    chords.transpose(transpose_amount)
    return self.encode(chords)


class OneHotChordsEncoderDecoder(ChordsEncoderDecoder):
  """A ChordsEncoderDecoder that uses a one-hot chord encoding."""

  def __init__(self, one_hot_encoding=MajorMinorChordOneHotEncoding()):
    """Initialize a OneHotChordsEncoderDecoder object.

    Args:
      one_hot_encoding: An encoding.OneHotEncoding object that specifies a
          one-hot encoding of chord symbol strings.
    """
    super(OneHotChordsEncoderDecoder, self).__init__(
        encoding.OneHotEventSequenceEncoding(one_hot_encoding))


class LookbackChordsEncoderDecoder(ChordsEncoderDecoder):
  """A ChordsEncoderDecoder that uses a one-hot chord encoding with lookback."""

  def __init__(self, one_hot_encoding=MajorMinorChordOneHotEncoding(),
               lookback_distances=None, binary_counter_bits=5):
    """Initialize a LookbackChordsEncoderDecoder object.

    Args:
      one_hot_encoding: An encoding.OneHotEncoding object that specifies a
          one-hot encoding of chord symbol strings.
      lookback_distances: A list of step intervals to look back in history to
          encode both the following event and whether the current step is a
          repeat.
      binary_counter_bits: The number of input bits to use as a counter for the
          metric position of the next note.
    """
    super(LookbackChordsEncoderDecoder, self).__init__(
        encoding.LookbackEventSequenceEncoding(one_hot_encoding,
                                               lookback_distances,
                                               binary_counter_bits))
