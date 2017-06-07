# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Classes for converting between pianoroll input and model input/output."""

from __future__ import division

# internal imports

import numpy as np

from magenta.music import encoder_decoder


class PianorollEncoderDecoder(encoder_decoder.EventSequenceEncoderDecoder):
  """An EventSequenceEncoderDecoder that produces a pianoroll encoding.

  Inputs are binary arrays with active pitches (with some offset) at each step
  set to 1 and inactive pitches set to 0.

  Events are PianorollSequence events, which are tuples of active pitches
  (with some offset) at each step.
  """

  def __init__(self, input_size=88):
    """Initialize a PianorollEncoderDecoder object.

    Args:
      input_size: The size of the input vector.
    """
    self._input_size = input_size

  @property
  def input_size(self):
    return self._input_size

  @property
  def num_classes(self):
    return 2 ** self.input_size

  @property
  def default_event_label(self):
    return 0

  def _event_to_label(self, event):
    label = 0
    for pitch in event:
      label += 2**pitch
    return label

  def _event_to_input(self, event):
    input_ = np.zeros(self.input_size, np.float32)
    input_[list(event)] = 1
    return input_

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the event sequence.

    Args:
      events: A list-like sequence of PianorollSequence events.
      position: An integer event position in the event sequence.

    Returns:
      An input vector, a list of floats.
    """
    return self._event_to_input(events[position])

  def events_to_label(self, events, position):
    """Returns the label for the given position in the event sequence.

    Args:
      events: A list-like sequence of PianorollSequence events.
      position: An integer event position in the event sequence.

    Returns:
      A label, an integer.
    """
    return self._event_to_label(events[position])

  def class_index_to_event(self, class_index, events):
    """Returns the event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A list-like sequence of events. This object is not used in this
          implementation.

    Returns:
      An PianorollSequence event value.
    """
    assert class_index < self.num_classes
    event = []
    for i in range(self.input_size):
      if class_index % 2:
        event.append(i)
      class_index >>= 1
    assert class_index == 0
    return tuple(event)

  def extend_event_sequences(self, pianoroll_seqs, samples):
    """Extends the event sequences by adding the new samples.

    Args:
      pianoroll_seqs: A collection of PianorollSequences to append `samples` to.
      samples: A collection of binary arrays with active pitches set to 1 and
         inactive pitches set to 0, which will be added to the corresponding
         `pianoroll_seqs`.
    Raises:
      ValueError: if inputs are not of equal length.
    """
    if len(pianoroll_seqs) != len(samples):
      raise ValueError(
          '`pianoroll_seqs` and `samples` must have equal lengths.')
    for pianoroll_seq, sample in zip(pianoroll_seqs, samples):
      event = tuple(np.where(sample)[0])
      pianoroll_seq.append(event)
