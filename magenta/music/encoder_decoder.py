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
"""Classes for converting between event sequences and models inputs/outputs.

OneHotEncoding is an abstract class for specifying a one-hot encoding, i.e.
how to convert back and forth between an arbitrary event space and integer
indices between 0 and the number of classes.

EventSequenceEncoderDecoder is an abstract class for translating event
_sequences_, i.e. how to convert event sequences to input vectors and output
labels to be fed into a model, and how to convert from output labels back to
events.

Use EventSequenceEncoderDecoder.encode to convert an event sequence to a
tf.train.SequenceExample of inputs and labels. These SequenceExamples are fed
into the model during training and evaluation.

During generation, use EventSequenceEncoderDecoder.get_inputs_batch to convert a
list of event sequences into an inputs batch which can be fed into the model to
predict what the next event should be for each sequence. Then use
EventSequenceEncoderDecoder.extend_event_sequences to extend each of those event
sequences with an event sampled from the softmax output by the model.

OneHotEventSequenceEncoderDecoder is an EventSequenceEncoderDecoder that uses a
OneHotEncoding of individual events. The input vectors are one-hot encodings of
the most recent event. The output labels are one-hot encodings of the next
event.

LookbackEventSequenceEncoderDecoder is an EventSequenceEncoderDecoder that also
uses a OneHotEncoding of individual events. However, its input and output
encodings also consider whether the event sequence is repeating, and the input
encoding includes binary counters for timekeeping.
"""

import abc

# internal imports
import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin

from magenta.common import sequence_example_lib
from magenta.music import constants


DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_LOOKBACK_DISTANCES = [DEFAULT_STEPS_PER_BAR, DEFAULT_STEPS_PER_BAR * 2]


class OneHotEncoding(object):
  """An interface for specifying a one-hot encoding of individual events."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def num_classes(self):
    """The number of distinct event encodings.

    Returns:
      An int, the range of ints that can be returned by self.encode_event.
    """
    pass

  @abc.abstractproperty
  def default_event(self):
    """An event value to use as a default.

    Returns:
      The default event value.
    """
    pass

  @abc.abstractmethod
  def encode_event(self, event):
    """Convert from an event value to an encoding integer.

    Args:
      event: An event value to encode.

    Returns:
      An integer representing the encoded event, in range [0, self.num_classes).
    """
    pass

  @abc.abstractmethod
  def decode_event(self, index):
    """Convert from an encoding integer to an event value.

    Args:
      index: The encoding, an integer in the range [0, self.num_classes).

    Returns:
      The decoded event value.
    """
    pass


class EventSequenceEncoderDecoder(object):
  """An abstract class for translating between events and model data.

  When building your dataset, the `encode` method takes in an event sequence
  and returns a SequenceExample of inputs and labels. These SequenceExamples
  are fed into the model during training and evaluation.

  During generation, the `get_inputs_batch` method takes in a list of the
  current event sequences and returns an inputs batch which is fed into the
  model to predict what the next event should be for each sequence. The
  `extend_event_sequences` method takes in the list of event sequences and the
  softmax returned by the model and extends each sequence by one step by
  sampling from the softmax probabilities. This loop (`get_inputs_batch` ->
  inputs batch is fed through the model to get a softmax ->
  `extend_event_sequences`) is repeated until the generated event sequences
  have reached the desired length.

  Properties:
    input_size: The length of the list returned by self.events_to_input.
    num_classes: The range of ints that can be returned by
        self.events_to_label.

  The `input_size`, `num_classes`, `events_to_input`, `events_to_label`, and
  `class_index_to_event` method must be overwritten to be specific to your
  model.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def input_size(self):
    """The size of the input vector used by this model.

    Returns:
        An integer, the length of the list returned by self.events_to_input.
    """
    pass

  @abc.abstractproperty
  def num_classes(self):
    """The range of labels used by this model.

    Returns:
        An integer, the range of integers that can be returned by
            self.events_to_label.
    """
    pass

  @abc.abstractproperty
  def default_event_label(self):
    """The class label that represents a default event.

    Returns:
      An int, the class label that represents a default event.
    """
    pass

  @abc.abstractmethod
  def events_to_input(self, events, position):
    """Returns the input vector for the event at the given position.

    Args:
      events: A list-like sequence of events.
      position: An integer event position in the sequence.

    Returns:
      An input vector, a self.input_size length list of floats.
    """
    pass

  @abc.abstractmethod
  def events_to_label(self, events, position):
    """Returns the label for the event at the given position.

    Args:
      events: A list-like sequence of events.
      position: An integer event position in the sequence.

    Returns:
      A label, an integer in the range [0, self.num_classes).
    """
    pass

  @abc.abstractmethod
  def class_index_to_event(self, class_index, events):
    """Returns the event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A list-like sequence of events.

    Returns:
      An event value.
    """
    pass

  def encode(self, events):
    """Returns a SequenceExample for the given event sequence.

    Args:
      events: A list-like sequence of events.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    inputs = []
    labels = []
    for i in range(len(events) - 1):
      inputs.append(self.events_to_input(events, i))
      labels.append(self.events_to_label(events, i + 1))
    return sequence_example_lib.make_sequence_example(inputs, labels)

  def get_inputs_batch(self, event_sequences, full_length=False):
    """Returns an inputs batch for the given event sequences.

    Args:
      event_sequences: A list of list-like event sequences.
      full_length: If True, the inputs batch will be for the full length of
          each event sequence. If False, the inputs batch will only be for the
          last event of each event sequence. A full-length inputs batch is used
          for the first step of extending the event sequences, since the RNN
          cell state needs to be initialized with the priming sequence. For
          subsequent generation steps, only a last-event inputs batch is used.

    Returns:
      An inputs batch. If `full_length` is True, the shape will be
      [len(event_sequences), len(event_sequences[0]), INPUT_SIZE]. If
      `full_length` is False, the shape will be
      [len(event_sequences), 1, INPUT_SIZE].
    """
    inputs_batch = []
    for events in event_sequences:
      inputs = []
      if full_length and len(event_sequences):
        for i in range(len(events)):
          inputs.append(self.events_to_input(events, i))
      else:
        inputs.append(self.events_to_input(events, len(events) - 1))
      inputs_batch.append(inputs)
    return inputs_batch

  def extend_event_sequences(self, event_sequences, softmax):
    """Extends the event_sequences by sampling the softmax probabilities.

    Args:
      event_sequences: A list of EventSequence objects.
      softmax: A list of softmax probability vectors. The list of softmaxes
          should be the same length as the list of event_sequences.

    Returns:
      A python list of chosen class indices, one for each event sequence.
    """
    num_classes = len(softmax[0][0])
    chosen_classes = []
    for i in xrange(len(event_sequences)):
      chosen_class = np.random.choice(num_classes, p=softmax[i][-1])
      event = self.class_index_to_event(chosen_class, event_sequences[i])
      event_sequences[i].append(event)
      chosen_classes.append(chosen_class)
    return chosen_classes


class OneHotEventSequenceEncoderDecoder(EventSequenceEncoderDecoder):
  """An EventSequenceEncoderDecoder that produces a one-hot encoding."""

  def __init__(self, one_hot_encoding):
    """Initialize a OneHotEventSequenceEncoderDecoder object.

    Args:
      one_hot_encoding: A OneHotEncoding object that transforms events to and
          from integer indices.
    """
    self._one_hot_encoding = one_hot_encoding

  @property
  def input_size(self):
    return self._one_hot_encoding.num_classes

  @property
  def num_classes(self):
    return self._one_hot_encoding.num_classes

  @property
  def default_event_label(self):
    return self._one_hot_encoding.encode_event(
        self._one_hot_encoding.default_event)

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the event sequence.

    Returns a one-hot vector for the given position in the event sequence, as
    determined by the one hot encoding.

    Args:
      events: A list-like sequence of events.
      position: An integer event position in the event sequence.

    Returns:
      An input vector, a list of floats.
    """
    input_ = [0.0] * self.input_size
    input_[self._one_hot_encoding.encode_event(events[position])] = 1.0
    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the event sequence.

    Returns the zero-based index value for the given position in the event
    sequence, as determined by the one hot encoding.

    Args:
      events: A list-like sequence of events.
      position: An integer event position in the event sequence.

    Returns:
      A label, an integer.
    """
    return self._one_hot_encoding.encode_event(events[position])

  def class_index_to_event(self, class_index, events):
    """Returns the event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A list-like sequence of events. This object is not used in this
          implementation.

    Returns:
      An event value.
    """
    return self._one_hot_encoding.decode_event(class_index)


class LookbackEventSequenceEncoderDecoder(EventSequenceEncoderDecoder):
  """An EventSequenceEncoderDecoder that encodes repeated events and meter."""

  def __init__(self, one_hot_encoding, lookback_distances=None,
               binary_counter_bits=5):
    """Initializes the LookbackEventSequenceEncoderDecoder.

    Args:
      one_hot_encoding: A OneHotEncoding object that transforms events to and
         from integer indices.
      lookback_distances: A list of step intervals to look back in history to
         encode both the following event and whether the current step is a
         repeat. If None, use default lookback distances.
      binary_counter_bits: The number of input bits to use as a counter for the
         metric position of the next event.
    """
    self._one_hot_encoding = one_hot_encoding
    self._lookback_distances = (lookback_distances
                                if lookback_distances is not None
                                else DEFAULT_LOOKBACK_DISTANCES)
    self._binary_counter_bits = binary_counter_bits

  @property
  def input_size(self):
    one_hot_size = self._one_hot_encoding.num_classes
    num_lookbacks = len(self._lookback_distances)
    return (one_hot_size +                  # current event
            num_lookbacks * one_hot_size +  # next event for each lookback
            self._binary_counter_bits +     # binary counters
            num_lookbacks)                  # whether event matches lookbacks

  @property
  def num_classes(self):
    return self._one_hot_encoding.num_classes + len(self._lookback_distances)

  @property
  def default_event_label(self):
    return self._one_hot_encoding.encode_event(
        self._one_hot_encoding.default_event)

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the event sequence.

    Returns a self.input_size length list of floats. Assuming a one-hot
    encoding with 38 classes, two lookback distances, and five binary counters,
    self.input_size will = 121. Each index represents a different input signal
    to the model.

    Indices [0, 120]:
    [0, 37]: Event of current step.
    [38, 75]: Event of next step for first lookback.
    [76, 113]: Event of next step for second lookback.
    114: 16th note binary counter.
    115: 8th note binary counter.
    116: 4th note binary counter.
    117: Half note binary counter.
    118: Whole note binary counter.
    119: The current step is repeating (first lookback).
    120: The current step is repeating (second lookback).

    Args:
      events: A list-like sequence of events.
      position: An integer position in the event sequence.

    Returns:
      An input vector, an self.input_size length list of floats.
    """
    input_ = [0.0] * self.input_size
    offset = 0

    # Last event.
    index = self._one_hot_encoding.encode_event(events[position])
    input_[index] = 1.0
    offset += self._one_hot_encoding.num_classes

    # Next event if repeating N positions ago.
    for i, lookback_distance in enumerate(self._lookback_distances):
      lookback_position = position - lookback_distance + 1
      if lookback_position < 0:
        event = self._one_hot_encoding.default_event
      else:
        event = events[lookback_position]
      index = self._one_hot_encoding.encode_event(event)
      input_[offset + index] = 1.0
      offset += self._one_hot_encoding.num_classes

    # Binary time counter giving the metric location of the *next* event.
    n = position + 1
    for i in range(self._binary_counter_bits):
      input_[offset] = 1.0 if (n / 2 ** i) % 2 else -1.0
      offset += 1

    # Last event is repeating N bars ago.
    for i, lookback_distance in enumerate(self._lookback_distances):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        input_[offset] = 1.0
      offset += 1

    assert offset == self.input_size

    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the event sequence.

    Returns an integer in the range [0, self.num_classes). Indices in the range
    [0, self._one_hot_encoding.num_classes) map to standard events. Indices
    self._one_hot_encoding.num_classes and self._one_hot_encoding.num_classes +
    1 are signals to repeat events from earlier in the sequence. More distant
    repeats are selected first and standard events are selected last.

    Assuming a one-hot encoding with 38 classes and two lookback distances,
    self.num_classes = 40 and the values will be as follows.

    Values [0, 39]:
      [0, 37]: Event of the last step in the event sequence, if not repeating
               any of the lookbacks.
      38: If the last event is repeating the first lookback, if not also
          repeating the second lookback.
      39: If the last event is repeating the second lookback.

    Args:
      events: A list-like sequence of events.
      position: An integer position in the event sequence.

    Returns:
      A label, an integer.
    """
    if (position < self._lookback_distances[-1] and
        events[position] == self._one_hot_encoding.default_event):
      return (self._one_hot_encoding.num_classes +
              len(self._lookback_distances) - 1)

    # If last step repeated N bars ago.
    for i, lookback_distance in reversed(
        list(enumerate(self._lookback_distances))):
      lookback_position = position - lookback_distance
      if (lookback_position >= 0 and
          events[position] == events[lookback_position]):
        return self._one_hot_encoding.num_classes + i

    # If last step didn't repeat at one of the lookback positions, use the
    # specific event.
    return self._one_hot_encoding.encode_event(events[position])

  def class_index_to_event(self, class_index, events):
    """Returns the event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An int in the range [0, self.num_classes).
      events: The current event sequence.

    Returns:
      An event value.
    """
    # Repeat N bar ago.
    for i, lookback_distance in reversed(
        list(enumerate(self._lookback_distances))):
      if class_index == self._one_hot_encoding.num_classes + i:
        if len(events) < lookback_distance:
          return self._one_hot_encoding.default_event
        return events[-lookback_distance]

    # Return the event for that class index.
    return self._one_hot_encoding.decode_event(class_index)
