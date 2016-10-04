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
"""Abstract base classes for working with musical event sequences.

The abstract `EventSequence` class is an interface for a sequence of musical
events. The `SimpleEventSequence` class is a basic implementation of this
interface.

The `EventsEncoderDecoder` is an abstract class for translating between event
sequences and model data.
"""

import abc
import copy
import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin

from magenta.common import sequence_example_lib
from magenta.music import constants


DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER
STANDARD_PPQ = constants.STANDARD_PPQ


class NonIntegerStepsPerBarException(Exception):
  pass


class EventSequence(object):
  """Stores a quantized stream of events.

  EventSequence is an abstract class to use as an interface for interacting
  with musical event sequences. Concrete implementations SimpleEventSequence
  (and its descendants Melody and ChordProgression) and LeadSheet represent
  sequences of musical events of particular types. In all cases, model-specific
  code is responsible for converting this representation to SequenceExample
  protos for TensorFlow.

  EventSequence represents an iterable object. Simply iterate to retrieve the
  events.

  Attributes:
    start_step: The offset of the first step of the sequence relative to the
        beginning of the source sequence. Should always be the first step of a
        bar.
    end_step: The offset to the beginning of the bar following the last step
       of the sequence relative to the beginning of the source sequence. Will
       always be the first step of a bar.
    steps_per_quarter: Number of steps in in a quarter note.
    steps_per_bar: Number of steps in a bar (measure) of music.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def start_step(self):
    pass

  @abc.abstractproperty
  def end_step(self):
    pass

  @abc.abstractproperty
  def steps_per_quarter(self):
    pass

  @abc.abstractproperty
  def steps_per_bar(self):
    pass

  @abc.abstractmethod
  def append_event(self, event):
    """Appends event to the end of the sequence and increments the end step.

    Args:
      event: The event to append to the end.
    """
    pass

  @abc.abstractmethod
  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads with `pad_event` to make the
    sequence the specified length. If it is too long, it will be truncated to
    the requested length.

    Args:
      steps: How many steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    pass

  @abc.abstractmethod
  def increase_resolution(self, k):
    """Increase the resolution of an event sequence.

    Increases the resolution of an EventSequence object by a factor of `k`.

    Args:
      k: An integer, the factor by which to increase the resolution of the
          event sequence.
    """


class SimpleEventSequence(EventSequence):
  """Stores a quantized stream of events.

  This class can be instantiated, but its main purpose is to serve as a base
  class for Melody, ChordProgression, and any other simple stream of musical
  events.
  """

  def __init__(self, pad_event, events=None, start_step=0,
               steps_per_bar=DEFAULT_STEPS_PER_BAR,
               steps_per_quarter=DEFAULT_STEPS_PER_QUARTER):
    """Construct a SimpleEventSequence.

    If `events` is specified, instantiate with the provided event list.
    Otherwise, create an empty SimpleEventSequence.

    Args:
      pad_event: Event value to use when padding sequences.
      events: List of events to instantiate with.
      start_step: The integer starting step offset.
      steps_per_bar: The number of steps in a bar.
      steps_per_quarter: The number of steps in a quarter note.
    """
    self._pad_event = pad_event
    if events is not None:
      self._from_event_list(events, start_step=start_step,
                            steps_per_bar=steps_per_bar,
                            steps_per_quarter=steps_per_quarter)
    else:
      self._events = []
      self._steps_per_bar = steps_per_bar
      self._steps_per_quarter = steps_per_quarter
      self._start_step = start_step
      self._end_step = start_step

  def _reset(self):
    """Clear events and reset object state."""
    self._events = []
    self._steps_per_bar = DEFAULT_STEPS_PER_BAR
    self._steps_per_quarter = DEFAULT_STEPS_PER_QUARTER
    self._start_step = 0
    self._end_step = 0

  def _from_event_list(self, events, start_step=0,
                       steps_per_bar=DEFAULT_STEPS_PER_BAR,
                       steps_per_quarter=DEFAULT_STEPS_PER_QUARTER):
    """Initializes with a list of event values and sets attributes."""
    self._events = list(events)
    self._start_step = start_step
    self._end_step = start_step + len(self)
    self._steps_per_bar = steps_per_bar
    self._steps_per_quarter = steps_per_quarter

  def __iter__(self):
    """Return an iterator over the events in this SimpleEventSequence.

    Returns:
      Python iterator over events.
    """
    return iter(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __getslice__(self, i, j):
    """Returns the events in the given slice range."""
    return self._events[i:j]

  def __len__(self):
    """How many events are in this SimpleEventSequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def __deepcopy__(self, unused_memo=None):
    return type(self)(pad_event=self._pad_event,
                      events=copy.deepcopy(self._events),
                      start_step=self.start_step,
                      steps_per_bar=self.steps_per_bar,
                      steps_per_quarter=self.steps_per_quarter)

  def __eq__(self, other):
    if not isinstance(other, SimpleEventSequence):
      return False
    return (list(self) == list(other) and
            self.steps_per_bar == other.steps_per_bar and
            self.steps_per_quarter == other.steps_per_quarter and
            self.start_step == other.start_step and
            self.end_step == other.end_step)

  @property
  def start_step(self):
    return self._start_step

  @property
  def end_step(self):
    return self._end_step

  @property
  def steps_per_bar(self):
    return self._steps_per_bar

  @property
  def steps_per_quarter(self):
    return self._steps_per_quarter

  def append_event(self, event):
    """Appends event to the end of the sequence and increments the end step.

    Args:
      event: The event to append to the end.
    """
    self._events.append(event)
    self._end_step += 1

  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads to make the sequence the
    specified length. If it is too long, it will be truncated to the requested
    length.

    Args:
      steps: How many steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if steps > len(self):
      if from_left:
        self._events[:0] = [self._pad_event] * (steps - len(self))
      else:
        self._events.extend([self._pad_event] * (steps - len(self)))
    else:
      if from_left:
        del self._events[0:-steps]
      else:
        del self._events[steps:]

    if from_left:
      self._start_step = self._end_step - steps
    else:
      self._end_step = self._start_step + steps

  def increase_resolution(self, k, fill_event=None):
    """Increase the resolution of an event sequence.

    Increases the resolution of a SimpleEventSequence object by a factor of
    `k`.

    Args:
      k: An integer, the factor by which to increase the resolution of the
          event sequence.
      fill_event: Event value to use to extend each low-resolution event. If
          None, each low-resolution event value will be repeated `k` times.
    """
    if fill_event is None:
      fill = lambda event: [event] * k
    else:
      fill = lambda event: [event] + [fill_event] * (k - 1)

    new_events = []
    for event in self._events:
      new_events += fill(event)

    self._events = new_events
    self._start_step *= k
    self._end_step *= k
    self._steps_per_bar *= k
    self._steps_per_quarter *= k


class EventsEncoderDecoder(object):
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

  The `events_to_input`, `events_to_label`, and `class_index_to_event` methods
  must be overwritten to be specific to your model.
  """
  __metaclass__ = abc.ABCMeta

  def _encode(self, events):
    """Returns a SequenceExample for the given event sequence.

    Args:
      events: An EventSequence object.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    inputs = []
    labels = []
    for i in range(len(events) - 1):
      inputs.append(self.events_to_input(events, i))
      labels.append(self.events_to_label(events, i + 1))
    return sequence_example_lib.make_sequence_example(inputs, labels)

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

  @abc.abstractmethod
  def events_to_input(self, events, position):
    """Returns the input vector for the event at the given position.

    Args:
      events: An EventSequence object.
      position: An integer event position in the sequence.

    Returns:
      An input vector, a self.input_size length list of floats.
    """
    pass

  @abc.abstractmethod
  def events_to_label(self, events, position):
    """Returns the label for the event at the given position.

    Args:
      events: An EventSequence object.
      position: An integer event position in the sequence.

    Returns:
      A label, an integer in the range [0, self.num_classes).
    """
    pass

  def get_inputs_batch(self, event_sequences, full_length=False):
    """Returns an inputs batch for the given event sequences.

    Args:
      event_sequences: A list of EventSequence objects.
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

  @abc.abstractmethod
  def class_index_to_event(self, class_index, events):
    """Returns the event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: An EventSequence object.

    Returns:
      An event value.
    """
    pass

  def extend_event_sequences(self, event_sequences, softmax):
    """Extends the event_sequences by sampling the softmax probabilities.

    Args:
      event_sequences: A list of EventSequence objects.
      softmax: A list of softmax probability vectors. The list of softmaxes
          should be the same length as the list of event_sequences.
    """
    num_classes = len(softmax[0][0])
    for i in xrange(len(event_sequences)):
      chosen_class = np.random.choice(num_classes, p=softmax[i][-1])
      event = self.class_index_to_event(chosen_class, event_sequences[i])
      event_sequences[i].append_event(event)
