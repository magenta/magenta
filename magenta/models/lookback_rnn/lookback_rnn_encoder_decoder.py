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
"""A MelodyEncoderDecoder specific to the lookback RNN model."""

# internal imports
from magenta.lib import melodies_lib

NUM_SPECIAL_EVENTS = melodies_lib.NUM_SPECIAL_EVENTS
NO_EVENT = melodies_lib.NO_EVENT

MIN_NOTE = 48  # Inclusive
MAX_NOTE = 84  # Exclusive
TRANSPOSE_TO_KEY = 0  # C Major


class MelodyEncoderDecoder(melodies_lib.MelodyEncoderDecoder):
  """A MelodyEncoderDecoder specific to the lookback RNN model."""

  def __init__(self):
    """Initializes the MelodyEncoderDecoder.

    Sets self.input_size the the size of the input vectors that this model uses.
    Sets self.num_classes to the number of classes the labels can be.
    These values are used when building the TensorFlow graph for this model.
    """
    super(MelodyEncoderDecoder, self).__init__(MIN_NOTE, MAX_NOTE,
                                               TRANSPOSE_TO_KEY)
    self.num_model_events = self.max_note - self.min_note + NUM_SPECIAL_EVENTS
    self._input_size = 3 * self.num_model_events + 7
    self._num_classes = self.num_model_events + 2

  @property
  def input_size(self):
    return self._input_size

  @property
  def num_classes(self):
    return self._num_classes

  def melody_event_to_model_event(self, melody_event):
    """Collapses a melody event value into a zero-based index range.

    Args:
      melody_event: A Melody event value. -2 = no event,
          -1 = note-off event, [0, 127] = note-on event for that midi pitch.

    Returns:
      An int in the range [0, self._num_model_events). 0 = no event,
      1 = note-off event, [2, self._num_model_events) = note-on event for
      that pitch relative to the [self._min_note, self._max_note) range.
    """
    if melody_event < 0:
      return melody_event + NUM_SPECIAL_EVENTS
    return melody_event - self.min_note + NUM_SPECIAL_EVENTS

  def model_event_to_melody_event(self, model_event):
    """Expands a zero-based index value to its equivalent melody event value.

    Args:
      model_event: An int in the range [0, self._num_model_events).
          0 = no event, 1 = note-off event,
          [2, self._num_model_events) = note-on event for that pitch relative
          to the [self._min_note, self._max_note) range.

    Returns:
      A Melody event value. -2 = no event, -1 = note-off event,
      [0, 127] = note-on event for that midi pitch.
    """
    if model_event < NUM_SPECIAL_EVENTS:
      return model_event - NUM_SPECIAL_EVENTS
    return model_event - NUM_SPECIAL_EVENTS + self.min_note

  def melody_to_input(self, melody):
    """Returns the input vector for the last event in the melody.

    Returns a self.input_size length list of floats. If MIN_NOTE = 48 and
    MAX_NOTE = 84, self.input_size = 121. Each index represents a different
    input signal to the model.

    Indices [0, 121):
    [0, 38): Event of current step.
    [38, 76): Event of next step if repeating 1 bar ago.
    [76, 114): Event of next step if repeating 2 bars ago.
    114: 16th note binary counter.
    115: 8th note binary counter.
    116: 4th note binary counter.
    117: Half note binary counter.
    118: Whole note binary counter.
    119: The current step is repeating 1 bar ago.
    120: The current step is repeating 2 bars ago.

    Args:
      melody: A melodies_lib.Melody object.

    Returns:
      An input vector, an self.input_size length list of floats.
    """
    input_ = [0.0] * self._input_size

    # Last event.
    model_event = self.melody_event_to_model_event(
        melody.events[-1] if len(melody) >= 1 else NO_EVENT)
    input_[model_event] = 1.0

    # Next event if repeating 1 bar ago.
    model_event = self.melody_event_to_model_event(
        melody.events[-16] if len(melody) >= 16 else NO_EVENT)
    input_[self.num_model_events + model_event] = 1.0

    # Next event if repeating 2 bars ago.
    model_event = self.melody_event_to_model_event(
        melody.events[-32] if len(melody) >= 32 else NO_EVENT)
    input_[2 * self.num_model_events + model_event] = 1.0

    # Binary time counter.
    i = len(melody) - 1
    input_[3 * self.num_model_events + 0] = 1.0 if i % 2 else -1.0
    input_[3 * self.num_model_events + 1] = 1.0 if i / 2 % 2 else -1.0
    input_[3 * self.num_model_events + 2] = 1.0 if i / 4 % 2 else -1.0
    input_[3 * self.num_model_events + 3] = 1.0 if i / 8 % 2 else -1.0
    input_[3 * self.num_model_events + 4] = 1.0 if i / 16 % 2 else -1.0

    # Last event is repeating 1 bar ago.
    if len(melody) >= 17 and melody.events[-1] == melody.events[-17]:
      input_[3 * self.num_model_events + 5] = 1.0

    # Last event is repeating 2 bars ago.
    if len(melody) >= 33 and melody.events[-1] == melody.events[-33]:
      input_[3 * self.num_model_events + 6] = 1.0

    return input_

  def melody_to_label(self, melody):
    """Returns the label for the last event in the melody.

    Returns an int the range [0, self.num_classes). Indices in the range
    [0, self.num_model_events) map to standard midi events. Indices
    self.num_model_events and self.num_model_events + 1 are signals to repeat
    events from earlier in the melody.

    If MIN_NOTE = 48 and MAX_NOTE = 84, self.num_classes will = 40,
    self.num_model_events will = 38, and the values will be as follows.
    Values [0, 40):
      [0, 38): Event of the last step in the melody, if not repeating 1 or 2
               bars ago.
      38: If the last event in the melody is repeating 1 bar ago, if not
          repeating 2 bars ago.
      39: If the last event in the melody is repeating 2 bars ago.

    Args:
      melody: A melodies_lib.Melody object.

    Returns:
      A label, an int.
    """
    # If last step repeated 2 bars ago.
    if ((len(melody.events) <= 32 and melody.events[-1] == NO_EVENT) or
        (len(melody.events) > 32 and melody.events[-1] == melody.events[-33])):
      return self.num_model_events + 1

    # If last step repeated 1 bar ago.
    if len(melody.events) > 16 and melody.events[-1] == melody.events[-17]:
      return self.num_model_events

    # If last step didn't repeat 1 or 2 bars ago, use the specific event.
    return self.melody_event_to_model_event(melody.events[-1])

  def class_index_to_melody_event(self, class_index, melody):
    """Returns the melody event for the given class index.

    This is the reverse process of the self.melody_to_label method.

    Args:
      class_index: An int in the range [0, self.num_classes).
      melody: The melodies_lib.Melody events list of the current melody.

    Returns:
      A melodies_lib.Melody event value.
    """
    # Repeat 1 bar ago.
    if class_index == self.num_model_events + 1:
      return NO_EVENT if len(melody) < 32 else melody.events[-32]

    # Repeat 2 bars ago.
    if class_index == self.num_model_events:
      return NO_EVENT if len(melody) < 16 else melody.events[-16]

    # Return the melody event for that class index.
    return self.model_event_to_melody_event(class_index)
