# Copyright 2024 The Magenta Authors.
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

"""Encode and decode events."""

import dataclasses
import math

from typing import List, Tuple


@dataclasses.dataclass
class EventRange:
  type: str
  min_value: int
  max_value: int


@dataclasses.dataclass
class Event:
  type: str
  value: int


def _round_num_classes(num_classes: int) -> int:
  return 128 * math.ceil(num_classes / 128)


class Codec:
  """Encode and decode events.

  Useful for declaring what certain ranges of a vocabulary should be used for.
  This is intended to be used from Python before encoding or after decoding with
  PassThroughVocabulary.

  Takes care of ensuring that 0 = PAD and 1 = EOS.
  """

  def __init__(
      self,
      event_ranges: List[EventRange],
  ):
    """Define Codec.

    Args:
      event_ranges: Other supported event types and their ranges.
    """
    self._event_ranges = [
        EventRange('pad', 1, 1),
        EventRange('eos', 1, 1),
    ] + event_ranges
    rounded_num_classes = _round_num_classes(self.num_classes)
    if rounded_num_classes > self.num_classes:
      self._event_ranges += [
          EventRange('round', 1, rounded_num_classes - self.num_classes)
      ]
    # Ensure all event types have unique names.
    assert len(self._event_ranges) == len(
        set([er.type for er in self._event_ranges])
    )

  @property
  def num_classes(self) -> int:
    return sum(er.max_value - er.min_value + 1 for er in self._event_ranges)

  def encode_event(self, event: Event) -> int:
    """Encode an event to an index."""
    offset = 0
    for er in self._event_ranges:
      if event.type == er.type:
        if not er.min_value <= event.value <= er.max_value:
          raise ValueError(
              f'Event value {event.value} is not within valid range '
              f'[{er.min_value}, {er.max_value}] for type {event.type}'
          )
        return offset + event.value - er.min_value
      offset += er.max_value - er.min_value + 1

    raise ValueError(f'Unknown event type: {event.type}')

  def event_type_range(self, event_type: str) -> Tuple[int, int]:
    """Return [min_id, max_id] for an event type."""
    offset = 0
    for er in self._event_ranges:
      if event_type == er.type:
        return offset, offset + (er.max_value - er.min_value)
      offset += er.max_value - er.min_value + 1

    raise ValueError(f'Unknown event type: {event_type}')

  def decode_event_index(self, index: int) -> Event:
    """Decode an event index to an Event."""
    offset = 0
    for er in self._event_ranges:
      if offset <= index <= offset + er.max_value - er.min_value:
        return Event(type=er.type, value=er.min_value + index - offset)
      offset += er.max_value - er.min_value + 1

    raise ValueError(f'Unknown event index: {index}')
