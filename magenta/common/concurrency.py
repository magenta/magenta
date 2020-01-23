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

"""Utility functions for concurrency."""

import functools
import threading
import time


def serialized(func):
  """Decorator to provide mutual exclusion for method using _lock attribute."""

  @functools.wraps(func)
  def serialized_method(self, *args, **kwargs):
    lock = getattr(self, '_lock')
    with lock:
      return func(self, *args, **kwargs)

  return serialized_method


class Singleton(type):
  """A threadsafe singleton meta-class."""

  _singleton_lock = threading.RLock()
  _instances = {}

  def __call__(cls, *args, **kwargs):
    with Singleton._singleton_lock:
      if cls not in cls._instances:
        cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
      return cls._instances[cls]


class Sleeper(object):
  """A threadsafe, singleton wrapper for time.sleep that improves accuracy.

  The time.sleep function is inaccurate and sometimes returns early or late. To
  improve accuracy, this class sleeps for a shorter time than requested and
  enters a spin lock for the remainder of the requested time.

  The offset is automatically calibrated based on when the thread is actually
  woken from sleep by increasing/decreasing the offset halfway between its
  current value and `_MIN_OFFSET`.

  Accurate to approximately 5ms after some initial burn-in.

  Args:
    initial_offset: The initial amount to shorten the time.sleep call in float
        seconds.
  Raises:
    ValueError: When `initial_offset` is less than `_MIN_OFFSET`.
  """
  __metaclass__ = Singleton

  _MIN_OFFSET = 0.001

  def __init__(self, initial_offset=0.001):
    if initial_offset < Sleeper._MIN_OFFSET:
      raise ValueError(
          '`initial_offset` must be at least %f. Got %f.' %
          (Sleeper._MIN_OFFSET, initial_offset))
    self._lock = threading.RLock()
    self.offset = initial_offset

  @property
  @serialized
  def offset(self):
    """Threadsafe accessor for offset attribute."""
    return self._offset

  @offset.setter
  @serialized
  def offset(self, value):
    """Threadsafe mutator for offset attribute."""
    self._offset = value

  def sleep(self, seconds):
    """Sleeps the requested number of seconds."""
    wake_time = time.time() + seconds
    self.sleep_until(wake_time)

  def sleep_until(self, wake_time):
    """Sleeps until the requested time."""
    delta = wake_time - time.time()

    if delta <= 0:
      return

    # Copy the current offset, since it might change.
    offset_ = self.offset

    if delta > offset_:
      time.sleep(delta - offset_)

    remaining_time = time.time() - wake_time
    # Enter critical section for updating the offset.
    with self._lock:
      # Only update if the current offset value is what was used in this call.
      if self.offset == offset_:
        offset_delta = (offset_ - Sleeper._MIN_OFFSET) / 2
        if remaining_time > 0:
          self.offset -= offset_delta
        elif remaining_time < -Sleeper._MIN_OFFSET:
          self.offset += offset_delta

    while time.time() < wake_time:
      pass
