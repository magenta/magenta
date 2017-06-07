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
"""Utility functions for working with nested state structures."""

# internal imports

import numpy as np

from tensorflow.python.util import nest as tf_nest


def unbatch(batched_states, batch_size=1):
  """Splits a state structure into a list of individual states.

  Args:
    batched_states: A nested structure with entries whose first dimensions all
      equal `batch_size`.
    batch_size: The number of states in the batch.

  Returns:
    A list of `batch_size` state structures, each representing a single state.
  """
  return [extract_state(batched_states, i) for i in range(batch_size)]


def extract_state(batched_states, i):
  """Extracts a single state from a batch of states.

  Args:
    batched_states: A nested structure with entries whose first dimensions all
      equal N.
    i: The index of the state to extract.

  Returns:
    A tuple containing tensors (or tuples of tensors) of the same structure as
    rnn_nade_state, but containing only the state values that represent the
    state at index i. The tensors will now have the shape (1, N).
  """
  return tf_nest.map_structure(lambda x: x[i], batched_states)


def batch(states, batch_size=None):
  """Combines a collection of state structures into a batch, padding if needed.

  Args:
    states: A collection of individual nested state structures.
    batch_size: The desired final batch size. If the nested state structure
        that results from combining the states is smaller than this, it will be
        padded with zeros.
  Returns:
    A single state structure that results from stacking the structures in
    `states`, with padding if needed.

  Raises:
    ValueError: If the number of input states is larger than `batch_size`.
  """
  if batch_size and len(states) > batch_size:
    raise ValueError('Combined state is larger than the requested batch size')

  def stack_and_pad(*states):
    stacked = np.stack(states)
    if batch_size:
      stacked.resize([batch_size] + list(stacked.shape)[1:])
    return stacked
  return tf_nest.map_structure(stack_and_pad, *states)
