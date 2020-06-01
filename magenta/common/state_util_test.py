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

"""Tests for state_util."""

from magenta.common import state_util
import numpy as np
import tensorflow.compat.v1 as tf


class StateUtilTest(tf.test.TestCase):

  def setUp(self):
    self._unbatched_states = [
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            (np.array([7, 8]), np.array([9])),
            np.array([[10], [11]])
        ),
        (
            np.array([[12, 13, 14], [15, 16, 17]]),
            (np.array([18, 19]), np.array([20])),
            np.array([[21], [22]])
        )]

    self._batched_states = (
        np.array([[[1, 2, 3], [4, 5, 6]],
                  [[12, 13, 14], [15, 16, 17]],
                  [[0, 0, 0], [0, 0, 0]]]),
        (np.array([[7, 8], [18, 19], [0, 0]]), np.array([[9], [20], [0]])),
        np.array([[[10], [11]], [[21], [22]], [[0], [0]]]))

  def _assert_sructures_equal(self, struct1, struct2):
    tf.nest.assert_same_structure(struct1, struct2)
    for a, b in zip(tf.nest.flatten(struct1), tf.nest.flatten(struct2)):
      np.testing.assert_array_equal(a, b)

  def testBatch(self):
    # Combine these two states, which each have a batch size of 2, together.
    # Request a batch_size of 5, which means that a new batch of all zeros will
    # be created.
    batched_states = state_util.batch(self._unbatched_states, batch_size=3)

    self._assert_sructures_equal(self._batched_states, batched_states)

  def testBatch_Single(self):
    batched_state = state_util.batch(self._unbatched_states[0:1], batch_size=1)
    expected_batched_state = (
        np.array([[[1, 2, 3], [4, 5, 6]]]),
        (np.array([[7, 8]]), np.array([[9]])),
        np.array([[[10], [11]]])
    )

    self._assert_sructures_equal(expected_batched_state, batched_state)

  def test_Unbatch(self):
    unbatched_states = state_util.unbatch(self._batched_states, batch_size=2)

    self._assert_sructures_equal(self._unbatched_states, unbatched_states)

  def test_ExtractState(self):
    extracted_state = state_util.extract_state(self._batched_states, 1)

    self._assert_sructures_equal(self._unbatched_states[1], extracted_state)


if __name__ == '__main__':
  tf.test.main()
