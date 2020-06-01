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

"""Tests for beam search."""

from magenta.common import beam_search
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class BeamSearchTest(tf.test.TestCase):

  def _generate_step_fn(self, sequences, states, scores):
    # This acts as a binary counter for testing purposes. For scoring, zeros
    # accumulate value exponentially in the state, ones "cash in". The highest-
    # scoring sequence would be all zeros followed by a single one.
    value = 0
    for i, seq in enumerate(sequences):
      seq.append(value)
      if value == 0:
        states[i] *= 2
      else:
        scores[i] += states[i]
        states[i] = 1
      if (i - 1) % (2 ** len(seq)) == 0:
        value = 1 - value
    return sequences, states, scores

  def testNoBranchingSingleStepPerIteration(self):
    sequence, state, score = beam_search(
        initial_sequence=[], initial_state=1,
        generate_step_fn=self._generate_step_fn, num_steps=5, beam_size=1,
        branch_factor=1, steps_per_iteration=1)

    # The generator should emit all zeros, as only a single sequence is ever
    # considered so the counter doesn't reach one.
    self.assertEqual(sequence, [0, 0, 0, 0, 0])
    self.assertEqual(state, 32)
    self.assertEqual(score, 0)

  def testNoBranchingMultipleStepsPerIteration(self):
    sequence, state, score = beam_search(
        initial_sequence=[], initial_state=1,
        generate_step_fn=self._generate_step_fn, num_steps=5, beam_size=1,
        branch_factor=1, steps_per_iteration=2)

    # Like the above case, the counter should never reach one as only a single
    # sequence is ever considered.
    self.assertEqual(sequence, [0, 0, 0, 0, 0])
    self.assertEqual(state, 32)
    self.assertEqual(score, 0)

  def testBranchingSingleBeamEntry(self):
    sequence, state, score = beam_search(
        initial_sequence=[], initial_state=1,
        generate_step_fn=self._generate_step_fn, num_steps=5, beam_size=1,
        branch_factor=32, steps_per_iteration=1)

    # Here the beam search should greedily choose ones.
    self.assertEqual(sequence, [1, 1, 1, 1, 1])
    self.assertEqual(state, 1)
    self.assertEqual(score, 5)

  def testNoBranchingMultipleBeamEntries(self):
    sequence, state, score = beam_search(
        initial_sequence=[], initial_state=1,
        generate_step_fn=self._generate_step_fn, num_steps=5, beam_size=32,
        branch_factor=1, steps_per_iteration=1)

    # Here the beam has enough capacity to find the optimal solution without
    # branching.
    self.assertEqual(sequence, [0, 0, 0, 0, 1])
    self.assertEqual(state, 1)
    self.assertEqual(score, 16)


if __name__ == '__main__':
  tf.test.main()
