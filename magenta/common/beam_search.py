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
"""Beam search library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import heapq


# A beam entry containing a) the current sequence, b) a "state" containing any
# information needed to extend the sequence, and c) a score for the current
# sequence e.g. log-likelihood.
BeamEntry = collections.namedtuple('BeamEntry', ['sequence', 'state', 'score'])


def _generate_branches(beam_entries, generate_step_fn, branch_factor,
                       num_steps):
  """Performs a single iteration of branch generation for beam search.

  This method generates `branch_factor` branches for each sequence in the beam,
  where each branch extends the event sequence by `num_steps` steps (via calls
  to `generate_step_fn`). The resulting beam is returned.

  Args:
    beam_entries: A list of BeamEntry tuples, the current beam.
    generate_step_fn: A function that takes three parameters: a list of
        sequences, a list of states, and a list of scores, all of the same size.
        The function should generate a single step for each of the sequences and
        return the extended sequences, updated states, and updated (total)
        scores, as three lists. The function may modify the sequences, states,
        and scores in place, but should also return the modified values.
    branch_factor: The integer branch factor to use.
    num_steps: The integer number of steps to take per branch.

  Returns:
    The updated beam, with `branch_factor` times as many BeamEntry tuples.
  """
  if branch_factor > 1:
    branched_entries = beam_entries * branch_factor
    all_sequences = [copy.deepcopy(entry.sequence)
                     for entry in branched_entries]
    all_states = [copy.deepcopy(entry.state) for entry in branched_entries]
    all_scores = [entry.score for entry in branched_entries]
  else:
    # No need to make copies if there's no branching.
    all_sequences = [entry.sequence for entry in beam_entries]
    all_states = [entry.state for entry in beam_entries]
    all_scores = [entry.score for entry in beam_entries]

  for _ in range(num_steps):
    all_sequences, all_states, all_scores = generate_step_fn(
        all_sequences, all_states, all_scores)

  return [BeamEntry(sequence, state, score)
          for sequence, state, score
          in zip(all_sequences, all_states, all_scores)]


def _prune_branches(beam_entries, k):
  """Prune all but the `k` sequences with highest score from the beam."""
  indices = heapq.nlargest(k, range(len(beam_entries)),
                           key=lambda i: beam_entries[i].score)
  return [beam_entries[i] for i in indices]


def beam_search(initial_sequence, initial_state, generate_step_fn, num_steps,
                beam_size, branch_factor, steps_per_iteration):
  """Generates a sequence using beam search.

  Initially, the beam is filled with `beam_size` copies of the initial sequence.

  Each iteration, the beam is pruned to contain only the `beam_size` event
  sequences with highest score. Then `branch_factor` new event sequences are
  generated for each sequence in the beam. These new sequences are formed by
  extending each sequence in the beam by `steps_per_iteration` steps. So between
  a branching and a pruning phase, there will be `beam_size` * `branch_factor`
  active event sequences.

  After the final iteration, the single sequence in the beam with highest
  likelihood will be returned.

  The `generate_step_fn` function operates on lists of sequences + states +
  scores rather than single sequences. This is to allow for the possibility of
  batching.

  Args:
    initial_sequence: The initial sequence, a Python list-like object.
    initial_state: The state corresponding to the initial sequence, with any
        auxiliary information needed for extending the sequence.
    generate_step_fn: A function that takes three parameters: a list of
        sequences, a list of states, and a list of scores, all of the same size.
        The function should generate a single step for each of the sequences and
        return the extended sequences, updated states, and updated (total)
        scores, as three lists.
    num_steps: The integer length in steps of the final sequence, after
        generation.
    beam_size: The integer beam size to use.
    branch_factor: The integer branch factor to use.
    steps_per_iteration: The integer number of steps to take per iteration.

  Returns:
    A tuple containing a) the highest-scoring sequence as computed by the beam
    search, b) the state corresponding to this sequence, and c) the score of
    this sequence.
  """
  sequences = [copy.deepcopy(initial_sequence) for _ in range(beam_size)]
  states = [copy.deepcopy(initial_state) for _ in range(beam_size)]
  scores = [0] * beam_size

  beam_entries = [BeamEntry(sequence, state, score)
                  for sequence, state, score
                  in zip(sequences, states, scores)]

  # Choose the number of steps for the first iteration such that subsequent
  # iterations can all take the same number of steps.
  first_iteration_num_steps = (num_steps - 1) % steps_per_iteration + 1

  beam_entries = _generate_branches(
      beam_entries, generate_step_fn, branch_factor, first_iteration_num_steps)

  num_iterations = (num_steps -
                    first_iteration_num_steps) // steps_per_iteration

  for _ in range(num_iterations):
    beam_entries = _prune_branches(beam_entries, k=beam_size)
    beam_entries = _generate_branches(
        beam_entries, generate_step_fn, branch_factor, steps_per_iteration)

  # Prune to the single best beam entry.
  beam_entry = _prune_branches(beam_entries, k=1)[0]

  return beam_entry.sequence, beam_entry.state, beam_entry.score
