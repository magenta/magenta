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

# Lint as: python3
"""Library functions for creating dataset."""

import collections
import copy

import numpy as np


def _dedup_mixes(mixes):
  # Retain only mixes with unique lists of examples.
  mixes_unique_examples = [tuple(sorted(mix)) for mix in mixes
                           if len(set(mix)) == len(mix)]
  # Retain only unique mixes.
  unique_mixes = set(mixes_unique_examples)
  return unique_mixes


def generate_mixes_using_all_examples(sourceid_to_exids, rs):
  """Generates mixes guaranteed to use every example at least once."""
  # Brute force way to ensure every example from every source is used at least
  # once. Just goes through every source, selects each example, and generates
  # a mix using random examples from other sources.
  # Number of final outputs is variable, depending on how many end up getting
  # deduped.
  sourceid_to_exids = copy.deepcopy(sourceid_to_exids)
  mixes = []
  for current_source in sourceid_to_exids:
    for current_exid in sourceid_to_exids[current_source]:
      mix = [current_exid]
      for other_source in sourceid_to_exids.keys() - set([current_source]):
        rs.shuffle(sourceid_to_exids[other_source])
        mix.append(next(x for x in sourceid_to_exids[other_source]
                        if x not in mix))
      mixes.append(mix)
  return _dedup_mixes(mixes)


def generate_mixes_random_examples(sourceid_to_exids, num_mixes, rs):
  # Generate 5x the number of mixes requested. We'll dedup them later, so
  # this helps increase the chance we'll end up with as many as requested.
  # This isn't guaranteed to always work, but it almost always will and is
  # easier than trying to generate every possible mix and sampling from that
  # since the numbers blow up pretty quickly.
  mixes = zip(
      *[rs.choice(k, num_mixes * 5, replace=True).tolist()
        for k in sourceid_to_exids.values()])
  return _dedup_mixes(mixes)


def generate_mixes(val, num_mixes, sourceid_to_exids, seed=0):
  """Generate lists of Example IDs to be mixed."""
  del val
  rs = np.random.RandomState(seed=seed)  # Make the selection deterministic
  sourceid_to_exids_dict = collections.defaultdict(list)
  for sourceid, exid in sourceid_to_exids:
    sourceid_to_exids_dict[sourceid].append(exid)

  # First, generate mixes ensuring that every example is used at least once.
  mixes = generate_mixes_using_all_examples(sourceid_to_exids_dict, rs)
  if len(mixes) > num_mixes:
    raise ValueError(
        'Requested {} mixes, but {} mixes needed to use all examples.'.format(
            num_mixes, len(mixes)))
  else:
    # We need more mixes, so generate some random ones, then crop the excess
    # from the random mixes, ensuring that we preserve the mixes that use every
    # example at least once.
    num_mixes_remaining = num_mixes - len(mixes)
    random_example_mixes = generate_mixes_random_examples(
        sourceid_to_exids_dict, num_mixes_remaining, rs)
    # If the random mixes happened to include any of the mixes we already
    # generated, remove them.
    random_example_mixes -= mixes
    # Do a random crop.
    random_example_mixes_list = list(random_example_mixes)
    rs.shuffle(random_example_mixes_list)
    # Combine the two mix lists.
    unique_mixes = list(mixes) + random_example_mixes_list[:num_mixes_remaining]

  keyed_mixes = dict(enumerate(unique_mixes))
  exid_to_mixids = collections.defaultdict(list)
  for mixid, exids in keyed_mixes.items():
    for exid in exids:
      exid_to_mixids[exid].append(mixid)
  return exid_to_mixids
