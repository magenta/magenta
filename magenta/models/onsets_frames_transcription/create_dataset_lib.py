# Copyright 2019 The Magenta Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np


def generate_mixes(val, num_mixes, sourceid_to_exids, seed=0):
  """Generate lists of Example IDs to be mixed."""
  del val
  rs = np.random.RandomState(seed=seed)  # Make the selection deterministic
  sourceid_to_exids_dict = collections.defaultdict(list)
  for sourceid, exid in sourceid_to_exids:
    sourceid_to_exids_dict[sourceid].append(exid)
  # Generate 5x the number of mixes requested. We'll dedup them later, so
  # this helps increase the chance we'll end up with as many as requested.
  # This isn't guaranteed to always work, but it almost always will and is
  # easier than trying to generate every possible mix and sampling from that
  # since the numbers blow up pretty quickly.
  mixes = zip(
      *[rs.choice(k, num_mixes * 5, replace=True).tolist()
        for k in sourceid_to_exids_dict.values()])
  # Retain only mixes with unique lists of examples.
  mixes_unique_examples = [tuple(sorted(mix)) for mix in mixes
                           if len(set(mix)) == len(mix)]
  # Retain only unique mixes.
  unique_mixes = list(set(mixes_unique_examples))
  # Limit to only num_mixes.
  rs.shuffle(unique_mixes)
  unique_mixes = unique_mixes[:num_mixes]

  keyed_mixes = dict(enumerate(unique_mixes))
  exid_to_mixids = collections.defaultdict(list)
  for mixid, exids in keyed_mixes.items():
    for exid in exids:
      exid_to_mixids[exid].append(mixid)
  return exid_to_mixids
