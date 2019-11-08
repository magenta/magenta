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

"""Tests for create_dataset_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from magenta.models.onsets_frames_transcription import create_dataset_lib

import tensorflow.compat.v1 as tf


class CreateDatasetLibTest(tf.test.TestCase):

  def test_generate_unique_mixes(self):
    sourceid_to_exids = [('source1', 'a'), ('source1', 'b'),
                         ('source2', 'c'), ('source2', 'd')]
    exid_to_mixids = create_dataset_lib.generate_mixes(
        val='unused', num_mixes=100, sourceid_to_exids=sourceid_to_exids)
    mix_ids = set(itertools.chain(*exid_to_mixids.values()))
    # Requested 100, but there are only 4 unique mixes, so that's how many
    # we should end up with.
    self.assertEqual(4, len(mix_ids))

  def test_generate_num_mixes(self):
    sourceid_to_exids = [('source1', 'a'), ('source1', 'b'), ('source1', 'c'),
                         ('source2', 'd'), ('source2', 'e'), ('source2', 'f')]
    exid_to_mixids = create_dataset_lib.generate_mixes(
        val='unused', num_mixes=2, sourceid_to_exids=sourceid_to_exids)
    mix_ids = set(itertools.chain(*exid_to_mixids.values()))
    # Ensure we get the number of mixes we requested even when more unique mixes
    # would be possible.
    self.assertEqual(2, len(mix_ids))

  def test_unique_mixes_duplicate_sources(self):
    sourceid_to_exids = [('source1', 'a'), ('source1', 'b'), ('source1', 'c'),
                         ('source2', 'a'), ('source2', 'b'), ('source2', 'c'),
                         ('source3', 'a'), ('source3', 'b'), ('source3', 'c')]
    exid_to_mixids = create_dataset_lib.generate_mixes(
        val='unused', num_mixes=100, sourceid_to_exids=sourceid_to_exids)
    mix_ids = set(itertools.chain(*exid_to_mixids.values()))
    # There are only 3 unique ids, but we're request mixes of 3 items, so only
    # 1 unique mix is possible.
    self.assertEqual(1, len(mix_ids))

if __name__ == '__main__':
  tf.test.main()
