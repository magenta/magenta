# pylint: skip-file
# TODO(iansimon): Enable when Apache Beam supports Python 3.
# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for Score2Perf datagen using beam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import apache_beam as beam
import tensorflow as tf

from magenta.models.score2perf import datagen_beam
from magenta.models.score2perf import music_encoders
from magenta.music import testing_lib
from magenta.protobuf import music_pb2


class GenerateExamplesTest(tf.test.TestCase):

  def testGenerateExamples(self):
    ns = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        ns, 0, [(60, 100, 0.0, 1.0), (64, 100, 1.0, 2.0), (67, 127, 2.0, 3.0)])
    input_transform = beam.transforms.Create([('0', ns.SerializeToString())])
    output_dir = tempfile.mkdtemp()
    encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=100,
        num_velocity_bins=32,
        min_pitch=21,
        max_pitch=108)

    datagen_beam.generate_examples(
        input_transform=input_transform,
        output_dir=output_dir,
        problem_name='test_problem',
        splits={'train': 1.0},
        min_hop_size_seconds=3.0,
        max_hop_size_seconds=3.0,
        min_pitch=21,
        max_pitch=108,
        num_replications=1,
        encode_performance_fn=encoder.encode_note_sequence)


if __name__ == '__main__':
  tf.test.main()
