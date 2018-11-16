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

"""Test for splitting of files / midis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from magenta.models.onsets_frames_transcription import onsets_frames_transcription_create_dataset as create_dataset
from magenta.protobuf import music_pb2


class CreateDatasetTest(tf.test.TestCase):

  def testSplitMidi(self):
    sequence = music_pb2.NoteSequence()
    sequence.notes.add(pitch=60, start_time=1.0, end_time=2.9)
    sequence.notes.add(pitch=60, start_time=8.0, end_time=11.0)
    sequence.notes.add(pitch=60, start_time=14.0, end_time=17.0)
    sequence.notes.add(pitch=60, start_time=20.0, end_time=23.0)
    sequence.total_time = 25.

    sample_rate = 160
    samples = np.zeros(sample_rate * int(sequence.total_time))
    splits = create_dataset.find_split_points(sequence, samples,
                                              sample_rate, 0, 3)

    self.assertEqual(splits, [0., 3., 6., 9., 12., 15., 18., 21., 24., 25.])

    samples[int(8.5 * sample_rate)] = 1
    samples[int(8.5 * sample_rate) + 1] = -1
    splits = create_dataset.find_split_points(sequence, samples,
                                              sample_rate, 0, 3)

    self.assertEqual(splits, [
        0.0, 3.0, 6.0, 8.50625, 11.50625, 14.50625, 17.50625, 20.50625,
        23.50625, 25.
    ])


if __name__ == '__main__':
  tf.test.main()
