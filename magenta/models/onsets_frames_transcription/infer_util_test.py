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

"""Tests for metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.onsets_frames_transcription import infer_util

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class InferUtilTest(tf.test.TestCase):

  def testProbsToPianorollViterbi(self):
    frame_probs = np.array([[0.2, 0.1], [0.5, 0.1], [0.5, 0.1], [0.8, 0.1]])
    onset_probs = np.array([[0.1, 0.1], [0.1, 0.1], [0.9, 0.1], [0.1, 0.1]])
    pianoroll = infer_util.probs_to_pianoroll_viterbi(frame_probs, onset_probs)
    np.testing.assert_array_equal(
        [[False, False], [False, False], [True, False], [True, False]],
        pianoroll)


if __name__ == '__main__':
  tf.test.main()
