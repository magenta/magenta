# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Tests for drums_lib."""

# internal imports
import tensorflow as tf

from magenta.common import testing_lib as common_testing_lib
from magenta.music import polyphony_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2

class PolyphonyLibTest(tf.test.TestCase):

  def setUp(self):
    self.note_sequence = common_testing_lib.parse_test_proto(
        music_pb2.NoteSequence,
        """
        time_signatures: {
          numerator: 4
          denominator: 4
        }
        tempos: {
          qpm: 60
        }
        """)

  def testFromQuantizedNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(12, 100, 0.0, 10.0), (11, 55, 0.25, 0.50), (40, 45, 2.50, 3.50),
         (55, 120, 4.0, 4.25), (60, 100, 4.0, 5.5), (52, 99, 4.75, 5.0)],
        is_drum=True)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    poly_seq = polyphony_lib.PolyphonicSequence(quantized_sequence)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
  tf.test.main()
