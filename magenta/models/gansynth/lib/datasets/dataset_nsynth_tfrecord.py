# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Dataset loading code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from magenta.models.gansynth.lib.datasets import dataset_nsynth_tfrecord_impl

PITCH_COUNTS = {
    24: 711,
    25: 720,
    26: 715,
    27: 725,
    28: 726,
    29: 723,
    30: 738,
    31: 829,
    32: 839,
    33: 840,
    34: 860,
    35: 870,
    36: 999,
    37: 1007,
    38: 1063,
    39: 1070,
    40: 1084,
    41: 1121,
    42: 1134,
    43: 1129,
    44: 1155,
    45: 1149,
    46: 1169,
    47: 1154,
    48: 1432,
    49: 1406,
    50: 1454,
    51: 1432,
    52: 1593,
    53: 1613,
    54: 1578,
    55: 1784,
    56: 1738,
    57: 1756,
    58: 1718,
    59: 1738,
    60: 1789,
    61: 1746,
    62: 1765,
    63: 1748,
    64: 1764,
    65: 1744,
    66: 1677,
    67: 1746,
    68: 1682,
    69: 1705,
    70: 1694,
    71: 1667,
    72: 1695,
    73: 1580,
    74: 1608,
    75: 1546,
    76: 1576,
    77: 1485,
    78: 1408,
    79: 1438,
    80: 1333,
    81: 1369,
    82: 1331,
    83: 1295,
    84: 1291
}


def provide_one_hot_labels(batch_size):
  """Provides one hot labels."""
  return dataset_nsynth_tfrecord_impl.provide_one_hot_labels(
      PITCH_COUNTS, batch_size)


AUDIO_FILE_PATTERN = (
    '/tmp/path-to/nsynth-train.tfrecord')


def provide_audio_dataset(length=64000, channels=1):
  """Provides dataset."""
  return dataset_nsynth_tfrecord_impl.provide_audio_dataset(
      file_pattern=AUDIO_FILE_PATTERN,
      pitches=sorted(PITCH_COUNTS.keys()),
      length=length,
      channels=channels)
