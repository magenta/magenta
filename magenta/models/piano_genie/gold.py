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

# Lint as: python2, python3
"""Gold standard musical sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint:disable=g-line-too-long,line-too-long
_MARYLMB = (
    "64,62,60,62,64,64,64,62,62,62,64,67,67,64,62,60,62,64,64,64,64,62,62,64,62,60",
    "32123332223553212333322321")
_TWINKLE = (
    "60,60,67,67,69,69,67,65,65,64,64,62,62,60,67,67,65,65,64,64,62,67,67,65,65,64,64,62,60,60,67,67,69,69,67,65,65,64,64,62,62,60",
    "115566544332215544332554433211556654433221")
_MARIOTM = (
    "64,64,64,60,64,67,55,60,55,52,57,59,58,57,55,64,67,69,65,67,64,60,62,59",
    "666567464256543678564342")
_HAPPYBD = (
    "60,60,62,60,65,64,60,60,62,60,67,65,60,60,72,69,65,64,62,70,70,69,65,67,65",
    "1121431121541186432776454")
_JINGLEB = (
    "64,64,64,64,64,64,64,67,60,62,64,65,65,65,65,65,64,64,64,64,62,62,64,62,67,64,64,64,64,64,64,64,67,60,62,64,65,65,65,65,65,64,64,64,67,67,65,62,60",
    "3333333512344444333322325333333351234444433355421")
_TETRIST = (
    "64,59,60,62,64,62,60,59,57,57,60,64,62,60,59,59,60,62,64,60,57,57,62,65,69,67,65,64,60,64,62,60,59,59,60,62,64,60,57,57",
    "5345654322465433456322468765354322345311")
_FRERJCQ = (
    "60,62,64,60,60,62,64,60,64,65,67,64,65,67,67,69,67,65,64,60,67,69,67,65,64,60,60,55,60,60,55,60",
    "34533453456456676542676542212212")
_GODSVQU = (
    "65,65,67,64,65,67,69,69,70,69,67,65,67,65,64,65,72,72,72,72,70,69,70,70,70,70,69,67,69,70,69,67,65,69,70,72,74,70,69,67,65",
    "33423455654343237777656666545654356786432")
# pylint:enable=g-line-too-long,line-too-long

_GOLD = [
    _MARYLMB, _TWINKLE, _MARIOTM, _HAPPYBD, _JINGLEB, _TETRIST, _FRERJCQ,
    _GODSVQU
]


def gold_longest():
  """Returns the length of the longest gold standard sequence."""
  return max([len(x[0].split(",")) for x in _GOLD])


def gold_iterator(transpose_range=(0, 1)):
  """Iterates through pairs of MIDI notes and buttons."""
  maxlen = gold_longest()
  for transpose in range(*transpose_range):
    for midi_notes, buttons in _GOLD:
      midi_notes = [int(x) + transpose for x in midi_notes.split(",")]
      buttons = [int(x) for x in list(buttons)]
      seqlen = len(midi_notes)
      assert len(buttons) == len(midi_notes)
      assert seqlen <= maxlen
      midi_notes += [21] * (maxlen - seqlen)
      buttons += [0] * (maxlen - seqlen)
      yield [midi_notes], [buttons], seqlen
