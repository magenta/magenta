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
"""MusicVAE chord progression utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import magenta.music as mm
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2


def event_list_chords(quantized_sequence, event_lists):
  """Extract corresponding chords for multiple EventSequences.

  Args:
    quantized_sequence: The underlying quantized NoteSequence from which to
        extract the chords. It is assumed that the step numbering in this
        sequence matches the step numbering in each EventSequence in
        `event_lists`.
    event_lists: A list of EventSequence objects.

  Returns:
    A nested list of chord the same length as `event_lists`, where each list is
    the same length as the corresponding EventSequence (in events, not steps).
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

  chords = mm.ChordProgression()
  if quantized_sequence.total_quantized_steps > 0:
    chords.from_quantized_sequence(
        quantized_sequence, 0, quantized_sequence.total_quantized_steps)

  chord_lists = []
  for e in event_lists:
    chord_lists.append([chords[step] if step < len(chords) else mm.NO_CHORD
                        for step in e.steps])

  return chord_lists


def add_chords_to_sequence(note_sequence, chords, chord_times):
  """Add chords to a NoteSequence at specified times.

  Args:
    note_sequence: The NoteSequence proto to which chords will be added (in
        place).
    chords: A Python list of chord figure strings to add to `note_sequence` as
        text annotations.
    chord_times: A Python list containing the time in seconds at which to add
        each chord. Should be the same length as `chords` and nondecreasing.
  """
  current_chord = mm.NO_CHORD
  for chord, time in zip(chords, chord_times):
    if chord != current_chord:
      current_chord = chord
      ta = note_sequence.text_annotations.add()
      ta.annotation_type = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
      ta.time = time
      ta.text = chord
