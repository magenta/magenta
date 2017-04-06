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
"""Testing support code."""

# internal imports
from magenta.music import encoder_decoder
from magenta.protobuf import music_pb2

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


def add_track_to_sequence(note_sequence, instrument, notes, is_drum=False):
  for pitch, velocity, start_time, end_time in notes:
    note = note_sequence.notes.add()
    note.pitch = pitch
    note.velocity = velocity
    note.start_time = start_time
    note.end_time = end_time
    note.instrument = instrument
    note.is_drum = is_drum
    if end_time > note_sequence.total_time:
      note_sequence.total_time = end_time


def add_chords_to_sequence(note_sequence, chords):
  for figure, time in chords:
    annotation = note_sequence.text_annotations.add()
    annotation.time = time
    annotation.text = figure
    annotation.annotation_type = CHORD_SYMBOL


def add_control_changes_to_sequence(note_sequence, instrument, control_changes):
  for time, control_number, control_value in control_changes:
    control_change = note_sequence.control_changes.add()
    control_change.time = time
    control_change.control_number = control_number
    control_change.control_value = control_value
    control_change.instrument = instrument


def add_quantized_steps_to_sequence(sequence, quantized_steps):
  assert len(sequence.notes) == len(quantized_steps)

  for note, quantized_step in zip(sequence.notes, quantized_steps):
    note.quantized_start_step = quantized_step[0]
    note.quantized_end_step = quantized_step[1]

    if quantized_step[1] > sequence.total_quantized_steps:
      sequence.total_quantized_steps = quantized_step[1]


def add_quantized_chord_steps_to_sequence(sequence, quantized_steps):
  chord_annotations = [a for a in sequence.text_annotations
                       if a.annotation_type == CHORD_SYMBOL]
  assert len(chord_annotations) == len(quantized_steps)
  for chord, quantized_step in zip(chord_annotations, quantized_steps):
    chord.quantized_step = quantized_step


class TrivialOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding that uses the identity encoding."""

  def __init__(self, num_classes):
    self._num_classes = num_classes

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def default_event(self):
    return 0

  def encode_event(self, event):
    return event

  def decode_event(self, event):
    return event
