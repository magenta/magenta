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
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


def add_track_to_sequence(note_sequence, instrument, notes):
  for pitch, velocity, start_time, end_time in notes:
    note = note_sequence.notes.add()
    note.pitch = pitch
    note.velocity = velocity
    note.start_time = start_time
    note.end_time = end_time
    note.instrument = instrument


def add_chords_to_sequence(note_sequence, chords):
  for figure, time in chords:
    annotation = note_sequence.text_annotations.add()
    annotation.time = time
    annotation.text = figure
    annotation.annotation_type = CHORD_SYMBOL


def add_quantized_track_to_sequence(quantized_sequence, instrument, notes):
  if instrument not in quantized_sequence.tracks:
    quantized_sequence.tracks[instrument] = []
  track = quantized_sequence.tracks[instrument]
  for pitch, velocity, start_step, end_step in notes:
    note = sequences_lib.QuantizedSequence.Note(pitch=pitch,
                                                velocity=velocity,
                                                start=start_step,
                                                end=end_step,
                                                instrument=instrument,
                                                program=0)
    track.append(note)


def add_quantized_chords_to_sequence(quantized_sequence, chords):
  for figure, step in chords:
    chord = sequences_lib.QuantizedSequence.ChordSymbol(step=step,
                                                        figure=figure)
    quantized_sequence.chords.append(chord)
