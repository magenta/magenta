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
"""Defines Module base class and implementations.

Modules are data processing building blocks for creating datasets.
"""
# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import sequences_lib
from magenta.pipelines import pipeline
from magenta.protobuf import music_pb2


class Quantizer(pipeline.Pipeline):
  """A Module that quantizes NoteSequence data."""

  def __init__(self, steps_per_beat=4):
    super(Quantizer, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=sequences_lib.QuantizedSequence)
    self.steps_per_beat = steps_per_beat

  def transform(self, note_sequence):
    quantized_sequence = sequences_lib.QuantizedSequence()
    try:
      quantized_sequence.from_note_sequence(note_sequence, self.steps_per_beat)
      return [quantized_sequence]
    except sequences_lib.MultipleTimeSignatureException:
      tf.logging.debug('Multiple time signatures found in NoteSequence')
      return []

  def get_stats(self):
    return {}


class MonophonicMelodyExtractor(pipeline.Pipeline):
  """Extracts monophonic melodies from a QuantizedSequence."""

  def __init__(self, min_bars=7, min_unique_pitches=5, gap_bars=1.0,
               ignore_polyphonic_notes=False):
    super(MonophonicMelodyExtractor, self).__init__(
        input_type=sequences_lib.QuantizedSequence,
        output_type=melodies_lib.MonophonicMelody)
    self.min_bars = min_bars
    self.min_unique_pitches = min_unique_pitches
    self.gap_bars = gap_bars
    self.ignore_polyphonic_notes = False

  def transform(self, quantized_sequence):
    return melodies_lib.extract_melodies(
        quantized_sequence,
        min_bars=self.min_bars,
        min_unique_pitches=self.min_unique_pitches,
        gap_bars=self.gap_bars,
        ignore_polyphonic_notes=self.ignore_polyphonic_notes)

  def get_stats(self):
    return {}
