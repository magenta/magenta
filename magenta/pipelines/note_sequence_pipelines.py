# Copyright 2024 The Magenta Authors.
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

"""NoteSequence processing pipelines."""

import copy

from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from note_seq import constants
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf

# Shortcut to chord symbol text annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class NoteSequencePipeline(pipeline.Pipeline):
  """Superclass for pipelines that input and output NoteSequences."""

  def __init__(self, name=None):
    """Construct a NoteSequencePipeline. Should only be called by subclasses.

    Args:
      name: Pipeline name.
    """
    super(NoteSequencePipeline, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=music_pb2.NoteSequence,
        name=name)


class Splitter(NoteSequencePipeline):
  """A Pipeline that splits NoteSequences at regular intervals."""

  def __init__(self, hop_size_seconds, name=None):
    """Creates a Splitter pipeline.

    Args:
      hop_size_seconds: Hop size in seconds that will be used to split a
          NoteSequence at regular intervals.
      name: Pipeline name.
    """
    super(Splitter, self).__init__(name=name)
    self._hop_size_seconds = hop_size_seconds

  def transform(self, input_object):
    note_sequence = input_object
    return sequences_lib.split_note_sequence(
        note_sequence, self._hop_size_seconds)


class TimeChangeSplitter(NoteSequencePipeline):
  """A Pipeline that splits NoteSequences on time signature & tempo changes."""

  def transform(self, input_object):
    note_sequence = input_object
    return sequences_lib.split_note_sequence_on_time_changes(note_sequence)


class Quantizer(NoteSequencePipeline):
  """A Pipeline that quantizes NoteSequence data."""

  def __init__(self, steps_per_quarter=None, steps_per_second=None, name=None):
    """Creates a Quantizer pipeline.

    Exactly one of `steps_per_quarter` and `steps_per_second` should be defined.

    Args:
      steps_per_quarter: Steps per quarter note to use for quantization.
      steps_per_second: Steps per second to use for quantization.
      name: Pipeline name.

    Raises:
      ValueError: If both or neither of `steps_per_quarter` and
          `steps_per_second` are set.
    """
    super(Quantizer, self).__init__(name=name)
    if (steps_per_quarter is not None) == (steps_per_second is not None):
      raise ValueError(
          'Exactly one of steps_per_quarter or steps_per_second must be set.')
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_second = steps_per_second

  def transform(self, input_object):
    note_sequence = input_object
    try:
      if self._steps_per_quarter is not None:
        quantized_sequence = sequences_lib.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
      else:
        quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
            note_sequence, self._steps_per_second)
      return [quantized_sequence]
    except sequences_lib.MultipleTimeSignatureError as e:
      tf.logging.warning('Multiple time signatures in NoteSequence %s: %s',
                         note_sequence.filename, e)
      self._set_stats([statistics.Counter(
          'sequences_discarded_because_multiple_time_signatures', 1)])
      return []
    except sequences_lib.MultipleTempoError as e:
      tf.logging.warning('Multiple tempos found in NoteSequence %s: %s',
                         note_sequence.filename, e)
      self._set_stats([statistics.Counter(
          'sequences_discarded_because_multiple_tempos', 1)])
      return []
    except sequences_lib.BadTimeSignatureError as e:
      tf.logging.warning('Bad time signature in NoteSequence %s: %s',
                         note_sequence.filename, e)
      self._set_stats([statistics.Counter(
          'sequences_discarded_because_bad_time_signature', 1)])
      return []


class SustainPipeline(NoteSequencePipeline):
  """Applies sustain pedal control changes to a NoteSequence."""

  def transform(self, input_object):
    note_sequence = input_object
    return [sequences_lib.apply_sustain_control_changes(note_sequence)]


class StretchPipeline(NoteSequencePipeline):
  """Creates stretched versions of the input NoteSequence."""

  def __init__(self, stretch_factors, name=None):
    """Creates a StretchPipeline.

    Args:
      stretch_factors: A Python list of uniform stretch factors to apply.
      name: Pipeline name.
    """
    super(StretchPipeline, self).__init__(name=name)
    self._stretch_factors = stretch_factors

  def transform(self, input_object):
    note_sequence = input_object
    return [sequences_lib.stretch_note_sequence(note_sequence, stretch_factor)
            for stretch_factor in self._stretch_factors]


class TranspositionPipeline(NoteSequencePipeline):
  """Creates transposed versions of the input NoteSequence."""

  def __init__(self, transposition_range, min_pitch=constants.MIN_MIDI_PITCH,
               max_pitch=constants.MAX_MIDI_PITCH, name=None):
    """Creates a TranspositionPipeline.

    Args:
      transposition_range: Collection of integer pitch steps to transpose.
      min_pitch: Integer pitch value below which notes will be considered
          invalid.
      max_pitch: Integer pitch value above which notes will be considered
          invalid.
      name: Pipeline name.
    """
    super(TranspositionPipeline, self).__init__(name=name)
    self._transposition_range = transposition_range
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch

  def transform(self, input_object):
    sequence = input_object
    stats = dict((state_name, statistics.Counter(state_name)) for state_name in
                 ['skipped_due_to_range_exceeded', 'transpositions_generated'])

    if sequence.key_signatures:
      tf.logging.warn('Key signatures ignored by TranspositionPipeline.')
    if any(note.pitch_name for note in sequence.notes):
      tf.logging.warn('Pitch names ignored by TranspositionPipeline.')
    if any(ta.annotation_type == CHORD_SYMBOL
           for ta in sequence.text_annotations):
      tf.logging.warn('Chord symbols ignored by TranspositionPipeline.')

    transposed = []
    for amount in self._transposition_range:
      # Note that transpose is called even with a transpose amount of zero, to
      # ensure that out-of-range pitches are handled correctly.
      ts = self._transpose(sequence, amount, stats)
      if ts is not None:
        transposed.append(ts)

    stats['transpositions_generated'].increment(len(transposed))
    self._set_stats(stats.values())
    return transposed

  def _transpose(self, ns, amount, stats):
    """Transposes a note sequence by the specified amount."""
    ts = copy.deepcopy(ns)
    for note in ts.notes:
      if not note.is_drum:
        note.pitch += amount
        if note.pitch < self._min_pitch or note.pitch > self._max_pitch:
          stats['skipped_due_to_range_exceeded'].increment()
          return None
    return ts

  class TranspositionToPipeline(NoteSequencePipeline):
    """Transposes the input NoteSequence to a given key. 
    
    The pipeline is useful for standardizing an entire 
    dataset to a single tonal center, i.e. middle C. 
    This pipeline relies on the `key_signatures` record. 
    If multiple key signatures are found, only the first 
    one is used. If none are found, the note sequence 
    is ignored. The key mode (major or minor) is not 
    altered. """

    def __init__(self, to_key=0, name=None):
        """Creates a TranspositionToPipeline.

        Args:
          to_key: The tonic to which to transpose. Must be an integer. 
              0 corresponds to middle C.
          name: Pipeline name.
        Returns:
            A list with the transposed NoteSequence or an empty list.
        """
        super(TranspositionToPipeline, self).__init__(name=name)
        self.to_key = to_key

    def transform(self, sequence):
        stats = dict([(state_name, statistics.Counter(state_name)) for state_name in
                      ['skipped_due_to_missing_key_signature', 
                       'seqs_with_multiple_key_signatures',
                       'transpositions_generated']])
        
        if not sequence.key_signatures:
            tf.logging.warning('No key signature in NoteSequence %s. \
                               Skipping sequence.', sequence.filename)
            self._set_stats([statistics.Counter( 'skipped_due_to_missing_key_signature', 1)])
            return []
        elif len(sequence.key_signatures) > 1:
            tf.logging.warning('Multiple key signatures in NoteSequence %s. \
                               The first one is used; the rest are ignored.', 
                               sequence.filename)
            stats['seqs_with_multiple_key_signatures'].increment()
        if any(note.pitch_name for note in sequence.notes):
            tf.logging.warn('Pitch names ignored by TranspositionToPipeline.')
        if any(ta.annotation_type == CHORD_SYMBOL for ta in sequence.text_annotations):
            tf.logging.warn('Chord symbols ignored by TranspositionToPipeline.')
    
        key = sequence.key_signatures[0].key 
        transposed = self._transpose(sequence, self.to_key-key)
        if transposed is not None:
            stats['transpositions_generated'].increment()
            self._set_stats(stats.values())
            return [transposed]
        else:
            return []
        
    def _transpose(self, ns, amount):
        """Transposes a NoteSequence `ns` by a specified `amount`."""
        ts = copy.deepcopy(ns)
        for note in ts.notes:
            if not note.is_drum:
                note.pitch += amount
        return ts
