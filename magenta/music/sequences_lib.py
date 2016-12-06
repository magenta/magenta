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
"""Defines sequence of notes objects for creating datasets."""

import copy

# internal imports

from magenta.music import constants
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2

# Set the quantization cutoff.
# Note events before this cutoff are rounded down to nearest step. Notes
# above this cutoff are rounded up to nearest step. The cutoff is given as a
# fraction of a step.
# For example, with quantize_cutoff = 0.75 using 0-based indexing,
# if .75 < event <= 1.75, it will be quantized to step 1.
# If 1.75 < event <= 2.75 it will be quantized to step 2.
# A number close to 1.0 gives less wiggle room for notes that start early,
# and they will be snapped to the previous step.
QUANTIZE_CUTOFF = 0.5

# Shortcut to chord symbol text annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class BadTimeSignatureException(Exception):
  pass


class MultipleTimeSignatureException(Exception):
  pass


class MultipleTempoException(Exception):
  pass


class NegativeTimeException(Exception):
  pass


class SequenceNotQuantizedException(Exception):
  """Exception for when a quantized sequence was expected but not received.

  Should not happen during normal operation and likely indicates a programming
  error.
  """
  pass


def extract_subsequence(sequence, start_time, end_time):
  """Extracts a subsequence from a NoteSequence.

  Notes starting before `start_time` are not included. Notes ending after
  `end_time` are truncated. Text annotations (including chord symbols) are also
  only included if between `start_time` and `end_time`.

  Args:
    sequence: The NoteSequence to extract a subsequence from.
    start_time: The float time in seconds to start the subsequence.
    end_time: The float time in seconds to end the subsequence.

  Returns:
    A new NoteSequence that is a subsequence of `sequence` in the specified time
    range.
  """
  subsequence = music_pb2.NoteSequence()
  subsequence.CopyFrom(sequence)

  del subsequence.notes[:]
  for note in sequence.notes:
    if note.start_time < start_time or note.start_time >= end_time:
      continue
    new_note = subsequence.notes.add()
    new_note.CopyFrom(note)
    new_note.end_time = min(note.end_time, end_time)
  subsequence.total_time = min(sequence.total_time, end_time)

  del subsequence.text_annotations[:]
  for annotation in sequence.text_annotations:
    if annotation.time < start_time or annotation.time >= end_time:
      continue
    new_annotation = subsequence.text_annotations.add()
    new_annotation.CopyFrom(annotation)
  return subsequence


def _is_power_of_2(x):
  return x and not x & (x - 1)


def assert_is_quantized_sequence(note_sequence):
  """Confirms that the given NoteSequence proto has been quantized.

  Args:
    note_sequence: A music_pb2.NoteSequence proto.

  Raises:
    SequenceNotQuantizedException: If the sequence is not quantized.
  """
  # If the QuantizationInfo message has a non-zero steps_per_quarter, assume
  # that the proto has been quantized.
  if not note_sequence.quantization_info.steps_per_quarter > 0:
    raise SequenceNotQuantizedException('NoteSequence %s is not quantized.' %
                                        note_sequence.id)


def steps_per_bar_in_quantized_sequence(note_sequence):
  """Calculates steps per bar in a NoteSequence that has been quantized.

  Args:
    note_sequence: The NoteSequence to examine.

  Returns:
    Steps per bar as a floating point number.
  """
  assert_is_quantized_sequence(note_sequence)

  quarters_per_beat = 4.0 / note_sequence.time_signatures[0].denominator
  quarters_per_bar = (quarters_per_beat *
                      note_sequence.time_signatures[0].numerator)
  steps_per_bar_float = (note_sequence.quantization_info.steps_per_quarter *
                         quarters_per_bar)
  return steps_per_bar_float


def quantize_to_step(unquantized_seconds, steps_per_second,
                     quantize_cutoff=QUANTIZE_CUTOFF):
  """Quantizes seconds to the nearest step, given steps_per_second.

  See the comments above `QUANTIZE_CUTOFF` for details on how the quantizing
  algorithm works.

  Args:
    unquantized_seconds: Seconds to quantize.
    steps_per_second: Quantizing resolution.
    quantize_cutoff: Value to use for quantizing cutoff.

  Returns:
    The input value quantized to the nearest step.
  """
  unquantized_steps = unquantized_seconds * steps_per_second
  return int(unquantized_steps + (1 - quantize_cutoff))


def steps_per_quarter_to_steps_per_second(steps_per_quarter, qpm):
  """Calculates steps per second given steps_per_quarter and a qpm."""
  return steps_per_quarter * qpm / 60.0


def quantize_note_sequence(note_sequence, steps_per_quarter):
  """Quantize a NoteSequence proto.

  The input NoteSequence is copied and quantization-related fields are
  populated.

  Note start and end times, and chord times are snapped to a nearby quantized
  step, and the resulting times are stored in a separate field (e.g.,
  quantized_start_step). See the comments above `QUANTIZE_CUTOFF` for details on
  how the quantizing algorithm works.

  Args:
    note_sequence: A music_pb2.NoteSequence protocol buffer.
    steps_per_quarter: Each quarter note of music will be divided into this
        many quantized time steps.

  Returns:
    A copy of the original NoteSequence, with quantized times added.

  Raises:
    MultipleTimeSignatureException: If there is a change in time signature
        in `note_sequence`.
    MultipleTempoException: If there is a change in tempo in `note_sequence`.
    BadTimeSignatureException: If the time signature found in `note_sequence`
        has a denominator which is not a power of 2.
    NegativeTimeException: If a note or chord occurs at a negative time.
  """
  qns = copy.deepcopy(note_sequence)

  qns.quantization_info.steps_per_quarter = steps_per_quarter

  if qns.time_signatures:
    time_signatures = sorted(qns.time_signatures, key=lambda ts: ts.time)
    # There is an implicit 4/4 time signature at 0 time. So if the first time
    # signature is something other than 4/4 and it's at a time other than 0,
    # that's an implicit time signature change.
    if time_signatures[0].time != 0 and not (
        time_signatures[0].numerator == 4 and
        time_signatures[0].denominator == 4):
      raise MultipleTimeSignatureException(
          'NoteSequence has an implicit change from initial 4/4 time '
          'signature to %d/%d at %.2f seconds.' % (
              time_signatures[0].numerator, time_signatures[0].denominator,
              time_signatures[0].time))

    for time_signature in time_signatures[1:]:
      if (time_signature.numerator != qns.time_signatures[0].numerator or
          time_signature.denominator != qns.time_signatures[0].denominator):
        raise MultipleTimeSignatureException(
            'NoteSequence has at least one time signature change from %d/%d to '
            '%d/%d at %.2f seconds.' % (
                time_signatures[0].numerator, time_signatures[0].denominator,
                time_signature.numerator, time_signature.denominator,
                time_signature.time))

    # Make it clear that there is only 1 time signature and it starts at the
    # beginning.
    qns.time_signatures[0].time = 0
    del qns.time_signatures[1:]
  else:
    time_signature = qns.time_signatures.add()
    time_signature.numerator = 4
    time_signature.denominator = 4
    time_signature.time = 0

  if not _is_power_of_2(qns.time_signatures[0].denominator):
    raise BadTimeSignatureException(
        'Denominator is not a power of 2. Time signature: %d/%d' %
        (qns.time_signatures[0].numerator, qns.time_signatures[0].denominator))

  if qns.tempos:
    tempos = sorted(qns.tempos, key=lambda t: t.time)
    # There is an implicit 120.0 qpm tempo at 0 time. So if the first tempo is
    # something other that 120.0 and it's at a time other than 0, that's an
    # implicit tempo change.
    if tempos[0].time != 0 and (
        tempos[0].qpm != constants.DEFAULT_QUARTERS_PER_MINUTE):
      raise MultipleTempoException(
          'NoteSequence has an implicit tempo change from initial %.1f qpm to '
          '%.1f qpm at %.2f seconds.' % (
              constants.DEFAULT_QUARTERS_PER_MINUTE, tempos[0].qpm,
              tempos[0].time))

    for tempo in tempos[1:]:
      if tempo.qpm != qns.tempos[0].qpm:
        raise MultipleTempoException(
            'NoteSequence has at least one tempo change from %f qpm to %f qpm '
            'at %.2f seconds.' % (tempos[0].qpm, tempo.qpm, tempo.time))

    # Make it clear that there is only 1 tempo and it starts at the beginning.
    qns.tempos[0].time = 0
    del qns.tempos[1:]
  else:
    tempo = qns.tempos.add()
    tempo.qpm = constants.DEFAULT_QUARTERS_PER_MINUTE
    tempo.time = 0

  # Compute quantization steps per second.
  steps_per_second = steps_per_quarter_to_steps_per_second(
      steps_per_quarter, qns.tempos[0].qpm)

  qns.total_quantized_steps = quantize_to_step(qns.total_time, steps_per_second)

  for note in qns.notes:
    # Quantize the start and end times of the note.
    note.quantized_start_step = quantize_to_step(
        note.start_time, steps_per_second)
    note.quantized_end_step = quantize_to_step(
        note.end_time, steps_per_second)
    if note.quantized_end_step == note.quantized_start_step:
      note.quantized_end_step += 1

    # Do not allow notes to start or end in negative time.
    if note.quantized_start_step < 0 or note.quantized_end_step < 0:
      raise NegativeTimeException(
          'Got negative note time: start_step = %s, end_step = %s' %
          (note.quantized_start_step, note.quantized_end_step))

    # Extend quantized sequence if necessary.
    if note.quantized_end_step > qns.total_quantized_steps:
      qns.total_quantized_steps = note.quantized_end_step

  # Also quantize chord symbol annotations.
  for annotation in qns.text_annotations:
    # Quantize the chord time, disallowing negative time.
    annotation.quantized_step = quantize_to_step(
        annotation.time, steps_per_second)
    if annotation.quantized_step < 0:
      raise NegativeTimeException(
          'Got negative chord time: step = %s' % annotation.quantized_step)

  return qns


class TranspositionPipeline(pipeline.Pipeline):
  """Creates transposed versions of the input NoteSequence."""

  def __init__(self, transposition_range, name=None):
    """Creates a TranspositionPipeline.

    Args:
      transposition_range: Collection of integer pitch steps to transpose.
      name: Pipeline name.
    """
    super(TranspositionPipeline, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=music_pb2.NoteSequence,
        name=name)
    self._transposition_range = transposition_range

  def transform(self, sequence):
    stats = dict([(state_name, statistics.Counter(state_name)) for state_name in
                  ['skipped_due_to_range_exceeded',
                   'transpositions_generated']])

    transposed = []
    # Transpose up to a major third in either direction.
    for amount in self._transposition_range:
      if amount == 0:
        transposed.append(sequence)
      else:
        ts = self._transpose(sequence, amount, stats)
        if ts is not None:
          transposed.append(ts)

    stats['transpositions_generated'].increment(len(transposed))
    self._set_stats(stats.values())
    return transposed

  @staticmethod
  def _transpose(ns, amount, stats):
    """Transposes a note sequence by the specified amount."""
    ts = copy.deepcopy(ns)
    for note in ts.notes:
      note.pitch += amount
      if (note.pitch < constants.MIN_MIDI_PITCH or
          note.pitch > constants.MAX_MIDI_PITCH):
        stats['skipped_due_to_range_exceeded'].increment()
        return None
    return ts
