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

import collections
import copy

# internal imports
import intervaltree
import tensorflow as tf

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


class QuantizationStatusException(Exception):
  """Exception for when a sequence was unexpectedly quantized or unquantized.

  Should not happen during normal operation and likely indicates a programming
  error.
  """
  pass


def trim_note_sequence(sequence, start_time, end_time):
  """Trim notes from a NoteSequence to lie within a specified time range.

  Notes starting before `start_time` are not included. Notes ending after
  `end_time` are truncated.

  Args:
    sequence: The NoteSequence for which to trim notes.
    start_time: The float time in seconds after which all notes should begin.
    end_time: The float time in seconds before which all notes should end.

  Returns:
    A copy of `sequence` with all notes trimmed to lie between `start_time` and
    `end_time`.

  Raises:
    QuantizationStatusException: If the sequence has already been quantized.
  """
  if is_quantized_sequence(sequence):
    raise QuantizationStatusException(
        'Can only trim notes and chords for unquantized NoteSequence.')

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

  return subsequence


def extract_subsequence(sequence, start_time, end_time):
  """Extracts a subsequence from a NoteSequence.

  Notes starting before `start_time` are not included. Notes ending after
  `end_time` are truncated. Time signature, tempo, key signature, and chord
  changes outside the specified time range are removed; however, the most recent
  event of each of these types prior to `start_time` is included at
  `start_time`. This means that e.g. if a time signature of 3/4 is specified in
  the original sequence prior to `start_time` (and is not followed by a
  different time signature), the extracted subsequence will include a 3/4 time
  signature event at `start_time`. Pitch bends and control changes are removed
  entirely.

  The extracted subsequence is shifted to start at time zero.

  Args:
    sequence: The NoteSequence to extract a subsequence from.
    start_time: The float time in seconds to start the subsequence.
    end_time: The float time in seconds to end the subsequence.

  Returns:
    A new NoteSequence containing the subsequence of `sequence` from the
    specified time range.

  Raises:
    QuantizationStatusException: If the sequence has already been quantized.
  """
  if is_quantized_sequence(sequence):
    raise QuantizationStatusException(
        'Can only extract subsequence from unquantized NoteSequence.')

  subsequence = music_pb2.NoteSequence()
  subsequence.CopyFrom(sequence)

  # Extract notes.
  del subsequence.notes[:]
  for note in sequence.notes:
    if note.start_time < start_time or note.start_time >= end_time:
      continue
    new_note = subsequence.notes.add()
    new_note.CopyFrom(note)
    new_note.start_time -= start_time
    new_note.end_time = min(note.end_time, end_time) - start_time

  # Extract time signatures.
  del subsequence.time_signatures[:]
  initial_time_signature = None
  for time_signature in sorted(sequence.time_signatures,
                               key=lambda time_signature: time_signature.time):
    if time_signature.time <= start_time:
      initial_time_signature = music_pb2.NoteSequence.TimeSignature()
      initial_time_signature.CopyFrom(time_signature)
      continue
    elif time_signature.time >= end_time:
      break
    new_time_signature = subsequence.time_signatures.add()
    new_time_signature.CopyFrom(time_signature)
    new_time_signature.time -= start_time
  if initial_time_signature:
    initial_time_signature.time = 0.0
    subsequence.time_signatures.extend([initial_time_signature])
  subsequence.time_signatures.sort(key=lambda ts: ts.time)

  # Extract key signatures.
  del subsequence.key_signatures[:]
  initial_key_signature = None
  for key_signature in sorted(sequence.key_signatures,
                              key=lambda key_signature: key_signature.time):
    if key_signature.time <= start_time:
      initial_key_signature = music_pb2.NoteSequence.KeySignature()
      initial_key_signature.CopyFrom(key_signature)
      continue
    elif key_signature.time >= end_time:
      break
    new_key_signature = subsequence.key_signatures.add()
    new_key_signature.CopyFrom(key_signature)
    new_key_signature.time -= start_time
  if initial_key_signature:
    initial_key_signature.time = 0.0
    subsequence.key_signatures.extend([initial_key_signature])
  subsequence.key_signatures.sort(key=lambda ks: ks.time)

  # Extract tempos.
  del subsequence.tempos[:]
  initial_tempo = None
  for tempo in sorted(sequence.tempos, key=lambda tempo: tempo.time):
    if tempo.time <= start_time:
      initial_tempo = music_pb2.NoteSequence.Tempo()
      initial_tempo.CopyFrom(tempo)
      continue
    elif tempo.time >= end_time:
      break
    new_tempo = subsequence.tempos.add()
    new_tempo.CopyFrom(tempo)
    new_tempo.time -= start_time
  if initial_tempo:
    initial_tempo.time = 0.0
    subsequence.tempos.extend([initial_tempo])
  subsequence.tempos.sort(key=lambda tempo: tempo.time)

  # Extract chord symbols (other text annotations are deleted).
  del subsequence.text_annotations[:]
  chord_symbols = [annotation for annotation in sequence.text_annotations
                   if annotation.annotation_type == CHORD_SYMBOL]
  initial_chord_symbol = None
  for chord_symbol in sorted(chord_symbols,
                             key=lambda chord_symbol: chord_symbol.time):
    if chord_symbol.time <= start_time:
      initial_chord_symbol = music_pb2.NoteSequence.TextAnnotation()
      initial_chord_symbol.CopyFrom(chord_symbol)
      continue
    elif chord_symbol.time >= end_time:
      break
    new_chord_symbol = subsequence.text_annotations.add()
    new_chord_symbol.CopyFrom(chord_symbol)
    new_chord_symbol.time -= start_time
  if initial_chord_symbol:
    initial_chord_symbol.time = 0.0
    subsequence.text_annotations.extend([initial_chord_symbol])
  subsequence.text_annotations.sort(key=lambda ta: ta.time)

  # Remove pitch bend and control change events.
  del subsequence.pitch_bends[:]
  del subsequence.control_changes[:]

  actual_end_time = min(sequence.total_time, end_time)
  subsequence.total_time = actual_end_time - start_time

  subsequence.subsequence_info.start_time_offset = start_time
  subsequence.subsequence_info.end_time_offset = (
      sequence.total_time - actual_end_time)

  return subsequence


def _is_power_of_2(x):
  return x and not x & (x - 1)


def is_quantized_sequence(note_sequence):
  """Returns whether or not a NoteSequence proto has been quantized.

  Args:
    note_sequence: A music_pb2.NoteSequence proto.

  Returns:
    True if `note_sequence` is quantized, otherwise False.
  """
  # If the QuantizationInfo message has a non-zero steps_per_quarter, assume
  # that the proto has been quantized.
  return note_sequence.quantization_info.steps_per_quarter > 0


def assert_is_quantized_sequence(note_sequence):
  """Confirms that the given NoteSequence proto has been quantized.

  Args:
    note_sequence: A music_pb2.NoteSequence proto.

  Raises:
    QuantizationStatusException: If the sequence is not quantized.
  """
  if not is_quantized_sequence(note_sequence):
    raise QuantizationStatusException('NoteSequence %s is not quantized.' %
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


def split_note_sequence_on_time_changes(note_sequence, split_notes=True):
  """Split one NoteSequence into many around time signature and tempo changes.

  This function splits a NoteSequence into multiple NoteSequences, each of which
  contains only a single time signature and tempo, unless `split_notes` is False
  in which case all time signature and tempo changes occur within sustained
  notes. Each of the resulting NoteSequences is shifted to start at time zero.

  Args:
    note_sequence: The NoteSequence to split.
    split_notes: If True, the NoteSequence will be split at all time changes,
        regardless of whether or not any notes are sustained across the time
        change. If False, the NoteSequence will not be split at time changes
        that occur within sustained notes.

  Returns:
    A Python list of NoteSequences.
  """
  prev_change_time = 0.0

  current_numerator = 4
  current_denominator = 4
  current_qpm = constants.DEFAULT_QUARTERS_PER_MINUTE

  time_signatures_and_tempos = sorted(
      list(note_sequence.time_signatures) + list(note_sequence.tempos),
      key=lambda t: t.time)

  notes_by_start_time = sorted(list(note_sequence.notes),
                               key=lambda note: note.start_time)
  note_idx = 0
  active_notes = []

  subsequences = []

  for time_change in time_signatures_and_tempos:
    if isinstance(time_change, music_pb2.NoteSequence.TimeSignature):
      if (time_change.numerator == current_numerator and
          time_change.denominator == current_denominator):
        # Time signature didn't actually change.
        continue
    else:
      if time_change.qpm == current_qpm:
        # Tempo didn't actually change.
        continue

    # Update active notes.
    while (note_idx < len(notes_by_start_time) and
           notes_by_start_time[note_idx].start_time < time_change.time):
      active_notes.append(notes_by_start_time[note_idx])
      note_idx += 1
    active_notes = [note for note in active_notes
                    if note.end_time > time_change.time]

    if time_change.time > prev_change_time:
      if split_notes or not active_notes:
        # Extract the subsequence between the previous time change and this
        # time change.
        subsequence = extract_subsequence(note_sequence, prev_change_time,
                                          time_change.time)
        subsequences.append(subsequence)
        prev_change_time = time_change.time

    # Even if we didn't split here, update the current time signature or tempo.
    if isinstance(time_change, music_pb2.NoteSequence.TimeSignature):
      current_numerator = time_change.numerator
      current_denominator = time_change.denominator
    else:
      current_qpm = time_change.qpm

  # Handle the final subsequence.
  if note_sequence.total_time > prev_change_time:
    subsequence = extract_subsequence(note_sequence, prev_change_time,
                                      note_sequence.total_time)
    subsequences.append(subsequence)

  return subsequences


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
            'NoteSequence has at least one tempo change from %.1f qpm to %.1f '
            'qpm at %.2f seconds.' % (tempos[0].qpm, tempo.qpm, tempo.time))

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


def apply_sustain_control_changes(note_sequence, sustain_control_number=64):
  """Returns a new NoteSequence with sustain pedal control changes applied.

  Extends each note within a sustain to the end of the sustain period. This is
  done on a per instrument/program basis, so notes are only affected by sustain
  events for the same instrument/program.

  Args:
    note_sequence: The NoteSequence for which to apply sustain. This object will
        not be modified.
    sustain_control_number: The MIDI control number for sustain pedal. Control
        events with this number and value 0-63 will be treated as sustain pedal
        OFF events, and control events with this number and value 64-127 will be
        treated as sustain pedal ON events.

  Returns:
    A copy of `note_sequence` but with note end times extended to account for
    sustain.

  Raises:
    QuantizationStatusException: If `note_sequence` is quantized. Sustain can
        only be applied to unquantized note sequences.
  """
  if is_quantized_sequence(note_sequence):
    raise QuantizationStatusException(
        'Can only apply sustain to unquantized NoteSequence.')

  # Find all sustain events, sort, and organize by instrument/program.
  sustain_events = sorted((cc.time, cc.control_value, cc.instrument, cc.program)
                          for cc in note_sequence.control_changes
                          if cc.control_number == sustain_control_number)
  sustain_events_by_instrument_and_program = collections.defaultdict(list)
  for time, value, instrument, program in sustain_events:
    sustain_events_by_instrument_and_program[(instrument, program)].append(
        (time, value))

  # Build an interval tree for each instrument/program.
  sustain_intervals = collections.defaultdict(intervaltree.IntervalTree)
  for key, events in sustain_events_by_instrument_and_program.iteritems():
    start_time = None
    for time, value in events:
      if value < 0 or value > 127:
        tf.logging.warn(
            'Sustain control change has out of range value: %d', value)
      if start_time is None and value >= 64:
        start_time = time
      elif start_time is not None and value < 64:
        sustain_intervals[key].addi(start_time, time)
        start_time = None
    if start_time is not None and start_time < note_sequence.total_time:
      sustain_intervals[key].addi(start_time, note_sequence.total_time)

  sequence = copy.deepcopy(note_sequence)

  # Extend the end time of each note inside a sustain. This extends each end
  # time to the end of the sustain period, even if a subsequent note with the
  # same pitch has an onset within the sustain.
  for note in sequence.notes:
    sustain = sustain_intervals[(note.instrument, note.program)][note.end_time]
    assert len(sustain) <= 1
    if sustain:
      end_time = sustain.pop().end
      assert note.end_time <= end_time
      note.end_time = end_time
      if end_time > note_sequence.total_time:
        note_sequence.total_time = end_time

  return sequence


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
