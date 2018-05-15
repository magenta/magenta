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
import itertools
from operator import itemgetter

# internal imports
import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

from magenta.music import chord_symbols_lib
from magenta.music import constants
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

# Shortcut to text annotation types.
BEAT = music_pb2.NoteSequence.TextAnnotation.BEAT
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


def _extract_subsequences(sequence, split_times, sustain_control_number=64):
  """Extracts multiple subsequences from a NoteSequence.

  Args:
    sequence: The NoteSequence to extract subsequences from.
    split_times: A Python list of subsequence boundary times. The first
        subsequence will start at `split_times[0]` and end at `split_times[1]`,
        the next subsequence will start at `split_times[1]` and end at
        `split_times[2]`, and so on with the last subsequence ending at
        `split_times[-1]`.
    sustain_control_number: The MIDI control number for sustain pedal.

  Returns:
    A Python list of new NoteSequence containing the subsequences of `sequence`.

  Raises:
    QuantizationStatusException: If the sequence has already been quantized.
    ValueError: If there are fewer than 2 split times, or the split times are
        unsorted, or if any of the subsequences would start past the end of the
        sequence.
  """
  if is_quantized_sequence(sequence):
    raise QuantizationStatusException(
        'Can only extract subsequences from unquantized NoteSequence.')

  if len(split_times) < 2:
    raise ValueError('Must provide at least a start and end time.')
  if any(t1 > t2 for t1, t2 in zip(split_times[:-1], split_times[1:])):
    raise ValueError('Split times must be sorted.')
  if any(time >= sequence.total_time for time in split_times[:-1]):
    raise ValueError('Cannot extract subsequence past end of sequence.')

  subsequence = music_pb2.NoteSequence()
  subsequence.CopyFrom(sequence)

  subsequence.total_time = 0.0

  del subsequence.notes[:]
  del subsequence.time_signatures[:]
  del subsequence.key_signatures[:]
  del subsequence.tempos[:]
  del subsequence.text_annotations[:]
  del subsequence.control_changes[:]
  del subsequence.pitch_bends[:]

  subsequences = [copy.deepcopy(subsequence)
                  for _ in range(len(split_times) - 1)]

  # Extract notes into subsequences.
  subsequence_index = -1
  for note in sorted(sequence.notes, key=lambda note: note.start_time):
    if note.start_time < split_times[0]:
      continue
    while (subsequence_index < len(split_times) - 1 and
           note.start_time >= split_times[subsequence_index + 1]):
      subsequence_index += 1
    if subsequence_index == len(split_times) - 1:
      break
    subsequences[subsequence_index].notes.extend([note])
    subsequences[subsequence_index].notes[-1].start_time -= (
        split_times[subsequence_index])
    subsequences[subsequence_index].notes[-1].end_time = min(
        note.end_time,
        split_times[subsequence_index + 1]) - split_times[subsequence_index]
    if (subsequences[subsequence_index].notes[-1].end_time >
        subsequences[subsequence_index].total_time):
      subsequences[subsequence_index].total_time = (
          subsequences[subsequence_index].notes[-1].end_time)

  # Extract time signatures, key signatures, tempos, and chord changes (beats
  # are handled below, other text annotations and pitch bends are deleted).
  # Additional state events will be added to the beginning of each subsequence.

  events_by_type = [
      sequence.time_signatures, sequence.key_signatures, sequence.tempos,
      [annotation for annotation in sequence.text_annotations
       if annotation.annotation_type == CHORD_SYMBOL]]
  new_event_containers = [
      [s.time_signatures for s in subsequences],
      [s.key_signatures for s in subsequences],
      [s.tempos for s in subsequences],
      [s.text_annotations for s in subsequences]
  ]

  for events, containers in zip(events_by_type, new_event_containers):
    previous_event = None
    subsequence_index = -1
    for event in sorted(events, key=lambda event: event.time):
      if event.time <= split_times[0]:
        previous_event = event
        continue
      while (subsequence_index < len(split_times) - 1 and
             event.time > split_times[subsequence_index + 1]):
        subsequence_index += 1
        if subsequence_index == len(split_times) - 1:
          break
        if previous_event is not None:
          # Add state event to the beginning of the subsequence.
          containers[subsequence_index].extend([previous_event])
          containers[subsequence_index][-1].time = 0.0
      if subsequence_index == len(split_times) - 1:
        break
      # Only add the event if it's actually inside the subsequence (and not on
      # the boundary with the next one).
      if event.time < split_times[subsequence_index + 1]:
        containers[subsequence_index].extend([event])
        containers[subsequence_index][-1].time -= split_times[subsequence_index]
      previous_event = event
    # Add final state event to the beginning of all remaining subsequences.
    while subsequence_index < len(split_times) - 2:
      subsequence_index += 1
      if previous_event is not None:
        containers[subsequence_index].extend([previous_event])
        containers[subsequence_index][-1].time = 0.0

  # Copy stateless events to subsequences. Unlike the stateful events above,
  # stateless events do not have an effect outside of the subsequence in which
  # they occur.
  stateless_events_by_type = [
      [annotation for annotation in sequence.text_annotations
       if annotation.annotation_type in (BEAT,)]]
  new_stateless_event_containers = [
      [s.text_annotations for s in subsequences]
  ]
  for events, containers in zip(stateless_events_by_type,
                                new_stateless_event_containers):
    subsequence_index = -1
    for event in sorted(events, key=lambda event: event.time):
      if event.time < split_times[0]:
        continue
      while (subsequence_index < len(split_times) - 1 and
             event.time >= split_times[subsequence_index + 1]):
        subsequence_index += 1
      if subsequence_index == len(split_times) - 1:
        break
      containers[subsequence_index].extend([event])
      containers[subsequence_index][-1].time -= split_times[subsequence_index]

  # Extract sustain pedal events (other control changes are deleted). Sustain
  # pedal state is maintained per-instrument and added to the beginning of each
  # subsequence.
  sustain_events = [cc for cc in sequence.control_changes
                    if cc.control_number == sustain_control_number]
  previous_sustain_events = {}
  subsequence_index = -1
  for sustain_event in sorted(sustain_events, key=lambda event: event.time):
    if sustain_event.time <= split_times[0]:
      previous_sustain_events[sustain_event.instrument] = sustain_event
      continue
    while (subsequence_index < len(split_times) - 1 and
           sustain_event.time > split_times[subsequence_index + 1]):
      subsequence_index += 1
      if subsequence_index == len(split_times) - 1:
        break
      # Add the current sustain pedal state to the beginning of the subsequence.
      for previous_sustain_event in previous_sustain_events.values():
        subsequences[subsequence_index].control_changes.extend(
            [previous_sustain_event])
        subsequences[subsequence_index].control_changes[-1].time = 0.0
    if subsequence_index == len(split_times) - 1:
      break
    # Only add the sustain event if it's actually inside the subsequence (and
    # not on the boundary with the next one).
    if sustain_event.time < split_times[subsequence_index + 1]:
      subsequences[subsequence_index].control_changes.extend([sustain_event])
      subsequences[subsequence_index].control_changes[-1].time -= (
          split_times[subsequence_index])
    previous_sustain_events[sustain_event.instrument] = sustain_event
  # Add final sustain pedal state to the beginning of all remaining
  # subsequences.
  while subsequence_index < len(split_times) - 2:
    subsequence_index += 1
    for _, previous_sustain_event in previous_sustain_events.items():
      subsequences[subsequence_index].control_changes.extend(
          [previous_sustain_event])
      subsequences[subsequence_index].control_changes[-1].time = 0.0

  # Set subsequence info for all subsequences.
  for subsequence, start_time in zip(subsequences, split_times[:-1]):
    subsequence.subsequence_info.start_time_offset = start_time
    subsequence.subsequence_info.end_time_offset = (
        sequence.total_time - start_time - subsequence.total_time)

  return subsequences


def extract_subsequence(sequence, start_time, end_time,
                        sustain_control_number=64):
  """Extracts a subsequence from a NoteSequence.

  Notes starting before `start_time` are not included. Notes ending after
  `end_time` are truncated. Time signature, tempo, key signature, chord changes,
  and sustain pedal events outside the specified time range are removed;
  however, the most recent event of each of these types prior to `start_time` is
  included at `start_time`. This means that e.g. if a time signature of 3/4 is
  specified in the original sequence prior to `start_time` (and is not followed
  by a different time signature), the extracted subsequence will include a 3/4
  time signature event at `start_time`. Pitch bends and control changes other
  than sustain are removed entirely.

  The extracted subsequence is shifted to start at time zero.

  Args:
    sequence: The NoteSequence to extract a subsequence from.
    start_time: The float time in seconds to start the subsequence.
    end_time: The float time in seconds to end the subsequence.
    sustain_control_number: The MIDI control number for sustain pedal.

  Returns:
    A new NoteSequence containing the subsequence of `sequence` from the
    specified time range.

  Raises:
    QuantizationStatusException: If the sequence has already been quantized.
    ValueError: If `start_time` is past the end of `sequence`.
  """
  return _extract_subsequences(
      sequence,
      split_times=[start_time, end_time],
      sustain_control_number=sustain_control_number)[0]


def shift_sequence_times(sequence, shift_seconds):
  """Shifts times in a notesequence.

  Only forward shifts are supported.

  Args:
    sequence: The NoteSequence to shift.
    shift_seconds: The amount to shift.

  Returns:
    A new NoteSequence with shifted times.

  Raises:
    ValueError: If the shift amount is invalid.
    QuantizationStatusException: If the sequence has already been quantized.
  """
  if shift_seconds <= 0:
    raise ValueError('Invalid shift amount: {}'.format(shift_seconds))
  if is_quantized_sequence(sequence):
    raise QuantizationStatusException(
        'Can shift only unquantized NoteSequences.')

  shifted = music_pb2.NoteSequence()
  shifted.CopyFrom(sequence)

  # Delete subsequence_info because our frame of reference has shifted.
  shifted.ClearField('subsequence_info')

  # Shift notes.
  for note in shifted.notes:
    note.start_time += shift_seconds
    note.end_time += shift_seconds

  events_to_shift = [
      shifted.time_signatures, shifted.key_signatures, shifted.tempos,
      shifted.pitch_bends, shifted.control_changes, shifted.text_annotations,
      shifted.section_annotations]

  for event in itertools.chain(*events_to_shift):
    event.time += shift_seconds

  shifted.total_time += shift_seconds

  return shifted


def remove_redundant_data(sequence):
  """Returns a copy of the sequence with redundant data removed.

  An event is considered redundant if it is a time signature, a key signature,
  or a tempo that differs from the previous event of the same type only by time.
  For example, a tempo mark of 120 qpm at 5 seconds would be considered
  redundant if it followed a tempo mark of 120 qpm and 4 seconds.

  Fields in sequence_metadata are considered redundant if the same string is
  repeated.

  Args:
    sequence: The sequence to process.

  Returns:
    A new sequence with redundant events removed.
  """
  fixed_sequence = copy.deepcopy(sequence)
  for events in [
      fixed_sequence.time_signatures, fixed_sequence.key_signatures,
      fixed_sequence.tempos]:
    events.sort(key=lambda e: e.time)
    for i in range(len(events) - 1, 0, -1):
      tmp_ts = copy.deepcopy(events[i])
      tmp_ts.time = events[i - 1].time
      # If the only difference between the two events is time, then delete the
      # second one.
      if tmp_ts == events[i - 1]:
        del events[i]

  if fixed_sequence.HasField('sequence_metadata'):
    # Add composers and genres, preserving order, but dropping duplicates.
    del fixed_sequence.sequence_metadata.composers[:]
    added_composer = set()
    for composer in sequence.sequence_metadata.composers:
      if composer not in added_composer:
        fixed_sequence.sequence_metadata.composers.append(composer)
        added_composer.add(composer)

    del fixed_sequence.sequence_metadata.genre[:]
    added_genre = set()
    for genre in sequence.sequence_metadata.genre:
      if genre not in added_genre:
        fixed_sequence.sequence_metadata.genre.append(genre)
        added_genre.add(genre)

  return fixed_sequence


def concatenate_sequences(sequences, sequence_durations=None):
  """Concatenate a series of NoteSequences together.

  Individual sequences will be shifted using shift_sequence_times and then
  merged together using the protobuf MergeFrom method. This means that any
  global values (e.g., ticks_per_quarter) will be overwritten by each sequence
  and only the final value will be used. After this, redundant data will be
  removed with remove_redundant_data.

  Args:
    sequences: A list of sequences to concatenate.
    sequence_durations: An optional list of sequence durations to use. If not
      specified, the total_time value will be used. Specifying durations is
      useful if the sequences to be concatenated are effectively longer than
      their total_time (e.g., a sequence that ends with a rest).

  Returns:
    A new sequence that is the result of concatenating *sequences.

  Raises:
    ValueError: If the length of sequences and sequence_durations do not match
        or if a specified duration is less than the total_time of the sequence.
  """
  if sequence_durations and len(sequences) != len(sequence_durations):
    raise ValueError(
        'sequences and sequence_durations must be the same length.')
  current_total_time = 0
  cat_seq = music_pb2.NoteSequence()
  for i in range(len(sequences)):
    sequence = sequences[i]
    if sequence_durations and sequence_durations[i] < sequence.total_time:
      raise ValueError(
          'Specified sequence duration ({}) must not be less than the '
          'total_time of the sequence ({})'.format(
              sequence_durations[i], sequence.total_time))
    if current_total_time > 0:
      cat_seq.MergeFrom(shift_sequence_times(sequence, current_total_time))
    else:
      cat_seq.MergeFrom(sequence)

    if sequence_durations:
      current_total_time += sequence_durations[i]
    else:
      current_total_time = cat_seq.total_time

  # Delete subsequence_info because we've joined several subsequences.
  cat_seq.ClearField('subsequence_info')

  return remove_redundant_data(cat_seq)


def expand_section_groups(sequence):
  """Expands a NoteSequence based on its section_groups.

  Args:
    sequence: The sequence to expand.

  Returns:
    A copy of the original sequence, expanded based on its section_groups. If
    the sequence has no section_groups, a copy of the original sequence will be
    returned.
  """
  if not sequence.section_groups:
    return copy.deepcopy(sequence)

  sections = {}
  section_durations = {}
  for i in range(len(sequence.section_annotations)):
    section_id = sequence.section_annotations[i].section_id
    start_time = sequence.section_annotations[i].time
    if i < len(sequence.section_annotations) - 1:
      end_time = sequence.section_annotations[i + 1].time
    else:
      end_time = sequence.total_time

    subsequence = extract_subsequence(sequence, start_time, end_time)
    # This is a subsequence, so the section_groups no longer make sense.
    del subsequence.section_groups[:]
    # This subsequence contains only 1 section and it has been shifted to time
    # 0.
    del subsequence.section_annotations[:]
    subsequence.section_annotations.add(time=0, section_id=section_id)

    sections[section_id] = subsequence
    section_durations[section_id] = end_time - start_time

  # Recursively expand section_groups.
  def sections_in_group(section_group):
    sections = []
    for section in section_group.sections:
      field = section.WhichOneof('section_type')
      if field == 'section_id':
        sections.append(section.section_id)
      elif field == 'section_group':
        sections.extend(sections_in_group(section.section_group))
    return sections * section_group.num_times

  sections_to_concat = []
  for section_group in sequence.section_groups:
    sections_to_concat.extend(sections_in_group(section_group))

  return concatenate_sequences(
      [sections[i] for i in sections_to_concat],
      [section_durations[i] for i in sections_to_concat])


def _is_power_of_2(x):
  return x and not x & (x - 1)


def is_quantized_sequence(note_sequence):
  """Returns whether or not a NoteSequence proto has been quantized.

  Args:
    note_sequence: A music_pb2.NoteSequence proto.

  Returns:
    True if `note_sequence` is quantized, otherwise False.
  """
  # If the QuantizationInfo message has a non-zero steps_per_quarter or
  # steps_per_second, assume that the proto has been quantized.
  return (note_sequence.quantization_info.steps_per_quarter > 0 or
          note_sequence.quantization_info.steps_per_second > 0)


def is_relative_quantized_sequence(note_sequence):
  """Returns whether a NoteSequence proto has been quantized relative to tempo.

  Args:
    note_sequence: A music_pb2.NoteSequence proto.

  Returns:
    True if `note_sequence` is quantized relative to tempo, otherwise False.
  """
  # If the QuantizationInfo message has a non-zero steps_per_quarter, assume
  # that the proto has been quantized relative to tempo.
  return note_sequence.quantization_info.steps_per_quarter > 0


def is_absolute_quantized_sequence(note_sequence):
  """Returns whether a NoteSequence proto has been quantized by absolute time.

  Args:
    note_sequence: A music_pb2.NoteSequence proto.

  Returns:
    True if `note_sequence` is quantized by absolute time, otherwise False.
  """
  # If the QuantizationInfo message has a non-zero steps_per_second, assume
  # that the proto has been quantized by absolute time.
  return note_sequence.quantization_info.steps_per_second > 0


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


def assert_is_relative_quantized_sequence(note_sequence):
  """Confirms that a NoteSequence proto has been quantized relative to tempo.

  Args:
    note_sequence: A music_pb2.NoteSequence proto.

  Raises:
    QuantizationStatusException: If the sequence is not quantized relative to
        tempo.
  """
  if not is_relative_quantized_sequence(note_sequence):
    raise QuantizationStatusException('NoteSequence %s is not quantized or is '
                                      'quantized based on absolute timing.' %
                                      note_sequence.id)


def assert_is_absolute_quantized_sequence(note_sequence):
  """Confirms that a NoteSequence proto has been quantized by absolute time.

  Args:
    note_sequence: A music_pb2.NoteSequence proto.

  Raises:
    QuantizationStatusException: If the sequence is not quantized by absolute
    time.
  """
  if not is_absolute_quantized_sequence(note_sequence):
    raise QuantizationStatusException('NoteSequence %s is not quantized or is '
                                      'quantized based on relative timing.' %
                                      note_sequence.id)


def steps_per_bar_in_quantized_sequence(note_sequence):
  """Calculates steps per bar in a NoteSequence that has been quantized.

  Args:
    note_sequence: The NoteSequence to examine.

  Returns:
    Steps per bar as a floating point number.
  """
  assert_is_relative_quantized_sequence(note_sequence)

  quarters_per_beat = 4.0 / note_sequence.time_signatures[0].denominator
  quarters_per_bar = (quarters_per_beat *
                      note_sequence.time_signatures[0].numerator)
  steps_per_bar_float = (note_sequence.quantization_info.steps_per_quarter *
                         quarters_per_bar)
  return steps_per_bar_float


def split_note_sequence(note_sequence, hop_size_seconds,
                        skip_splits_inside_notes=False):
  """Split one NoteSequence into many at specified time intervals.

  If `hop_size_seconds` is a scalar, this function splits a NoteSequence into
  multiple NoteSequences, all of fixed size (unless `split_notes` is False, in
  which case splits that would have truncated notes will be skipped; i.e. each
  split will either happen at a multiple of `hop_size_seconds` or not at all).
  Each of the resulting NoteSequences is shifted to start at time zero.

  If `hop_size_seconds` is a list, the NoteSequence will be split at each time
  in the list (unless `split_notes` is False as above).

  Args:
    note_sequence: The NoteSequence to split.
    hop_size_seconds: The hop size, in seconds, at which the NoteSequence will
        be split. Alternatively, this can be a Python list of times in seconds
        at which to split the NoteSequence.
    skip_splits_inside_notes: If False, the NoteSequence will be split at all
        hop positions, regardless of whether or not any notes are sustained
        across the potential split time, thus sustained notes will be truncated.
        If True, the NoteSequence will not be split at positions that occur
        within sustained notes.

  Returns:
    A Python list of NoteSequences.
  """
  notes_by_start_time = sorted(list(note_sequence.notes),
                               key=lambda note: note.start_time)
  note_idx = 0
  notes_crossing_split = []

  if isinstance(hop_size_seconds, list):
    split_times = sorted(hop_size_seconds)
  else:
    split_times = np.arange(
        hop_size_seconds, note_sequence.total_time, hop_size_seconds)

  valid_split_times = [0.0]

  for split_time in split_times:
    # Update notes crossing potential split.
    while (note_idx < len(notes_by_start_time) and
           notes_by_start_time[note_idx].start_time < split_time):
      notes_crossing_split.append(notes_by_start_time[note_idx])
      note_idx += 1
    notes_crossing_split = [note for note in notes_crossing_split
                            if note.end_time > split_time]

    if not (skip_splits_inside_notes and notes_crossing_split):
      valid_split_times.append(split_time)

  # Handle the final subsequence.
  if note_sequence.total_time > valid_split_times[-1]:
    valid_split_times.append(note_sequence.total_time)

  if len(valid_split_times) > 1:
    return _extract_subsequences(note_sequence, valid_split_times)
  else:
    return []


def split_note_sequence_on_time_changes(note_sequence,
                                        skip_splits_inside_notes=False):
  """Split one NoteSequence into many around time signature and tempo changes.

  This function splits a NoteSequence into multiple NoteSequences, each of which
  contains only a single time signature and tempo, unless `split_notes` is False
  in which case all time signature and tempo changes occur within sustained
  notes. Each of the resulting NoteSequences is shifted to start at time zero.

  Args:
    note_sequence: The NoteSequence to split.
    skip_splits_inside_notes: If False, the NoteSequence will be split at all
        time changes, regardless of whether or not any notes are sustained
        across the time change. If True, the NoteSequence will not be split at
        time changes that occur within sustained notes.

  Returns:
    A Python list of NoteSequences.
  """
  current_numerator = 4
  current_denominator = 4
  current_qpm = constants.DEFAULT_QUARTERS_PER_MINUTE

  time_signatures_and_tempos = sorted(
      list(note_sequence.time_signatures) + list(note_sequence.tempos),
      key=lambda t: t.time)
  time_signatures_and_tempos = [t for t in time_signatures_and_tempos
                                if t.time < note_sequence.total_time]

  notes_by_start_time = sorted(list(note_sequence.notes),
                               key=lambda note: note.start_time)
  note_idx = 0
  notes_crossing_split = []

  valid_split_times = [0.0]

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

    # Update notes crossing potential split.
    while (note_idx < len(notes_by_start_time) and
           notes_by_start_time[note_idx].start_time < time_change.time):
      notes_crossing_split.append(notes_by_start_time[note_idx])
      note_idx += 1
    notes_crossing_split = [note for note in notes_crossing_split
                            if note.end_time > time_change.time]

    if time_change.time > valid_split_times[-1]:
      if not (skip_splits_inside_notes and notes_crossing_split):
        valid_split_times.append(time_change.time)

    # Even if we didn't split here, update the current time signature or tempo.
    if isinstance(time_change, music_pb2.NoteSequence.TimeSignature):
      current_numerator = time_change.numerator
      current_denominator = time_change.denominator
    else:
      current_qpm = time_change.qpm

  # Handle the final subsequence.
  if note_sequence.total_time > valid_split_times[-1]:
    valid_split_times.append(note_sequence.total_time)

  if len(valid_split_times) > 1:
    return _extract_subsequences(note_sequence, valid_split_times)
  else:
    return []


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


def _quantize_notes(note_sequence, steps_per_second):
  """Quantize the notes and chords of a NoteSequence proto in place.

  Note start and end times, and chord times are snapped to a nearby quantized
  step, and the resulting times are stored in a separate field (e.g.,
  quantized_start_step). See the comments above `QUANTIZE_CUTOFF` for details on
  how the quantizing algorithm works.

  Args:
    note_sequence: A music_pb2.NoteSequence protocol buffer. Will be modified in
        place.
    steps_per_second: Each second will be divided into this many quantized time
        steps.

  Raises:
    NegativeTimeException: If a note or chord occurs at a negative time.
  """
  for note in note_sequence.notes:
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
    if note.quantized_end_step > note_sequence.total_quantized_steps:
      note_sequence.total_quantized_steps = note.quantized_end_step

  # Also quantize control changes and text annotations.
  for event in itertools.chain(
      note_sequence.control_changes, note_sequence.text_annotations):
    # Quantize the event time, disallowing negative time.
    event.quantized_step = quantize_to_step(event.time, steps_per_second)
    if event.quantized_step < 0:
      raise NegativeTimeException(
          'Got negative event time: step = %s' % event.quantized_step)


def quantize_note_sequence(note_sequence, steps_per_quarter):
  """Quantize a NoteSequence proto relative to tempo.

  The input NoteSequence is copied and quantization-related fields are
  populated. Sets the `steps_per_quarter` field in the `quantization_info`
  message in the NoteSequence.

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
        has a 0 numerator or a denominator which is not a power of 2.
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

  if qns.time_signatures[0].numerator == 0:
    raise BadTimeSignatureException(
        'Numerator is 0. Time signature: %d/%d' %
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
  _quantize_notes(qns, steps_per_second)

  return qns


def quantize_note_sequence_absolute(note_sequence, steps_per_second):
  """Quantize a NoteSequence proto using absolute event times.

  The input NoteSequence is copied and quantization-related fields are
  populated. Sets the `steps_per_second` field in the `quantization_info`
  message in the NoteSequence.

  Note start and end times, and chord times are snapped to a nearby quantized
  step, and the resulting times are stored in a separate field (e.g.,
  quantized_start_step). See the comments above `QUANTIZE_CUTOFF` for details on
  how the quantizing algorithm works.

  Tempos and time signatures will be copied but ignored.

  Args:
    note_sequence: A music_pb2.NoteSequence protocol buffer.
    steps_per_second: Each second will be divided into this many quantized time
        steps.

  Returns:
    A copy of the original NoteSequence, with quantized times added.

  Raises:
    NegativeTimeException: If a note or chord occurs at a negative time.
  """
  qns = copy.deepcopy(note_sequence)
  qns.quantization_info.steps_per_second = steps_per_second

  qns.total_quantized_steps = quantize_to_step(qns.total_time, steps_per_second)
  _quantize_notes(qns, steps_per_second)

  return qns


def stretch_note_sequence(note_sequence, stretch_factor):
  """Apply a constant temporal stretch to a NoteSequence proto.

  Args:
    note_sequence: The NoteSequence to stretch.
    stretch_factor: How much to stretch the NoteSequence. Values greater than
        one increase the length of the NoteSequence (making it "slower"). Values
        less than one decrease the length of the NoteSequence (making it
        "faster").

  Returns:
    A stretched copy of the original NoteSequence.

  Raises:
    QuantizationStatusException: If the `note_sequence` is quantized. Only
        unquantized NoteSequences can be stretched.
  """
  if is_quantized_sequence(note_sequence):
    raise QuantizationStatusException(
        'Can only stretch unquantized NoteSequence.')

  stretched_sequence = music_pb2.NoteSequence()
  stretched_sequence.CopyFrom(note_sequence)

  if stretch_factor == 1.0:
    return stretched_sequence

  # Stretch all notes.
  for note in stretched_sequence.notes:
    note.start_time *= stretch_factor
    note.end_time *= stretch_factor
  stretched_sequence.total_time *= stretch_factor

  # Stretch all other event times.
  events = itertools.chain(
      stretched_sequence.time_signatures,
      stretched_sequence.key_signatures,
      stretched_sequence.tempos,
      stretched_sequence.pitch_bends,
      stretched_sequence.control_changes,
      stretched_sequence.text_annotations)
  for event in events:
    event.time *= stretch_factor

  # Stretch tempos.
  for tempo in stretched_sequence.tempos:
    tempo.qpm /= stretch_factor

  return stretched_sequence


# Constants for processing the note/sustain stream.
# The order here matters because we we want to process 'on' events before we
# process 'off' events, and we want to process sustain events before note
# events.
_SUSTAIN_ON = 0
_SUSTAIN_OFF = 1
_NOTE_ON = 2
_NOTE_OFF = 3


def apply_sustain_control_changes(note_sequence, sustain_control_number=64):
  """Returns a new NoteSequence with sustain pedal control changes applied.

  Extends each note within a sustain to either the beginning of the next note of
  the same pitch or the end of the sustain period, whichever happens first. This
  is done on a per instrument basis, so notes are only affected by sustain
  events for the same instrument.

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

  sequence = copy.deepcopy(note_sequence)

  # Sort all note on/off and sustain on/off events.
  events = []
  events.extend([(note.start_time, _NOTE_ON, note)
                 for note in sequence.notes])
  events.extend([(note.end_time, _NOTE_OFF, note)
                 for note in sequence.notes])

  for cc in sequence.control_changes:
    if cc.control_number != sustain_control_number:
      continue
    value = cc.control_value
    if value < 0 or value > 127:
      tf.logging.warn(
          'Sustain control change has out of range value: %d', value)
    if value >= 64:
      events.append((cc.time, _SUSTAIN_ON, cc))
    elif value < 64:
      events.append((cc.time, _SUSTAIN_OFF, cc))

  # Sort, using the event type constants to ensure the order events are
  # processed.
  events.sort(key=itemgetter(0))

  # Lists of active notes, keyed by instrument.
  active_notes = collections.defaultdict(list)
  # Whether sustain is active for a given instrument.
  sus_active = collections.defaultdict(lambda: False)

  # Iterate through all sustain on/off and note on/off events in order.
  time = 0
  for time, event_type, event in events:
    if event_type == _SUSTAIN_ON:
      sus_active[event.instrument] = True
    elif event_type == _SUSTAIN_OFF:
      sus_active[event.instrument] = False
      # End all notes for the instrument that were being extended.
      new_active_notes = []
      for note in active_notes[event.instrument]:
        if note.end_time < time:
          # This note was being extended because of sustain.
          # Update the end time and don't keep it in the list.
          note.end_time = time
          if time > sequence.total_time:
            sequence.total_time = time
        else:
          # This note is actually still active, keep it.
          new_active_notes.append(note)
      active_notes[event.instrument] = new_active_notes
    elif event_type == _NOTE_ON:
      if sus_active[event.instrument]:
        # If sustain is on, end all previous notes with the same pitch.
        new_active_notes = []
        for note in active_notes[event.instrument]:
          if note.pitch == event.pitch:
            note.end_time = time
            if note.start_time == note.end_time:
              # This note now has no duration because another note of the same
              # pitch started at the same time. Only one of these notes should
              # be preserved, so delete this one.
              # TODO(fjord): A more correct solution would probably be to
              # preserve both notes and make the same duration, but that is a
              # little more complicated to implement. Will keep this solution
              # until we find that we need the more complex one.
              sequence.notes.remove(note)
          else:
            new_active_notes.append(note)
        active_notes[event.instrument] = new_active_notes
      # Add this new note to the list of active notes.
      active_notes[event.instrument].append(event)
    elif event_type == _NOTE_OFF:
      if sus_active[event.instrument]:
        # Note continues until another note of the same pitch or sustain ends.
        pass
      else:
        # Remove this particular note from the active list.
        # It may have already been removed if a note of the same pitch was
        # played when sustain was active.
        if event in active_notes[event.instrument]:
          active_notes[event.instrument].remove(event)
    else:
      raise AssertionError('Invalid event_type: %s' % event_type)

  # End any notes that were still active due to sustain.
  for instrument in active_notes.values():
    for note in instrument:
      note.end_time = time
      sequence.total_time = time

  return sequence


def infer_dense_chords_for_sequence(
    sequence, instrument=None, min_notes_per_chord=3):
  """Infers chords for a NoteSequence and adds them as TextAnnotations.

  For each set of simultaneously-active notes in a NoteSequence (optionally for
  only one instrument), infers a chord symbol and adds it to NoteSequence as a
  TextAnnotation. Every change in the set of active notes will result in a new
  chord symbol unless the new set is smaller than `min_notes_per_chord`.

  If `sequence` is quantized, simultaneity will be determined by quantized steps
  instead of time.

  Not to be confused with the chord inference in magenta.music.chord_inference
  that attempts to infer a more natural chord sequence with changes at regular
  metric intervals.

  Args:
    sequence: The NoteSequence for which chords will be inferred. Will be
        modified in place.
    instrument: The instrument number whose notes will be used for chord
        inference. If None, all instruments will be used.
    min_notes_per_chord: The minimum number of simultaneous notes for which to
        infer a chord.

  Raises:
    ChordSymbolException: If a chord cannot be determined for a set of
    simultaneous notes in `sequence`.
  """
  notes = [note for note in sequence.notes
           if not note.is_drum and (instrument is None or
                                    note.instrument == instrument)]
  sorted_notes = sorted(notes, key=lambda note: note.start_time)

  # If the sequence is quantized, use quantized steps instead of time.
  if is_quantized_sequence(sequence):
    note_start = lambda note: note.quantized_start_step
    note_end = lambda note: note.quantized_end_step
  else:
    note_start = lambda note: note.start_time
    note_end = lambda note: note.end_time

  # Sort all note start and end events.
  onsets = [(note_start(note), idx, False)
            for idx, note in enumerate(sorted_notes)]
  offsets = [(note_end(note), idx, True)
             for idx, note in enumerate(sorted_notes)]
  events = sorted(onsets + offsets)

  current_time = 0
  current_figure = constants.NO_CHORD
  active_notes = set()

  for time, idx, is_offset in events:
    if time > current_time:
      active_pitches = set(sorted_notes[idx].pitch for idx in active_notes)
      if len(active_pitches) >= min_notes_per_chord:
        # Infer a chord symbol for the active pitches.
        figure = chord_symbols_lib.pitches_to_chord_symbol(active_pitches)

        if figure != current_figure:
          # Add a text annotation to the sequence.
          text_annotation = sequence.text_annotations.add()
          text_annotation.text = figure
          text_annotation.annotation_type = CHORD_SYMBOL
          if is_quantized_sequence(sequence):
            text_annotation.time = (
                current_time * sequence.quantization_info.steps_per_quarter)
            text_annotation.quantized_step = current_time
          else:
            text_annotation.time = current_time

        current_figure = figure

    current_time = time
    if is_offset:
      active_notes.remove(idx)
    else:
      active_notes.add(idx)

  assert not active_notes
