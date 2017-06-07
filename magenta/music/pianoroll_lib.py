# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Utility functions for working with pianoroll sequences."""

from __future__ import division

import copy

# internal imports
import numpy as np

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2

DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER
MAX_MIDI_PITCH = 108  # Max piano pitch.
MIN_MIDI_PITCH = 21  # Min piano pitch.
STANDARD_PPQ = constants.STANDARD_PPQ


class PianorollSequence(events_lib.EventSequence):
  """Stores a polyphonic sequence as a pianoroll.

  Events are collections of active pitches at each step, offset from
  `min_pitch`.
  """

  def __init__(self, quantized_sequence=None, events_list=None,
               steps_per_quarter=None, start_step=0, min_pitch=MIN_MIDI_PITCH,
               max_pitch=MAX_MIDI_PITCH, split_repeats=True, shift_range=False):
    """Construct a PianorollSequence.

    Exactly one of `quantized_sequence` or `steps_per_quarter` must be supplied.
    At most one of `quantized_sequence` and `events_list` may be supplied.

    Args:
      quantized_sequence: an optional quantized NoteSequence proto to base
          PianorollSequence on.
      events_list: an optional list of Pianoroll events to base
          PianorollSequence on.
      steps_per_quarter: how many steps a quarter note represents. Must be
          provided if `quanitzed_sequence` not given.
      start_step: The offset of this sequence relative to the
          beginning of the source sequence. If a quantized sequence is used as
          input, only notes starting after this step will be considered.
      min_pitch: The minimum valid pitch value, inclusive.
      max_pitch: The maximum valid pitch value, inclusive.
      split_repeats: Whether to force repeated notes to have a 0-state step
          between them when initializing from a quantized NoteSequence.
      shift_range: If True, assume that the given events_list is in the full
         MIDI pitch range and needs to be shifted and filtered based on
         `min_pitch` and `max_pitch`.
    """
    assert (quantized_sequence, steps_per_quarter).count(None) == 1
    assert (quantized_sequence, events_list).count(None) >= 1

    self._min_pitch = min_pitch
    self._max_pitch = max_pitch

    if quantized_sequence:
      sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
      self._events = self._from_quantized_sequence(quantized_sequence,
                                                   start_step, min_pitch,
                                                   max_pitch, split_repeats)
      self._steps_per_quarter = (
          quantized_sequence.quantization_info.steps_per_quarter)
    else:
      self._events = []
      self._steps_per_quarter = steps_per_quarter
      if events_list:
        for e in events_list:
          self.append(e, shift_range)
    self._start_step = start_step

  @property
  def start_step(self):
    return self._start_step

  @property
  def steps_per_quarter(self):
    return self._steps_per_quarter

  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads with silence to make the
    sequence the specified length. If it is too long, it will be truncated to
    the requested length.

    Note that this will append a STEP_END event to the end of the sequence if
    there is an unfinished step.

    Args:
      steps: How many quantized steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if from_left:
      raise NotImplementedError('from_left is not supported')

    # Then trim or pad as needed.
    if self.num_steps < steps:
      self._events += [()] * (steps - self.num_steps)
    elif self.num_steps > steps:
      del self._events[steps:]
    assert self.num_steps == steps

  def append(self, event, shift_range=False):
    """Appends the event to the end of the sequence.

    Args:
      event: The polyphonic event to append to the end.
      shift_range: If True, assume that the given event is in the full MIDI
         pitch range and needs to be shifted and filtered based on `min_pitch`
         and `max_pitch`.
    Raises:
      ValueError: If `event` is not a valid polyphonic event.
    """
    if shift_range:
      event = tuple(p - self._min_pitch for p in event
                    if self._min_pitch <= p <= self._max_pitch)
    self._events.append(event)

  def __len__(self):
    """How many events are in this sequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __iter__(self):
    """Return an iterator over the events in this sequence."""
    return iter(self._events)

  @property
  def end_step(self):
    return self.start_step + self.num_steps

  @property
  def num_steps(self):
    """Returns how many steps long this sequence is.

    Returns:
      Length of the sequence in quantized steps.
    """
    return len(self)

  @staticmethod
  def _from_quantized_sequence(
      quantized_sequence, start_step, min_pitch, max_pitch, split_repeats):
    """Populate self with events from the given quantized NoteSequence object.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start converting the sequence at this time step.
          Assumed to be the beginning of a bar.
      min_pitch: The minimum valid pitch value, inclusive.
      max_pitch: The maximum valid pitch value, inclusive.
      split_repeats: Whether to force repeated notes to have a 0-state step
          between them.

    Returns:
      A list of events.
    """
    piano_roll = np.zeros(
        (quantized_sequence.total_quantized_steps - start_step,
         max_pitch - min_pitch + 1), np.bool)

    for note in quantized_sequence.notes:
      if note.quantized_start_step < start_step:
        continue
      if not min_pitch <= note.pitch <= max_pitch:
        continue
      note_pitch_offset = note.pitch - min_pitch
      note_start_offset = note.quantized_start_step - start_step
      note_end_offset = note.quantized_end_step - start_step

      if split_repeats:
        piano_roll[note_start_offset - 1, note_pitch_offset] = 0
      piano_roll[note_start_offset:note_end_offset, note_pitch_offset] = 1

    events = [tuple(np.where(frame)[0]) for frame in piano_roll]

    return events

  def to_sequence(self,
                  velocity=100,
                  instrument=0,
                  program=0,
                  qpm=constants.DEFAULT_QUARTERS_PER_MINUTE,
                  base_note_sequence=None):
    """Converts the PianorollSequence to NoteSequence proto.

    Args:
      velocity: Midi velocity to give each note. Between 1 and 127 (inclusive).
      instrument: Midi instrument to give each note.
      program: Midi program to give each note.
      qpm: Quarter notes per minute (float).
      base_note_sequence: A NoteSequence to use a starting point. Must match the
          specified qpm.

    Raises:
      ValueError: if an unknown event is encountered.

    Returns:
      A NoteSequence proto.
    """
    seconds_per_step = 60.0 / qpm / self._steps_per_quarter

    sequence_start_time = self.start_step * seconds_per_step

    if base_note_sequence:
      sequence = copy.deepcopy(base_note_sequence)
      if sequence.tempos[0].qpm != qpm:
        raise ValueError(
            'Supplied QPM (%d) does not match QPM of base_note_sequence (%d)'
            % (qpm, sequence.tempos[0].qpm))
    else:
      sequence = music_pb2.NoteSequence()
      sequence.tempos.add().qpm = qpm
      sequence.ticks_per_quarter = STANDARD_PPQ

    step = 0
    # Keep a dictionary of open notes for each pitch.
    open_notes = {}
    for step, event in enumerate(self):
      frame_pitches = set(event)
      open_pitches = set(open_notes)

      for pitch_to_close in open_pitches - frame_pitches:
        note_to_close = open_notes[pitch_to_close]
        note_to_close.end_time = step * seconds_per_step + sequence_start_time
        del open_notes[pitch_to_close]

      for pitch_to_open in frame_pitches - open_pitches:
        new_note = sequence.notes.add()
        new_note.start_time = step * seconds_per_step + sequence_start_time
        new_note.pitch = pitch_to_open + self._min_pitch
        new_note.velocity = velocity
        new_note.instrument = instrument
        new_note.program = program
        open_notes[pitch_to_open] = new_note

    final_step = step + (len(open_notes) > 0)  # pylint: disable=g-explicit-length-test
    for note_to_close in open_notes.values():
      note_to_close.end_time = (
          final_step * seconds_per_step + sequence_start_time)

    sequence.total_time = seconds_per_step * final_step + sequence_start_time
    if sequence.notes:
      assert sequence.total_time >= sequence.notes[-1].end_time

    return sequence


def extract_pianoroll_sequences(
    quantized_sequence, start_step=0, min_steps_discard=None,
    max_steps_discard=None):
  """Extracts a polyphonic track from the given quantized NoteSequence.

  Currently, this extracts only one pianoroll from a given track.

  Args:
    quantized_sequence: A quantized NoteSequence.
    start_step: Start extracting a sequence at this time step. Assumed
        to be the beginning of a bar.
    min_steps_discard: Minimum length of tracks in steps. Shorter tracks are
        discarded.
    max_steps_discard: Maximum length of tracks in steps. Longer tracks are
        discarded.

  Returns:
    pianoroll_seqs: A python list of PianorollSequence instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

  stats = dict([(stat_name, statistics.Counter(stat_name)) for stat_name in
                ['pianoroll_tracks_discarded_too_short',
                 'pianoroll_tracks_discarded_too_long',
                 'pianoroll_tracks_discarded_more_than_1_program']])

  steps_per_bar = sequences_lib.steps_per_bar_in_quantized_sequence(
      quantized_sequence)

  # Create a histogram measuring lengths (in bars not steps).
  stats['pianoroll_track_lengths_in_bars'] = statistics.Histogram(
      'pianoroll_track_lengths_in_bars',
      [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, 1000])

  # Allow only 1 program.
  programs = set()
  for note in quantized_sequence.notes:
    programs.add(note.program)
  if len(programs) > 1:
    stats['pianoroll_tracks_discarded_more_than_1_program'].increment()
    return [], stats.values()

  # Translate the quantized sequence into a PianorollSequence.
  pianoroll_seq = PianorollSequence(quantized_sequence=quantized_sequence,
                                    start_step=start_step)

  pianoroll_seqs = []
  num_steps = pianoroll_seq.num_steps

  if min_steps_discard is not None and num_steps < min_steps_discard:
    stats['pianoroll_tracks_discarded_too_short'].increment()
  elif max_steps_discard is not None and num_steps > max_steps_discard:
    stats['pianoroll_tracks_discarded_too_long'].increment()
  else:
    pianoroll_seqs.append(pianoroll_seq)
    stats['pianoroll_track_lengths_in_bars'].increment(
        num_steps // steps_per_bar)
  return pianoroll_seqs, stats.values()
