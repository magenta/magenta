# Copyright 2019 The Magenta Authors.
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

"""Utility functions for working with drums.

Use extract_drum_tracks to extract drum tracks from a quantized NoteSequence.

Use DrumTrack.to_sequence to write a drum track to a NoteSequence proto. Then
use midi_io.sequence_proto_to_midi_file to write that NoteSequence to a midi
file.
"""

import collections
import operator

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.music.protobuf import music_pb2

MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH
MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER
STANDARD_PPQ = constants.STANDARD_PPQ


class DrumTrack(events_lib.SimpleEventSequence):
  """Stores a quantized stream of drum events.

  DrumTrack is an intermediate representation that all drum models can use.
  Quantized sequence to DrumTrack code will do work to align drum notes and
  extract drum tracks. Model-specific code then needs to convert DrumTrack
  to SequenceExample protos for TensorFlow.

  DrumTrack implements an iterable object. Simply iterate to retrieve the drum
  events.

  DrumTrack events are Python frozensets of simultaneous MIDI drum "pitches",
  where each pitch indicates a type of drum. An empty frozenset indicates no
  drum notes. Unlike melody notes, drum notes are not considered to have
  durations.

  Drum tracks can start at any non-negative time, and are shifted left so that
  the bar containing the first drum event is the first bar.

  Attributes:
    start_step: The offset of the first step of the drum track relative to the
        beginning of the source sequence. Will always be the first step of a
        bar.
    end_step: The offset to the beginning of the bar following the last step
       of the drum track relative the beginning of the source sequence. Will
       always be the first step of a bar.
    steps_per_quarter: Number of steps in in a quarter note.
    steps_per_bar: Number of steps in a bar (measure) of music.
  """

  def __init__(self, events=None, **kwargs):
    """Construct a DrumTrack."""
    if 'pad_event' in kwargs:
      del kwargs['pad_event']
    super(DrumTrack, self).__init__(pad_event=frozenset(),
                                    events=events, **kwargs)

  def _from_event_list(self, events, start_step=0,
                       steps_per_bar=DEFAULT_STEPS_PER_BAR,
                       steps_per_quarter=DEFAULT_STEPS_PER_QUARTER):
    """Initializes with a list of event values and sets attributes.

    Args:
      events: List of drum events to set drum track to.
      start_step: The integer starting step offset.
      steps_per_bar: The number of steps in a bar.
      steps_per_quarter: The number of steps in a quarter note.

    Raises:
      ValueError: If `events` contains an event that is not a valid drum event.
    """
    for event in events:
      if not isinstance(event, frozenset):
        raise ValueError('Invalid drum event: %s' % event)
      if not all(MIN_MIDI_PITCH <= drum <= MAX_MIDI_PITCH for drum in event):
        raise ValueError('Drum event contains invalid note: %s' % event)
    super(DrumTrack, self)._from_event_list(
        events, start_step=start_step, steps_per_bar=steps_per_bar,
        steps_per_quarter=steps_per_quarter)

  def append(self, event):
    """Appends the event to the end of the drums and increments the end step.

    Args:
      event: The drum event to append to the end.
    Raises:
      ValueError: If `event` is not a valid drum event.
    """
    if not isinstance(event, frozenset):
      raise ValueError('Invalid drum event: %s' % event)
    if not all(MIN_MIDI_PITCH <= drum <= MAX_MIDI_PITCH for drum in event):
      raise ValueError('Drum event contains invalid note: %s' % event)
    super(DrumTrack, self).append(event)

  def from_quantized_sequence(self,
                              quantized_sequence,
                              search_start_step=0,
                              gap_bars=1,
                              pad_end=False,
                              ignore_is_drum=False):
    """Populate self with drums from the given quantized NoteSequence object.

    A drum track is extracted from the given quantized sequence starting at time
    step `start_step`. `start_step` can be used to drive extraction of multiple
    drum tracks from the same quantized sequence. The end step of the extracted
    drum track will be stored in `self._end_step`.

    0 velocity notes are ignored. The drum extraction is ended when there are
    no drums for a time stretch of `gap_bars` in bars (measures) of music. The
    number of time steps per bar is computed from the time signature in
    `quantized_sequence`.

    Each drum event is a Python frozenset of simultaneous (after quantization)
    drum "pitches", or an empty frozenset to indicate no drums are played.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      search_start_step: Start searching for drums at this time step. Assumed to
          be the beginning of a bar.
      gap_bars: If this many bars or more follow a non-empty drum event, the
          drum track is ended.
      pad_end: If True, the end of the drums will be padded with empty events so
          that it will end at a bar boundary.
      ignore_is_drum: Whether accept notes where `is_drum` is False.

    Raises:
      NonIntegerStepsPerBarError: If `quantized_sequence`'s bar length
          (derived from its time signature) is not an integer number of time
          steps.
    """
    sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
    self._reset()

    steps_per_bar_float = sequences_lib.steps_per_bar_in_quantized_sequence(
        quantized_sequence)
    if steps_per_bar_float % 1 != 0:
      raise events_lib.NonIntegerStepsPerBarError(
          'There are %f timesteps per bar. Time signature: %d/%d' %
          (steps_per_bar_float, quantized_sequence.time_signatures[0].numerator,
           quantized_sequence.time_signatures[0].denominator))
    self._steps_per_bar = steps_per_bar = int(steps_per_bar_float)
    self._steps_per_quarter = (
        quantized_sequence.quantization_info.steps_per_quarter)

    # Group all drum notes that start at the same step.
    all_notes = [note for note in quantized_sequence.notes
                 if ((note.is_drum or ignore_is_drum)  # drums only
                     and note.velocity  # no zero-velocity notes
                     # after start_step only
                     and note.quantized_start_step >= search_start_step)]
    grouped_notes = collections.defaultdict(list)
    for note in all_notes:
      grouped_notes[note.quantized_start_step].append(note)

    # Sort by note start times.
    notes = sorted(grouped_notes.items(), key=operator.itemgetter(0))

    if not notes:
      return

    gap_start_index = 0

    track_start_step = (
        notes[0][0] - (notes[0][0] - search_start_step) % steps_per_bar)
    for start, group in notes:

      start_index = start - track_start_step
      pitches = frozenset(note.pitch for note in group)

      # If a gap of `gap` or more steps is found, end the drum track.
      note_distance = start_index - gap_start_index
      if len(self) and note_distance >= gap_bars * steps_per_bar:  # pylint:disable=len-as-condition
        break

      # Add a drum event, a set of drum "pitches".
      self.set_length(start_index + 1)
      self._events[start_index] = pitches

      gap_start_index = start_index + 1

    if not self._events:
      # If no drum events were added, don't set `_start_step` and `_end_step`.
      return

    self._start_step = track_start_step

    length = len(self)
    # Optionally round up `_end_step` to a multiple of `steps_per_bar`.
    if pad_end:
      length += -len(self) % steps_per_bar
    self.set_length(length)

  def to_sequence(self,
                  velocity=100,
                  instrument=9,
                  program=0,
                  sequence_start_time=0.0,
                  qpm=120.0):
    """Converts the DrumTrack to NoteSequence proto.

    Args:
      velocity: Midi velocity to give each note. Between 1 and 127 (inclusive).
      instrument: Midi instrument to give each note.
      program: Midi program to give each note.
      sequence_start_time: A time in seconds (float) that the first event in the
          sequence will land on.
      qpm: Quarter notes per minute (float).

    Returns:
      A NoteSequence proto encoding the given drum track.
    """
    seconds_per_step = 60.0 / qpm / self.steps_per_quarter

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().qpm = qpm
    sequence.ticks_per_quarter = STANDARD_PPQ

    sequence_start_time += self.start_step * seconds_per_step
    for step, event in enumerate(self):
      for pitch in event:
        # Add a note. All drum notes last a single step.
        note = sequence.notes.add()
        note.start_time = step * seconds_per_step + sequence_start_time
        note.end_time = (step + 1) * seconds_per_step + sequence_start_time
        note.pitch = pitch
        note.velocity = velocity
        note.instrument = instrument
        note.program = program
        note.is_drum = True

    if sequence.notes:
      sequence.total_time = sequence.notes[-1].end_time

    return sequence

  def increase_resolution(self, k):
    """Increase the resolution of a DrumTrack.

    Increases the resolution of a DrumTrack object by a factor of `k`. This uses
    empty events to extend each event in the drum track to be `k` steps long.

    Args:
      k: An integer, the factor by which to increase the resolution of the
          drum track.
    """
    super(DrumTrack, self).increase_resolution(
        k, fill_event=frozenset())


def midi_file_to_drum_track(midi_file, steps_per_quarter=4):
  """Loads a drum track from a MIDI file.

  Args:
    midi_file: Absolute path to MIDI file.
    steps_per_quarter: Quantization of DrumTrack. For example, 4 = 16th notes.

  Returns:
    A DrumTrack object extracted from the MIDI file.
  """
  sequence = midi_io.midi_file_to_sequence_proto(midi_file)
  quantized_sequence = sequences_lib.quantize_note_sequence(
      sequence, steps_per_quarter=steps_per_quarter)
  drum_track = DrumTrack()
  drum_track.from_quantized_sequence(quantized_sequence)
  return drum_track
