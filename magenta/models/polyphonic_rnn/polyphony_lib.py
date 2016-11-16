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
"""Utility functions for working with drums.

Use extract_drum_tracks to extract drum tracks from a quantized NoteSequence.

Use DrumTrack.to_sequence to write a drum track to a NoteSequence proto. Then
use midi_io.sequence_proto_to_midi_file to write that NoteSequence to a midi
file.
"""

from __future__ import division

import collections
from six.moves import range  # pylint: disable=redefined-builtin

# internal imports

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib

DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER

class PolyphonicSequence(events_lib.EventSequence):
  """Stores a polyphonic sequence as a stream of single-note events.

  Events are tuples that encode event type and pitch.
  """

  EVENT_START = 0
  EVENT_END = 1
  EVENT_STEP_END = 2
  EVENT_NEW_NOTE = 3
  EVENT_CONTINUED_NOTE = 4
  PolyphonicEvent = collections.namedtuple('PolyphonicEvent',
                                           ['event_type', 'pitch'])

  def __init__(self, quantized_sequence):
    """Construct a PolyphonicSequence."""
    self._events = self._from_quantized_sequence(quantized_sequence)

  def append(self, event):
    """Appends the event to the end of the sequence.

    Args:
      event: The polyphonic event to append to the end.
    Raises:
      ValueError: If `event` is not a valid polyphonic event.
    """
    if not isinstance(event, PolyphonicEvent):
      raise ValueError('Invalid polyphonic event: %s' % event)
    if not EVENT_START <= event.event_type <= EVENT_CONTINUED_NOTE:
      raise ValueError('Invalid polyphonic type: %s' % event.event_type)
    if not MIN_MIDI_PITCH <= event.pitch <= MAX_MIDI_PITCH:
      raise ValueError('Invalid pitch: %s' % event.pitch)
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

  @staticmethod
  def _from_quantized_sequence(quantized_sequence):
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
      start_step: Start searching for drums at this time step.
      gap_bars: If this many bars or more follow a non-empty drum event, the
          drum track is ended.
      pad_end: If True, the end of the drums will be padded with empty events so
          that it will end at a bar boundary.

    Raises:
      NonIntegerStepsPerBarException: If `quantized_sequence`'s bar length
          (derived from its time signature) is not an integer number of time
          steps.
    """
    sequences_lib.assert_is_quantized_sequence(quantized_sequence)

    pitch_start_steps = collections.defaultdict(list)
    pitch_end_steps = collections.defaultdict(list)

    # !!! filter for only 1 instrument. maybe in encoding pipeline?

    last_step = 0
    for note in quantized_sequence.notes:
      pitch_start_steps[note.quantized_start_step] = note.pitch
      pitch_end_steps[note.quantized_end_step] = note.pitch
      if note.quantized_end_step > last_step:
        last_step = note.quantized_end_step

    events = [PolyphonicEvent(event_type=EVENT_START, pitch=0)]

    active_pitches = []
    for step in range(last_step + 1):
      step_events = []
      for pitch in active_pitches:
        step_events.append(PolyphonicEvent(event_type=EVENT_CONTINUED_NOTE,
                                           pitch=pitch))
      for pitch in pitch_end_steps[step]:
        active_pitches.remove(pitch)

      for pitch in pitch_start_steps[step]:
        active_pitches.append(pitch)
        step_events.append(PolyphonicEvent(event_type=EVENT_NEW_NOTE,
                                           pitch=pitch))

      events.extend(sorted(step_events, key=lambda e: e.pitch))
      events.append(PolyphonicEvent(event_type=EVENT_STEP_END, pitch=0))
    events[-1] = PolyphonicEvent(event_type=EVENT_END, pitch=0)

    return events

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


def extract_polyphonic_sequences(quantized_sequence,
                        min_bars=7,
                        max_steps_truncate=None,
                        max_steps_discard=None,
                        gap_bars=1.0,
                        pad_end=False):
  """Extracts a list of drum tracks from the given quantized NoteSequence.

  This function will search through `quantized_sequence` for drum tracks. A drum
  track can span multiple "tracks" in the sequence. Only one drum track can be
  active at a given time, but multiple drum tracks can be extracted from the
  sequence if gaps are present.

  Once a note-on drum event is encountered, a drum track begins. Gaps of silence
  will be splitting points that divide the sequence into separate drum tracks.
  The minimum size of these gaps are given in `gap_bars`. The size of a bar
  (measure) of music in time steps is computed form the time signature stored in
  `quantized_sequence`.

  A drum track is only used if it is at least `min_bars` bars long.

  After scanning the quantized NoteSequence, a list of all extracted DrumTrack
  objects is returned.

  Args:
    quantized_sequence: A quantized NoteSequence.
    min_bars: Minimum length of drum tracks in number of bars. Shorter drum
        tracks are discarded.
    max_steps_truncate: Maximum number of steps in extracted drum tracks. If
        defined, longer drum tracks are truncated to this threshold. If pad_end
        is also True, drum tracks will be truncated to the end of the last bar
        below this threshold.
    max_steps_discard: Maximum number of steps in extracted drum tracks. If
        defined, longer drum tracks are discarded.
    gap_bars: A drum track comes to an end when this number of bars (measures)
        of no drums is encountered.
    pad_end: If True, the end of the drum track will be padded with empty events
        so that it will end at a bar boundary.

  Returns:
    drum_tracks: A python list of DrumTrack instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.

  Raises:
    NonIntegerStepsPerBarException: If `quantized_sequence`'s bar length
        (derived from its time signature) is not an integer number of time
        steps.
  """
  drum_tracks = []
  stats = dict([(stat_name, statistics.Counter(stat_name)) for stat_name in
                ['drum_tracks_discarded_too_short',
                 'drum_tracks_discarded_too_long',
                 'drum_tracks_truncated']])
  # Create a histogram measuring drum track lengths (in bars not steps).
  # Capture drum tracks that are very small, in the range of the filter lower
  # bound `min_bars`, and large. The bucket intervals grow approximately
  # exponentially.
  stats['drum_track_lengths_in_bars'] = statistics.Histogram(
      'drum_track_lengths_in_bars',
      [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, min_bars // 2, min_bars,
       min_bars + 1, min_bars - 1])

  start = 0

  # Quantize the track into a DrumTrack object.
  # If any notes start at the same time, only one is kept.
  while 1:
    drum_track = DrumTrack()
    try:
      drum_track.from_quantized_sequence(
          quantized_sequence,
          start_step=start,
          gap_bars=gap_bars,
          pad_end=pad_end)
    except events_lib.NonIntegerStepsPerBarException:
      raise
    start = drum_track.end_step
    if not drum_track:
      break

    # Require a certain drum track length.
    if len(drum_track) - 1 < drum_track.steps_per_bar * min_bars:
      stats['drum_tracks_discarded_too_short'].increment()
      continue

    # Discard drum tracks that are too long.
    if max_steps_discard is not None and len(drum_track) > max_steps_discard:
      stats['drum_tracks_discarded_too_long'].increment()
      continue

    # Truncate drum tracks that are too long.
    if max_steps_truncate is not None and len(drum_track) > max_steps_truncate:
      truncated_length = max_steps_truncate
      if pad_end:
        truncated_length -= max_steps_truncate % drum_track.steps_per_bar
      drum_track.set_length(truncated_length)
      stats['drum_tracks_truncated'].increment()

    stats['drum_track_lengths_in_bars'].increment(
        len(drum_track) // drum_track.steps_per_bar)

    drum_tracks.append(drum_track)

  return drum_tracks, stats.values()

