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
"""Utility functions for working with melodies.

Use extract_melodies to extract monophonic melodies from a QuantizedSequence
object.

Use Melody.to_sequence to write a melody to a NoteSequence proto. Then use
midi_io.sequence_proto_to_midi_file to write that NoteSequence to a midi file.

Use MelodyEncoderDecoder.squash_and_encode to convert a Melody object to a
tf.train.SequenceExample of inputs and labels. These SequenceExamples are fed
into the model during training and evaluation.

During melody generation, use MelodyEncoderDecoder.get_inputs_batch to convert
a list of melodies into an inputs batch which can be fed into the model to
predict what the next note should be for each melody. Then use
MelodyEncoderDecoder.extend_event_sequences to extend each of those melodies
with an event sampled from the softmax output by the model.
"""

import abc
import copy

# internal imports
import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


NUM_SPECIAL_MELODY_EVENTS = constants.NUM_SPECIAL_MELODY_EVENTS
MELODY_NOTE_OFF = constants.MELODY_NOTE_OFF
MELODY_NO_EVENT = constants.MELODY_NO_EVENT
MIN_MELODY_EVENT = constants.MIN_MELODY_EVENT
MAX_MELODY_EVENT = constants.MAX_MELODY_EVENT
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH
MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER
STANDARD_PPQ = constants.STANDARD_PPQ
NOTE_KEYS = constants.NOTE_KEYS


class PolyphonicMelodyException(Exception):
  pass


class BadNoteException(Exception):
  pass


class Melody(events_lib.SimpleEventSequence):
  """Stores a quantized stream of monophonic melody events.

  Melody is an intermediate representation that all melody models can use.
  QuantizedSequence to Melody code will do work to align notes and extract
  extract monophonic melodies. Model-specific code then needs to convert Melody
  to SequenceExample protos for TensorFlow.

  Melody implements an iterable object. Simply iterate to retrieve the melody
  events.

  Melody events are integers in range [-2, 127] (inclusive), where negative
  values are the special event events: MELODY_NOTE_OFF, and MELODY_NO_EVENT.
  Non-negative values [0, 127] are note-on events for that midi pitch. A note
  starts at a non-negative value (that is the pitch), and is held through
  subsequent MELODY_NO_EVENT events until either another non-negative value is
  reached (even if the pitch is the same as the previous note), or a
  MELODY_NOTE_OFF event is reached. A MELODY_NOTE_OFF starts at least one step
  of silence, which continues through MELODY_NO_EVENT events until the next
  non-negative value.

  MELODY_NO_EVENT values are treated as default filler. Notes must be inserted
  in ascending order by start time. Note end times will be truncated if the next
  note overlaps.

  Any sustained notes are implicitly turned off at the end of a melody.

  Melodies can start at any non-negative time, and are shifted left so that
  the bar containing the first note-on event is the first bar.

  Attributes:
    start_step: The offset of the first step of the melody relative to the
        beginning of the source sequence. Will always be the first step of a
        bar.
    end_step: The offset to the beginning of the bar following the last step
       of the melody relative the beginning of the source sequence. Will always
       be the first step of a bar.
    steps_per_quarter: Number of steps in in a quarter note.
    steps_per_bar: Number of steps in a bar (measure) of music.
  """

  def __init__(self, events=None, **kwargs):
    """Construct a Melody."""
    super(Melody, self).__init__(pad_event=MELODY_NO_EVENT,
                                 events=events, **kwargs)

  def _from_event_list(self, events, start_step=0,
                       steps_per_bar=DEFAULT_STEPS_PER_BAR,
                       steps_per_quarter=DEFAULT_STEPS_PER_QUARTER):
    """Initializes with a list of event values and sets attributes.

    Args:
      events: List of Melody events to set melody to.
      start_step: The integer starting step offset.
      steps_per_bar: The number of steps in a bar.
      steps_per_quarter: The number of steps in a quarter note.

    Raises:
      ValueError: If `events` contains an event that is not in the proper range.
    """
    for event in events:
      if not MIN_MELODY_EVENT <= event <= MAX_MELODY_EVENT:
        raise ValueError('Melody event out of range: %d' % event)
    super(Melody, self)._from_event_list(
        events, start_step=start_step, steps_per_bar=steps_per_bar,
        steps_per_quarter=steps_per_quarter)

  def __deepcopy__(self, unused_memo=None):
    return type(self)(events=copy.deepcopy(self._events),
                      start_step=self.start_step,
                      steps_per_bar=self.steps_per_bar,
                      steps_per_quarter=self.steps_per_quarter)

  def __eq__(self, other):
    if not isinstance(other, Melody):
      return False
    else:
      return super(Melody, self).__eq__(other)

  def _add_note(self, pitch, start_step, end_step):
    """Adds the given note to the `events` list.

    `start_step` is set to the given pitch. `end_step` is set to NOTE_OFF.
    Everything after `start_step` in `events` is deleted before the note is
    added. `events`'s length will be changed so that the last event has index
    `end_step`.

    Args:
      pitch: Midi pitch. An integer between 0 and 127 inclusive.
      start_step: A non-negative integer step that the note begins on.
      end_step: An integer step that the note ends on. The note is considered to
          end at the onset of the end step. `end_step` must be greater than
          `start_step`.

    Raises:
      BadNoteException: If `start_step` does not precede `end_step`.
    """
    if start_step >= end_step:
      raise BadNoteException(
          'Start step does not precede end step: start=%d, end=%d' %
          (start_step, end_step))

    self.set_length(end_step + 1)

    self._events[start_step] = pitch
    self._events[end_step] = MELODY_NOTE_OFF
    for i in range(start_step + 1, end_step):
      self._events[i] = MELODY_NO_EVENT

  def _get_last_on_off_events(self):
    """Returns indexes of the most recent pitch and NOTE_OFF events.

    Returns:
      A tuple (start_step, end_step) of the last note's on and off event
          indices.

    Raises:
      ValueError: If `events` contains no NOTE_OFF or pitch events.
    """
    last_off = len(self)
    for i in range(len(self) - 1, -1, -1):
      if self._events[i] == MELODY_NOTE_OFF:
        last_off = i
      if self._events[i] >= MIN_MIDI_PITCH:
        return (i, last_off)
    raise ValueError('No events in the stream')

  def get_note_histogram(self):
    """Gets a histogram of the note occurrences in a melody.

    Returns:
      A list of 12 ints, one for each note value (C at index 0 through B at
      index 11). Each int is the total number of times that note occurred in
      the melody.
    """
    np_melody = np.array(self._events, dtype=int)
    return np.bincount(np_melody[np_melody >= MIN_MIDI_PITCH] %
                       NOTES_PER_OCTAVE,
                       minlength=NOTES_PER_OCTAVE)

  def get_major_key_histogram(self):
    """Gets a histogram of the how many notes fit into each key.

    Returns:
      A list of 12 ints, one for each Major key (C Major at index 0 through
      B Major at index 11). Each int is the total number of notes that could
      fit into that key.
    """
    note_histogram = self.get_note_histogram()
    key_histogram = np.zeros(NOTES_PER_OCTAVE)
    for note, count in enumerate(note_histogram):
      key_histogram[NOTE_KEYS[note]] += count
    return key_histogram

  def get_major_key(self):
    """Finds the major key that this melody most likely belongs to.

    If multiple keys match equally, the key with the lowest index is returned,
    where the indexes of the keys are C Major = 0 through B Major = 11.

    Returns:
      An int for the most likely key (C Major = 0 through B Major = 11)
    """
    key_histogram = self.get_major_key_histogram()
    return key_histogram.argmax()

  def append_event(self, event):
    """Appends the event to the end of the melody and increments the end step.

    An implicit NOTE_OFF at the end of the melody will not be respected by this
    modification.

    Args:
      event: The integer Melody event to append to the end.
    Raises:
      ValueError: If `event` is not in the proper range.
    """
    if not MIN_MELODY_EVENT <= event <= MAX_MELODY_EVENT:
      raise ValueError('Event out of range: %d' % event)
    super(Melody, self).append_event(event)

  def from_quantized_sequence(self,
                              quantized_sequence,
                              start_step=0,
                              track=0,
                              gap_bars=1,
                              ignore_polyphonic_notes=False,
                              pad_end=False):
    """Populate self with a melody from the given QuantizedSequence object.

    A monophonic melody is extracted from the given `track` starting at time
    step `start_step`. `track` and `start_step` can be used to drive extraction
    of multiple melodies from the same QuantizedSequence. The end step of the
    extracted melody will be stored in `self._end_step`.

    0 velocity notes are ignored. The melody extraction is ended when there are
    no held notes for a time stretch of `gap_bars` in bars (measures) of music.
    The number of time steps per bar is computed from the time signature in
    `quantized_sequence`.

    `ignore_polyphonic_notes` determines what happens when polyphonic (multiple
    notes start at the same time) data is encountered. If
    `ignore_polyphonic_notes` is true, the highest pitch is used in the melody
    when multiple notes start at the same time. If false, an exception is
    raised.

    Args:
      quantized_sequence: A sequences_lib.QuantizedSequence instance.
      start_step: Start searching for a melody at this time step.
      track: Search for a melody in this track number.
      gap_bars: If this many bars or more follow a NOTE_OFF event, the melody
          is ended.
      ignore_polyphonic_notes: If True, the highest pitch is used in the melody
          when multiple notes start at the same time. If False,
          PolyphonicMelodyException will be raised if multiple notes start at
          the same time.
      pad_end: If True, the end of the melody will be padded with NO_EVENTs so
          that it will end at a bar boundary.

    Raises:
      NonIntegerStepsPerBarException: If `quantized_sequence`'s bar length
          (derived from its time signature) is not an integer number of time
          steps.
      PolyphonicMelodyException: If any of the notes start on the same step
          and `ignore_polyphonic_notes` is False.
    """
    self._reset()

    offset = None
    steps_per_bar_float = quantized_sequence.steps_per_bar()
    if steps_per_bar_float % 1 != 0:
      raise events_lib.NonIntegerStepsPerBarException(
          'There are %f timesteps per bar. Time signature: %d/%d' %
          (steps_per_bar_float, quantized_sequence.time_signature.numerator,
           quantized_sequence.time_signature.denominator))
    self._steps_per_bar = steps_per_bar = int(steps_per_bar_float)
    self._steps_per_quarter = quantized_sequence.steps_per_quarter

    # Sort track by note start times, and secondarily by pitch descending.
    notes = sorted(quantized_sequence.tracks[track],
                   key=lambda note: (note.start, -note.pitch))

    for note in notes:
      if note.start < start_step:
        continue

      # Ignore 0 velocity notes.
      if not note.velocity:
        continue

      if offset is None:
        offset = note.start - note.start % steps_per_bar

      start_index = note.start - offset
      end_index = note.end - offset

      if not self._events:
        # If there are no events, we don't need to check for polyphony.
        self._add_note(note.pitch, start_index, end_index)
        continue

      # If start_step comes before or lands on an already added note's start
      # step, we cannot add it. In that case either discard the melody or keep
      # the highest pitch.
      last_on, last_off = self._get_last_on_off_events()
      on_distance = start_index - last_on
      off_distance = start_index - last_off
      if on_distance == 0:
        if ignore_polyphonic_notes:
          # Keep highest note.
          # Notes are sorted by pitch descending, so if a note is already at
          # this position its the highest pitch.
          continue
        else:
          self._reset()
          raise PolyphonicMelodyException()
      elif on_distance < 0:
        raise PolyphonicMelodyException(
            'Unexpected note. Not in ascending order.')

      # If a gap of `gap` or more steps is found, end the melody.
      if len(self) and off_distance >= gap_bars * steps_per_bar:
        break

      # Add the note-on and off events to the melody.
      self._add_note(note.pitch, start_index, end_index)

    if not self._events:
      # If no notes were added, don't set `start_step` and `end_step`.
      return

    self._start_step = offset

    # Strip final MELODY_NOTE_OFF event.
    if self._events[-1] == MELODY_NOTE_OFF:
      del self._events[-1]

    length = len(self)
    # Optionally round up `end_step` to a multiple of `steps_per_bar`.
    if pad_end:
      length += -len(self) % steps_per_bar
    self.set_length(length)

  def to_sequence(self,
                  velocity=100,
                  instrument=0,
                  program=0,
                  sequence_start_time=0.0,
                  qpm=120.0):
    """Converts the Melody to NoteSequence proto.

    The end of the melody is treated as a NOTE_OFF event for any sustained
    notes.

    Args:
      velocity: Midi velocity to give each note. Between 1 and 127 (inclusive).
      instrument: Midi instrument to give each note.
      program: Midi program to give each note.
      sequence_start_time: A time in seconds (float) that the first note in the
          sequence will land on.
      qpm: Quarter notes per minute (float).

    Returns:
      A NoteSequence proto encoding the given melody.
    """
    seconds_per_step = 60.0 / qpm / self.steps_per_quarter

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().qpm = qpm
    sequence.ticks_per_quarter = STANDARD_PPQ

    sequence_start_time += self.start_step * seconds_per_step
    current_sequence_note = None
    for step, note in enumerate(self):
      if MIN_MIDI_PITCH <= note <= MAX_MIDI_PITCH:
        # End any sustained notes.
        if current_sequence_note is not None:
          current_sequence_note.end_time = (
              step * seconds_per_step + sequence_start_time)

        # Add a note.
        current_sequence_note = sequence.notes.add()
        current_sequence_note.start_time = (
            step * seconds_per_step + sequence_start_time)
        current_sequence_note.pitch = note
        current_sequence_note.velocity = velocity
        current_sequence_note.instrument = instrument
        current_sequence_note.program = program

      elif note == MELODY_NOTE_OFF:
        # End any sustained notes.
        if current_sequence_note is not None:
          current_sequence_note.end_time = (
              step * seconds_per_step + sequence_start_time)
          current_sequence_note = None

    # End any sustained notes.
    if current_sequence_note is not None:
      current_sequence_note.end_time = (
          len(self) * seconds_per_step + sequence_start_time)

    if sequence.notes:
      sequence.total_time = sequence.notes[-1].end_time

    return sequence

  def transpose(self, transpose_amount, min_note=0, max_note=128):
    """Transpose notes in this Melody.

    All notes are transposed the specified amount. Additionally, all notes
    are octave shifted to lie within the [min_note, max_note) range.

    Args:
      transpose_amount: The number of half steps to transpose this Melody.
          Positive values transpose up. Negative values transpose down.
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
    """
    for i in range(len(self)):
      # Transpose MIDI pitches. Special events below MIN_MIDI_PITCH are not
      # changed.
      if self._events[i] >= MIN_MIDI_PITCH:
        self._events[i] += transpose_amount
        if self._events[i] < min_note:
          self._events[i] = (
              min_note + (self._events[i] - min_note) % NOTES_PER_OCTAVE)
        elif self._events[i] >= max_note:
          self._events[i] = (max_note - NOTES_PER_OCTAVE +
                             (self._events[i] - max_note) % NOTES_PER_OCTAVE)

  def squash(self, min_note, max_note, transpose_to_key):
    """Transpose and octave shift the notes in this Melody.

    The key center of this melody is computed with a heuristic, and the notes
    are transposed to be in the given key. The melody is also octave shifted
    to be centered in the given range. Additionally, all notes are octave
    shifted to lie within a given range.

    Args:
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
      transpose_to_key: The melody is transposed to be in this key. 0 = C Major.

    Returns:
      How much notes are transposed by.
    """
    melody_key = self.get_major_key()
    key_diff = transpose_to_key - melody_key
    midi_notes = [note for note in self._events
                  if MIN_MIDI_PITCH <= note <= MAX_MIDI_PITCH]
    if not midi_notes:
      return 0
    melody_min_note = min(midi_notes)
    melody_max_note = max(midi_notes)
    melody_center = (melody_min_note + melody_max_note) / 2
    target_center = (min_note + max_note - 1) / 2
    center_diff = target_center - (melody_center + key_diff)
    transpose_amount = (
        key_diff +
        NOTES_PER_OCTAVE * int(round(center_diff / float(NOTES_PER_OCTAVE))))
    self.transpose(transpose_amount, min_note, max_note)

    return transpose_amount

  def set_length(self, steps, from_left=False):
    """Sets the length of the melody to the specified number of steps.

    If the melody is not long enough, ends any sustained notes and adds NO_EVENT
    steps for padding. If it is too long, it will be truncated to the requested
    length.

    Args:
      steps: How many steps long the melody should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    old_len = len(self)
    super(Melody, self).set_length(steps, from_left=from_left)
    if steps > old_len and not from_left:
      # When extending the melody on the right, we end any sustained notes.
      for i in reversed(range(old_len)):
        if self._events[i] == MELODY_NOTE_OFF:
          break
        elif self._events[i] != MELODY_NO_EVENT:
          self._events[old_len] = MELODY_NOTE_OFF
          break

  def increase_resolution(self, k):
    """Increase the resolution of a Melody.

    Increases the resolution of a Melody object by a factor of `k`. This uses
    MELODY_NO_EVENT to extend each event in the melody to be `k` steps long.

    Args:
      k: An integer, the factor by which to increase the resolution of the
          melody.
    """
    super(Melody, self).increase_resolution(
        k, fill_event=MELODY_NO_EVENT)


def extract_melodies(quantized_sequence,
                     min_bars=7,
                     max_steps_truncate=None,
                     max_steps_discard=None,
                     gap_bars=1.0,
                     min_unique_pitches=5,
                     ignore_polyphonic_notes=True,
                     pad_end=False):
  """Extracts a list of melodies from the given QuantizedSequence object.

  This function will search through `quantized_sequence` for monophonic
  melodies in every track at every time step.

  Once a note-on event in a track is encountered, a melody begins.
  Gaps of silence in each track will be splitting points that divide the
  track into separate melodies. The minimum size of these gaps are given
  in `gap_bars`. The size of a bar (measure) of music in time steps is
  computed from the time signature stored in `quantized_sequence`.

  The melody is then checked for validity. The melody is only used if it is
  at least `min_bars` bars long, and has at least `min_unique_pitches` unique
  notes (preventing melodies that only repeat a few notes, such as those found
  in some accompaniment tracks, from being used).

  After scanning each instrument track in the QuantizedSequence, a list of all
  extracted Melody objects is returned.

  Args:
    quantized_sequence: A sequences_lib.QuantizedSequence object.
    min_bars: Minimum length of melodies in number of bars. Shorter melodies are
        discarded.
    max_steps_truncate: Maximum number of steps in extracted melodies. If
        defined, longer melodies are truncated to this threshold. If pad_end is
        also True, melodies will be truncated to the end of the last bar below
        this threshold.
    max_steps_discard: Maximum number of steps in extracted melodies. If
        defined, longer melodies are discarded.
    gap_bars: A melody comes to an end when this number of bars (measures) of
        silence is encountered.
    min_unique_pitches: Minimum number of unique notes with octave equivalence.
        Melodies with too few unique notes are discarded.
    ignore_polyphonic_notes: If True, melodies will be extracted from
        `quantized_sequence` tracks that contain polyphony (notes start at
        the same time). If False, tracks with polyphony will be ignored.
    pad_end: If True, the end of the melody will be padded with NO_EVENTs so
        that it will end at a bar boundary.

  Returns:
    melodies: A python list of Melody instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.

  Raises:
    NonIntegerStepsPerBarException: If `quantized_sequence`'s bar length
        (derived from its time signature) is not an integer number of time
        steps.
  """
  # TODO(danabo): Convert `ignore_polyphonic_notes` into a float which controls
  # the degree of polyphony that is acceptable.
  melodies = []
  stats = dict([(stat_name, statistics.Counter(stat_name)) for stat_name in
                ['polyphonic_tracks_discarded',
                 'melodies_discarded_too_short',
                 'melodies_discarded_too_few_pitches',
                 'melodies_discarded_too_long',
                 'melodies_truncated']])
  # Create a histogram measuring melody lengths (in bars not steps).
  # Capture melodies that are very small, in the range of the filter lower
  # bound `min_bars`, and large. The bucket intervals grow approximately
  # exponentially.
  stats['melody_lengths_in_bars'] = statistics.Histogram(
      'melody_lengths_in_bars',
      [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, min_bars // 2, min_bars,
       min_bars + 1, min_bars - 1])
  for track in quantized_sequence.tracks:
    start = 0

    # Quantize the track into a Melody object.
    # If any notes start at the same time, only one is kept.
    while 1:
      melody = Melody()
      try:
        melody.from_quantized_sequence(
            quantized_sequence,
            track=track,
            start_step=start,
            gap_bars=gap_bars,
            ignore_polyphonic_notes=ignore_polyphonic_notes,
            pad_end=pad_end)
      except PolyphonicMelodyException:
        stats['polyphonic_tracks_discarded'].increment()
        break  # Look for monophonic melodies in other tracks.
      except events_lib.NonIntegerStepsPerBarException:
        raise
      start = melody.end_step
      if not melody:
        break

      # Require a certain melody length.
      stats['melody_lengths_in_bars'].increment(
          len(melody) // melody.steps_per_bar)
      if len(melody) - 1 < melody.steps_per_bar * min_bars:
        stats['melodies_discarded_too_short'].increment()
        continue

      # Discard melodies that are too long.
      if max_steps_discard is not None and len(melody) > max_steps_discard:
        stats['melodies_discarded_too_long'].increment()
        continue

      # Truncate melodies that are too long.
      if max_steps_truncate is not None and len(melody) > max_steps_truncate:
        truncated_length = max_steps_truncate
        if pad_end:
          truncated_length -= max_steps_truncate % melody.steps_per_bar
        melody.set_length(truncated_length)
        stats['melodies_truncated'].increment()

      # Require a certain number of unique pitches.
      note_histogram = melody.get_note_histogram()
      unique_pitches = np.count_nonzero(note_histogram)
      if unique_pitches < min_unique_pitches:
        stats['melodies_discarded_too_few_pitches'].increment()
        continue

      # TODO(danabo)
      # Add filter for rhythmic diversity.

      melodies.append(melody)

  return melodies, stats.values()


def midi_file_to_melody(midi_file, steps_per_quarter=4, qpm=None,
                        ignore_polyphonic_notes=True):
  """Loads a melody from a MIDI file.

  Args:
    midi_file: Absolute path to MIDI file.
    steps_per_quarter: Quantization of Melody. For example, 4 = 16th notes.
    qpm: Tempo in quarters per a minute. If not set, tries to use the first
        tempo of the midi track and defaults to
        magenta.music.DEFAULT_QUARTERS_PER_MINUTE if fails.
    ignore_polyphonic_notes: Only use the highest simultaneous note if True.

  Returns:
    A Melody object extracted from the MIDI file.
  """
  sequence = midi_io.midi_file_to_sequence_proto(midi_file)
  if qpm is None:
    if sequence.tempos:
      qpm = sequence.tempos[0].qpm
    else:
      qpm = constants.DEFAULT_QUARTERS_PER_MINUTE
  quantized_sequence = sequences_lib.QuantizedSequence()
  quantized_sequence.qpm = qpm
  quantized_sequence.from_note_sequence(
      sequence, steps_per_quarter=steps_per_quarter)
  melody = Melody()
  melody.from_quantized_sequence(
      quantized_sequence, ignore_polyphonic_notes=ignore_polyphonic_notes)
  return melody


class MelodyEncoderDecoder(events_lib.EventsEncoderDecoder):
  """An abstract class for translating between melodies and model data.

  When building your dataset, the `encode` method takes in a melody and
  returns a SequenceExample of inputs and labels. These SequenceExamples are
  fed into the model during training and evaluation.

  During melody generation, the `get_inputs_batch` method takes in a list of
  the current melodies and returns an inputs batch which is fed into the
  model to predict what the next note should be for each melody.
  The `extend_event_sequences` method takes in the list of melodies and the
  softmax returned by the model and extends each melody by one step by sampling
  from the softmax probabilities. This loop (`get_inputs_batch` -> inputs batch
  is fed through the model to get a softmax -> `extend_event_sequences`) is
  repeated until the generated melodies have reached the desired length.

  Attributes:
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.
    transpose_to_key: The key that encoded melodies will be transposed into.

  Properties:
    input_size: The length of the list returned by self.events_to_input.
    num_classes: The range of ints that can be returned by
        self.events_to_label.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, min_note=48, max_note=84, transpose_to_key=0):
    """Initializes a MelodyEncoderDecoder object.

    You can change `min_note` and `max_note` to increase/decrease the melody
    range. Since melodies are transposed into this range to be run through
    the model and then transposed back into their original range after the
    melodies have been extended, the location of the range is somewhat
    arbitrary, but the size of the range determines the possible size of the
    generated melodies range. `transpose_to_key` should be set to the key
    that if melodies were transposed into that key, they would best sit
    between `min_note` and `max_note` with having as few notes outside that
    range. The same `min_note`, `max_note`, and `transpose_to_key` values
    should be used when creating your dataset, training your model,
    and generating melodies from it. If you change `min_note`, `max_note`,
    or `transpose_to_key`, you will have to recreate your dataset and retrain
    your model before you can accurately generate melodies from it.

    Args:
      min_note: The minimum midi pitch the encoded melodies can have.
      max_note: The maximum midi pitch (exclusive) the encoded melodies can
          have.
      transpose_to_key: The key that encoded melodies will be transposed into.

    Raises:
      ValueError: If `min_note` or `max_note` are outside the midi range, or
          if the [`min_note`, `max_note`) range is less than an octave. A range
          of at least an octave is required to be able to octave shift notes
          into that range while preserving their scale value.
    """
    if min_note < MIN_MIDI_PITCH:
      raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > MAX_MIDI_PITCH + 1:
      raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note - min_note < NOTES_PER_OCTAVE:
      raise ValueError('max_note - min_note must be >= 12. min_note is %d. '
                       'max_note is %d. max_note - min_note is %d.' %
                       (min_note, max_note, max_note - min_note))
    if transpose_to_key < 0 or transpose_to_key > NOTES_PER_OCTAVE - 1:
      raise ValueError('transpose_to_key must be >= 0 and <= 11. '
                       'transpose_to_key is %d.' % transpose_to_key)

    self._min_note = min_note
    self._max_note = max_note
    self._transpose_to_key = transpose_to_key

  @property
  def min_note(self):
    """The minimum midi pitch the encoded melodies can have.

    Returns:
      An integer, the minimum midi pitch the encoded melodies can have.
    """
    return self._min_note

  @property
  def max_note(self):
    """The maximum midi pitch (exclusive) the encoded melodies can have.

    Returns:
      An integer, the maximum midi pitch (exclusive) the encoded melodies can
      have.
    """
    return self._max_note

  @property
  def transpose_to_key(self):
    """The key that encoded melodies will be transposed into.

    Returns:
      An integer in the range [0, 12), the key that encoded melodies will be
      transposed into.
    """
    return self._transpose_to_key

  @property
  def no_event_label(self):
    """The class label that represents a NO_EVENT Melody event.

    Returns:
      An int, the class label that represents a NO_EVENT.
    """
    melody = Melody([MELODY_NO_EVENT])
    return self.events_to_label(melody, 0)

  @abc.abstractmethod
  def events_to_input(self, events, position):
    """Returns the input vector for the melody event at the given position.

    Args:
      events: A Melody object.
      position: An integer event position in the melody.

    Returns:
      An input vector, a self.input_size length list of floats.
    """
    pass

  @abc.abstractmethod
  def events_to_label(self, events, position):
    """Returns the label for the melody event at the given position.

    Args:
      events: A Melody object.
      position: An integer event position in the melody.

    Returns:
      A label, an integer in the range [0, self.num_classes).
    """
    pass

  @abc.abstractmethod
  def class_index_to_event(self, class_index, events):
    """Returns the melody event for the given class index.

    This is the reverse process of the self.events_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A Melody object.

    Returns:
      An integer melody event value.
    """
    pass

  def squash_and_encode(self, melody):
    """Returns a SequenceExample for the given melody after squashing.

    Args:
      melody: A Melody object.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    melody.squash(self._min_note, self._max_note, self._transpose_to_key)
    return self._encode(melody)


class OneHotMelodyEncoderDecoder(MelodyEncoderDecoder):
  """A MelodyEncoderDecoder that produces a one-hot encoding for the input."""

  def __init__(self, min_note, max_note, transpose_to_key):
    super(OneHotMelodyEncoderDecoder, self).__init__(min_note, max_note,
                                                     transpose_to_key)
    self._input_size = (self.max_note - self.min_note +
                        NUM_SPECIAL_MELODY_EVENTS)
    self._num_classes = (self.max_note - self.min_note +
                         NUM_SPECIAL_MELODY_EVENTS)

  @property
  def input_size(self):
    return self._input_size

  @property
  def num_classes(self):
    return self._num_classes

  def events_to_input(self, events, position):
    input_ = [0.0] * self._input_size
    index = (events[position] + NUM_SPECIAL_MELODY_EVENTS
             if events[position] < 0
             else events[position] - self.min_note + NUM_SPECIAL_MELODY_EVENTS)
    input_[index] = 1.0
    return input_

  def events_to_label(self, events, position):
    return (events[position] + NUM_SPECIAL_MELODY_EVENTS
            if events[position] < 0
            else events[position] - self.min_note + NUM_SPECIAL_MELODY_EVENTS)

  def class_index_to_event(self, class_index, events):
    return (class_index - NUM_SPECIAL_MELODY_EVENTS
            if class_index < NUM_SPECIAL_MELODY_EVENTS
            else class_index + self.min_note - NUM_SPECIAL_MELODY_EVENTS)
