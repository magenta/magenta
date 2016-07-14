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

Use Melody.extract_melodies to extract monophonic melodies from a NoteSequence
proto.

Use Melody.to_sequence to write a melody to a NoteSequence proto. Then use
midi_io.sequence_proto_to_midi_file to write that NoteSequence to a midi file.

Use MelodyEncoderDecoder.encode to convert a Melody object to a
tf.train.SequenceExample of inputs and labels. These SequenceExamples are fed
into the model during training and evaluation.

During melody generation, use MelodyEncoderDecoder.get_inputs_batch to convert
a list of melodies into an inputs batch which can be fed into the model to
predict what the next note should be for each melody. Then use
MelodyEncoderDecoder.extend_melodies to extend each of those melodies with an
event sampled from the softmax output by the model.
"""

import abc
import math

# internal imports
import numpy as np
import tensorflow as tf

from magenta.lib import sequence_example_lib
from magenta.protobuf import music_pb2


# Special events.
NUM_SPECIAL_EVENTS = 2
NOTE_OFF = -1
NO_EVENT = -2

# Other constants.
MIN_MIDI_PITCH = 0  # Inclusive.
MAX_MIDI_PITCH = 127  # Inclusive.
NOTES_PER_OCTAVE = 12
DEFAULT_BEATS_PER_MINUTE = 120.0
BEATS_PER_BAR = 4  # This code assumes 4 beats per measure of music.

# Standard pulses per quarter.
# https://en.wikipedia.org/wiki/Pulses_per_quarter_note
STANDARD_PPQ = 96

# Set the quantization cutoff.
# Note events before this cutoff are rounded down to nearest step. Notes
# above this cutoff are rounded up to nearest step. The cutoff is given as a
# fraction of a step.
# For example, with quantize_cutoff = 0.75 using 0-based indexing,
# if .75 < event <= 1.75, it will be quantized to step 1.
# If 1.75 < event <= 2.75 it will be quantized to step 2.
# A number close to 1.0 gives less wiggle room for notes that start early,
# and they will be snapped to the previous step.
QUANTIZE_CUTOFF = 0.75

# NOTE_KEYS[note] = The major keys that note belongs to.
# ex. NOTE_KEYS[0] lists all the major keys that contain the note C,
# which are:
# [0, 1, 3, 5, 7, 8, 10]
# [C, C#, D#, F, G, G#, A#]
#
# 0 = C
# 1 = C#
# 2 = D
# 3 = D#
# 4 = E
# 5 = F
# 6 = F#
# 7 = G
# 8 = G#
# 9 = A
# 10 = A#
# 11 = B
#
# NOTE_KEYS can be generated using the code below, but is explicitly declared
# for readability:
# scale = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
# NOTE_KEYS = [[j for j in xrange(12) if scale[(i - j) % 12]]
#              for i in xrange(12)]
NOTE_KEYS = [
    [0, 1, 3, 5, 7, 8, 10],
    [1, 2, 4, 6, 8, 9, 11],
    [0, 2, 3, 5, 7, 9, 10],
    [1, 3, 4, 6, 8, 10, 11],
    [0, 2, 4, 5, 7, 9, 11],
    [0, 1, 3, 5, 6, 8, 10],
    [1, 2, 4, 6, 7, 9, 11],
    [0, 2, 3, 5, 7, 8, 10],
    [1, 3, 4, 6, 8, 9, 11],
    [0, 2, 4, 5, 7, 9, 10],
    [1, 3, 5, 6, 8, 10, 11],
    [0, 2, 4, 6, 7, 9, 11]]


class PolyphonicMelodyException(Exception):
  pass


class BadNoteException(Exception):
  pass


class Melody(object):
  """Stores a quantized stream of monophonic melody events.

  Melody is an intermediate representation that all melody models
  can use. NoteSequence proto to melody code will do work to align notes
  and extract monophonic melodies. Model specific code just needs to
  convert Melody to SequenceExample protos for TensorFlow.

  Melody implements an iterable object. Simply iterate to retrieve
  the melody events.

  Melody events are integers in range [-2, 127] (inclusive),
  where negative values are the special event events: NOTE_OFF, and NO_EVENT.
  Non-negative values [0, 127] are note-on events for that midi pitch. A note
  starts at a non-negative value (that is the pitch), and is held through
  subsequent NO_EVENT events until either another non-negative value is reached
  (even if the pitch is the same as the previous note), or a NOTE_OFF event is
  reached. A NOTE_OFF starts at least one step of silence, which continues
  through NO_EVENT events until the next non-negative value.

  NO_EVENT values are treated as default filler. Notes must be inserted
  in ascending order by start time. Note end times will be truncated if the next
  note overlaps.

  Melodies can start at any non-negative time, and are shifted left so that
  the bar containing the first note-on event is the first bar.

  Attributes:
    events: A python list of melody events which are integers. Melody events are
        described above.
    offset: When quantizing notes, this is the offset between indices in
        `events` and time steps of incoming melody events. An offset is chosen
        such that the first melody event is close to the beginning of `events`.
    steps_per_bar: Number of steps in a bar (measure) of music.
    last_on: Index of last note-on event added. This index will be within
        the range of `events`.
    last_off: Index of the NOTE_OFF event that belongs to the note-on event
        at `last_on`. This index is likely not in the range of `events` unless
        _write_all_notes was called.
  """

  def __init__(self, steps_per_bar=16):
    """Construct an empty Melody.

    Args:
      steps_per_bar: How many time steps per bar of music. Melody needs to know
          about bars to skip empty bars before the first note.
    """
    self.events = []
    self.offset = 0
    self.steps_per_bar = steps_per_bar
    self.last_on = None  # Index of the last note-on event in `events`.
    # last_off is the index of the NOTE_OFF event belonging to the most
    # recently added note.
    self.last_off = None

  def __iter__(self):
    """Return an iterator over the events in this Melody.

    Returns:
      Python iterator over events.
    """
    return iter(self.events)

  def __len__(self):
    """How many events are in this Melody.

    Returns:
      Number of events as an int.
    """
    return len(self.events)

  def _can_add_note(self, start_step):
    """Check if a note-on event can be added at the given time step.

    Note-on events cannot be added at the same time as or before previous
    note-on events.

    Args:
      start_step: Time step of the note-on that we wish to add. This is
          a non-negative integer.

    Returns:
      True if `start_step` is after all previous note-on events.
    """
    return self.last_on is None or start_step - self.offset > self.last_on

  def _add_note(self, pitch, start_step, end_step):
    """Adds the given note to the stream.

    The previous note's end step will be changed to end before this note if
    there is overlap.

    The note is not added if `start_step` is before the start step of the
    previously added note, or if `start_step` equals `end_step`.

    Args:
      pitch: Midi pitch. An integer between 0 and 127 inclusive.
      start_step: A non-negative integer step that the note begins on.
      end_step: An integer step that the note ends on. The note is considered to
          end at the onset of the end step. `end_step` must be greater than
          `start_step`.
    """
    if not self._can_add_note(start_step):
      raise BadNoteException(
          'Given start step %d is before last on event at %d'
          % (start_step, self.last_on))

    if start_step == end_step:
      raise BadNoteException('Given start step and end step are the same: %d'
                             % start_step)

    if not self.events:
      self.offset = start_step - start_step % self.steps_per_bar

    start_step_offset = start_step - self.offset
    end_step_offset = end_step - self.offset
    self.events += [NO_EVENT] * (start_step_offset - len(self.events) + 1)
    self.events[start_step_offset] = pitch
    if self.last_off is not None and self.last_off < start_step_offset:
      self.events[self.last_off] = NOTE_OFF
    self.last_on = start_step_offset
    self.last_off = end_step_offset

  def _write_all_notes(self):
    """Write remaining note off event to `events`.

    This internal method makes sure that all notes being temporarily stored in
    other instance variables, namely self.last_on and self.last_off, are
    written to self.events. __iter__ and __len__ will only return what is in
    self.events, so all notes must be written there after operating on the
    events in this instance.
    """
    self.events += [NO_EVENT] * (self.last_off - len(self.events) + 1)
    self.events[self.last_off] = NOTE_OFF
    self.last_on = None
    self.last_off = None

  def _clear(self):
    """Clear `events` and last note-on/off information."""
    self.events = []
    self.last_on = None
    self.last_off = None

  def _distance_to_last_event(self, step):
    """Returns distance of the given step to the last off event.

    Args:
      step: Step to compute the distance to.

    Returns:
      Distance between step and last off event. 0 if events are the same.
      Negative if step comes before the last off event.

    Raises:
      ValueError: When the stream is empty.
    """
    if self.last_off is None:
      raise ValueError('No events in the stream')
    return step - self.offset - self.last_off

  def get_note_histogram(self):
    """Gets a histogram of the note occurrences in a melody.

    Returns:
      A list of 12 ints, one for each note value (C at index 0 through B at
      index 11). Each int is the total number of times that note occurred in
      the melody.
    """
    np_melody = np.array(self.events, dtype=int)
    return np.bincount(
        np_melody[np_melody >= MIN_MIDI_PITCH] % NOTES_PER_OCTAVE,
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

  def from_notes(self, notes, bpm=120.0, gap=16, ignore_polyphonic_notes=False):
    """Populate self with an iterable of music_pb2.NoteSequence.Note.

    BEATS_PER_BAR/4 time signature is assumed.

    The given list of notes is quantized according to the given beats per minute
    and populated into self. Any existing notes in the instance are cleared.

    0 velocity notes are ignored. The melody is ended when there is a gap of
    `gap` steps or more after a note.

    If note-on events occur at the same step, this melody is cleared and an
    exception is thrown.

    Args:
      notes: Iterable of music_pb2.NoteSequence.Note
      bpm: Beats per minute. This determines the quantization step size in
          seconds. Beats are subdivided according to `steps_per_bar` given to
          the constructor.
      gap: If this many steps or more follow a note, the melody is ended.
      ignore_polyphonic_notes: If False, and multiple notes occur within the
          same melody step, a PolyphonicMelodyException will be raised. If true,
          only one of the notes occurring in the same step will be used. The
          note used will be the note with the earlier NoteSequence `start_time`.
          If both notes have the same `start_time`, the note with the higher
          pitch is used, since melodies usually follow the top voicing when
          played alongside chords.

    Raises:
      PolyphonicMelodyException: If any of the notes start on the same step when
      quantized and ignore_polyphonic_notes is False.
    """
    self._clear()

    # Compute quantization steps per second.
    steps_per_second = bpm / 60.0 * self.steps_per_bar / BEATS_PER_BAR

    quantize = lambda x: int(math.ceil(x - QUANTIZE_CUTOFF))

    # Sort track by note start times, and secondarily by pitch descending.
    notes.sort(key=lambda note: (note.start_time, -note.pitch))
    for note in notes:
      # Ignore 0 velocity notes.
      if not note.velocity:
        continue

      # Quantize the start and end times of the note.
      start_step = quantize(note.start_time * steps_per_second)
      end_step = quantize(note.end_time * steps_per_second)
      if end_step <= start_step:
        end_step = start_step + 1

      # Do not allow notes to start in negative time.
      if start_step < 0:
        raise BadNoteException(
            'Got negative note time: start_time = %s, end_time = %s'
            % (note.start_time, note.end_time))

      # If start_step comes before or lands on an already added note's start
      # step, we cannot add it. Discard the melody because it is not monophonic.
      if not self._can_add_note(start_step):
        if ignore_polyphonic_notes:
          continue
        else:
          self._clear()
          raise PolyphonicMelodyException()

      # If a gap of `gap` or more steps is found, end the melody.
      if (len(self) and
          self._distance_to_last_event(start_step) >= gap):
        break

      # Add the note-on and off events to the melody.
      self._add_note(note.pitch, start_step, end_step)

    self._write_all_notes()

  def from_event_list(self, events):
    """Populate self with a list of event values."""
    self.events = list(events)

  def to_sequence(self, velocity=100, instrument=0, sequence_start_time=0.0,
                  bpm=120.0):
    """Converts the Melody to Sequence proto.

    Args:
      velocity: Midi velocity to give each note. Between 1 and 127 (inclusive).
      instrument: Midi instrument to give each note.
      sequence_start_time: A time in seconds (float) that the first note in the
          sequence will land on.
      bpm: Beats per minute (float).

    Returns:
      A NoteSequence proto encoding the given melody.
    """
    seconds_per_step = 60.0 / bpm * BEATS_PER_BAR / self.steps_per_bar

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().bpm = bpm
    sequence.ticks_per_beat = STANDARD_PPQ

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
        # Give the note an end time now just to be sure it gets closed.
        current_sequence_note.end_time = (
            (step + 1) * seconds_per_step + sequence_start_time)
        current_sequence_note.pitch = note
        current_sequence_note.velocity = velocity
        current_sequence_note.instrument = instrument

      elif note == NOTE_OFF:
        # End any sustained notes.
        if current_sequence_note is not None:
          current_sequence_note.end_time = (
              step * seconds_per_step + sequence_start_time)
          current_sequence_note = None

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
    for i in xrange(len(self.events)):
      # Transpose MIDI pitches. Special events below MIN_MIDI_PITCH are not
      # changed.
      if self.events[i] >= MIN_MIDI_PITCH:
        self.events[i] += transpose_amount
        if self.events[i] < min_note:
          self.events[i] = (
              min_note + (self.events[i] - min_note) % NOTES_PER_OCTAVE)
        elif self.events[i] >= max_note:
          self.events[i] = (max_note - NOTES_PER_OCTAVE +
                            (self.events[i] - max_note) % NOTES_PER_OCTAVE)

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
    midi_notes = [note for note in self.events
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


def extract_melodies(sequence, steps_per_beat=4, min_bars=7,
                     min_unique_pitches=5):
  """Extracts a list of melodies from the given NoteSequence proto.

  A time signature of BEATS_PER_BAR is assumed for each sequence. If the
  sequence has an incompatible time signature, like 3/4, 5/4, etc, then
  the time signature is ignored and BEATS_PER_BAR/4 time is assumed.

  Once a note-on event in a track is encountered, a melody begins. Once a
  gap of silence since the last note-off event of a bar length or more is
  encountered, or the end of the track is reached, that melody is ended. Only
  the first melody of each track is used (this reduces the number of repeated
  melodies that may come from repeated choruses or verses, but may also cause
  unique non-first melodies, such as bridges and outros, to be missed, so maybe
  this should be changed).

  The melody is then checked for validity. The melody is only used if it is
  at least `min_bars` bars long, and has at least `min_unique_pitches` unique
  notes (preventing melodies that only repeat a few notes, such as those found
  in some accompaniment tracks, from being used).

  After scanning each instrument track in the NoteSequence, a list of all the
  valid melodies is returned.

  Args:
    sequence: A NoteSequence proto containing notes.
    steps_per_beat: How many subdivisions of each beat. BEATS_PER_BAR/4 time is
        assumed, so steps per bar is equal to
        `BEATS_PER_BAR` * `steps_per_beat`.
    min_bars: Minimum length of melodies in number of bars. Shorter melodies are
        discarded.
    min_unique_pitches: Minimum number of unique notes with octave equivalence.
        Melodies with too few unique notes are discarded.

  Returns:
    A python list of Melody instances.
  """

  # Assume bars contain 4 beats, or quarter notes.
  steps_per_bar = steps_per_beat * 4

  # Beats per minute is stored in the tempo change event. If there is no tempo
  # then assume 120 bpm per the MIDI standard.
  bpm = (sequence.tempos[0].bpm if len(sequence.tempos)
         else DEFAULT_BEATS_PER_MINUTE)

  # Group note messages into tracks.
  tracks = {}
  for note in sequence.notes:
    if note.instrument not in tracks:
      tracks[note.instrument] = []
    tracks[note.instrument].append(note)

  melodies = []
  for track in tracks.values():
    melody = Melody(steps_per_bar)

    # Quantize the track into a Melody object.
    # If any notes start at the same time, only one is kept.
    melody.from_notes(track, bpm=bpm, gap=steps_per_bar,
                      ignore_polyphonic_notes=True)

    # Require a certain melody length.
    if len(melody) - 1 < steps_per_bar * min_bars:
      tf.logging.debug('melody too short')
      continue

    # Require a certain number of unique pitches.
    note_histogram = melody.get_note_histogram()
    unique_pitches = np.count_nonzero(note_histogram)
    if unique_pitches < min_unique_pitches:
      tf.logging.debug('melody too simple')
      continue

    melodies.append(melody)

  return melodies


class MelodyEncoderDecoder(object):
  """An abstract class for translating between melodies and model data.

  When building your dataset, the `encode` method takes in a melody and
  returns a SequenceExample of inputs and labels. These SequenceExamples are
  fed into the model during training and evaluation.

  During melody generation, the `get_inputs_batch` method takes in a list of
  the current melodies and returns an inputs batch which is fed into the
  model to predict what the next note should be for each melody.
  The `extend_melodies` method takes in the list of melodies and the softmax
  returned by the model and extends each melody by one step by sampling from
  the softmax probabilities. This loop (`get_inputs_batch` -> inputs batch
  is fed through the model to get a softmax -> `extend_melodies`) is repeated
  until the generated melodies have reached the desired length.

  The `melody_to_input`, `melody_to_label`, and `class_index_to_melody_event`
  methods must be overwritten to be specific to your model. See
  basic_rnn/basic_rnn_encoder_decoder.py for an example of this.
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
      max_note: The maximum midi pitch the encoded melodies can have.
      transpose_to_key: The key that encoded melodies will be transposed into.

    Attributes:
      min_note: The minimum midi pitch the encoded melodies can have.
      max_note: The maximum midi pitch the encoded melodies can have.
      transpose_to_key: The key that encoded melodies will be transposed into.

    Properties:
      input_size: The length of the list returned by self.melody_to_input.
      num_classes: The range of ints that can be returned by
          self.melody_to_label.

    Raises:
      ValueError: If `min_note` or `max_note` are outside the midi range, or
      if the [`min_note`, `max_note`) range is less than an octave. A range
      of at least an octave is required to be able to octave shift notes into
      that range while preserving their scale value.
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

    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key

  @abc.abstractproperty
  def input_size(self):
    """The size of the input vector used by this model.

    Returns:
        An int, the length of the list returned by self.melody_to_input.
    """
    pass

  @abc.abstractproperty
  def num_classes(self):
    """The range of labels used by this model.

    Returns:
        An int, the range of ints that can be returned by self.melody_to_label.
    """
    pass

  @abc.abstractmethod
  def melody_to_input(self, melody):
    """Returns the input vector for the last event in the melody.

    Args:
      melody: A Melody object.

    Returns:
      An input vector, a self.input_size length list of floats.
    """
    pass

  @abc.abstractmethod
  def melody_to_label(self, melody):
    """Returns the label for the last event in the melody.

    Args:
      melody: A Melody object.

    Returns:
      A label, an int in the range [0, self.num_classes).
    """
    pass

  def encode(self, melody):
    """Returns a SequenceExample for the given melody.

    Args:
      melody: A Melody object.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    melody.squash(self.min_note, self.max_note, self.transpose_to_key)
    inputs = []
    labels = []
    melody_events = melody.events
    melody.events = melody_events[:1]
    for i in xrange(1, len(melody_events)):
      inputs.append(self.melody_to_input(melody))
      melody.events = melody_events[:i + 1]
      labels.append(self.melody_to_label(melody))
    return sequence_example_lib.make_sequence_example(inputs, labels)

  def get_inputs_batch(self, melodies, full_length=False):
    """Returns an inputs batch for the given melodies.

    Args:
      melodies: A list of Melody objects.
      full_length: If True, the inputs batch will be for the full length of
          each melody. If False, the inputs batch will only be for the last
          event of each melody. A full-length inputs batch is used for the
          first step of extending the melodies, since the rnn cell state needs
          to be initialized with the priming melody. For subsequent generation
          steps, only a last-event inputs batch is used.

    Returns:
      An inputs batch. If `full_length` is True, the shape will be
      [len(melodies), len(melodies[0]), INPUT_SIZE]. If `full_length` is False,
      the shape will be [len(melodies), 1, INPUT_SIZE].
    """
    inputs_batch = []
    for melody in melodies:
      inputs = []
      if full_length and len(melody):
        melody_events = melody.events
        for i in xrange(len(melody_events)):
          melody.events = melody_events[:i + 1]
          inputs.append(self.melody_to_input(melody))
      else:
        inputs.append(self.melody_to_input(melody))
      inputs_batch.append(inputs)
    return inputs_batch

  @abc.abstractmethod
  def class_index_to_melody_event(self, class_index, melody):
    """Returns the melody event for the given class index.

    This is the reverse process of the self.melody_to_label method.

    Args:
      class_index: An int in the range [0, self.num_classes).
      melody: A Melody object. This object is not used in this implementation,
          but see models/lookback_rnn/lookback_rnn_encoder_decoder.py for an
          example of how this object can be used.

    Returns:
      A Melody event value, an int in the range [-2, 127]. -2 = no event,
      -1 = note-off event, [0, 127] = note-on event for that midi pitch.
    """
    pass

  def extend_melodies(self, melodies, softmax):
    """Extends the melodies by sampling from the softmax probabilities.

    Args:
      melodies: A list of Melody objects.
      softmax: A list of softmax probability vectors. The list of softmaxes
          should be the same length as the list of melodies.
    """
    num_classes = len(softmax[0][0])
    for i in xrange(len(melodies)):
      chosen_class = np.random.choice(num_classes, p=softmax[i][-1])
      melody_event = self.class_index_to_melody_event(chosen_class, melodies[i])
      melodies[i].events.append(melody_event)
