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
"""Utility functions for working with chord progressions.

Use extract_chords_for_melodies to extract chord progressions from a
QuantizedSequence object, aligned with already-extracted melodies.

Use ChordProgression.to_sequence to write a chord progression to a
NoteSequence proto, encoding the chords as text annotations.
"""

import abc
import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin

from magenta.lib import chord_symbols_lib
from magenta.lib import melodies_lib
from magenta.lib import sequence_example_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


# Constants.
QUARTER_NOTES_PER_WHOLE_NOTE = 4.0
DEFAULT_QUARTERS_PER_MINUTE = 120.0
DEFAULT_STEPS_PER_BAR = 16  # 4/4 music sampled at 4 steps per quarter note.
DEFAULT_STEPS_PER_QUARTER = 4

NOTES_PER_OCTAVE = melodies_lib.NOTES_PER_OCTAVE

# Standard pulses per quarter.
# https://en.wikipedia.org/wiki/Pulses_per_quarter_note
STANDARD_PPQ = 96

# Chord symbol for "no chord".
NO_CHORD = 'N.C.'

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class NonIntegerStepsPerBarException(Exception):
  pass


class CoincidentChordsException(Exception):
  pass


class BadChordException(Exception):
  pass


class ChordEncodingException(Exception):
  pass


class ChordProgression(object):
  """Stores a quantized stream of chord events.

  ChordProgression is an intermediate representation that all chord or lead
  sheet models can use. Chords are represented here by a chord symbol string;
  model-specific code is responsible for converting this representation to
  SequenceExample protos for TensorFlow.

  ChordProgression implements an iterable object. Simply iterate to retrieve
  the chord events.

  ChordProgression events are chord symbol strings like "Cm7", with special
  event NO_CHORD to indicate no chordal harmony. When a chord lasts for longer
  than a single step, the chord symbol event is repeated multiple times. Note
  that this is different from MonophonicMelody, where the special NO_EVENT is
  used for subsequent steps of sustained notes; in the case of harmony, there's
  no distinction between a repeated chord and a sustained chord.

  Chords must be inserted in ascending order by start time.

  Attributes:
    events: A python list of chord events which are strings.  ChordProgression
        events are described above.
    start_step: The offset of the first step of the progression relative to the
        beginning of the source sequence. Will always be the first step of a
        bar.
    end_step: The offset to the beginning of the bar following the last step
       of the progression relative tothe beginning of the source sequence. Will
       always be the first step of a bar.
    steps_per_quarter: Number of steps in in a quarter note.
    steps_per_bar: Number of steps in a bar (measure) of music.
  """

  def __init__(self):
    """Construct an empty ChordProgression."""
    self._reset()

  def _reset(self):
    """Clear `events` and reset object state."""
    self._events = []
    self._steps_per_bar = DEFAULT_STEPS_PER_BAR
    self._steps_per_quarter = DEFAULT_STEPS_PER_QUARTER
    self._start_step = 0
    self._end_step = 0

  def __iter__(self):
    """Return an iterator over the events in this ChordProgression.

    Returns:
      Python iterator over events.
    """
    return iter(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __getslice__(self, i, j):
    """Returns the events in the given slice range."""
    return self._events[i:j]

  def __len__(self):
    """How many events are in this ChordProgression.

    Returns:
      Number of events as an int.
    """
    return len(self._events)

  def __eq__(self, other):
    if not isinstance(other, ChordProgression):
      return False
    return (list(self) == list(other) and
            self.steps_per_bar == other.steps_per_bar and
            self.start_step == other.start_step and
            self.end_step == other.end_step)

  def _add_chord(self, figure, start_step, end_step):
    """Adds the given chord to the `events` list.

    `start_step` is set to the given chord. Everything after `start_step` in
    `events` is deleted before the chord is added. `events`'s length will be
     changed so that the last event has index `end_step` - 1.

    Args:
      figure: Chord symbol figure. A string like "Cm9" representing the chord.
      start_step: A non-negative integer step that the chord begins on.
      end_step: An integer step that the chord ends on. The chord is considered
          to end at the onset of the end step. `end_step` must be greater than
          `start_step`.

    Raises:
      BadChordException: If `start_step` does not precede `end_step`.
    """
    if start_step >= end_step:
      raise BadChordException(
          'Start step does not precede end step: start=%d, end=%d' %
          (start_step, end_step))

    self.set_length(end_step)

    for i in range(start_step, end_step):
      self._events[i] = figure

  @property
  def start_step(self):
    return self._start_step

  @property
  def end_step(self):
    return self._end_step

  @property
  def steps_per_bar(self):
    return self._steps_per_bar

  @property
  def steps_per_quarter(self):
    return self._steps_per_quarter

  def append_event(self, event):
    """Appends event to the end of the progression and increments the end step.

    Args:
      event: The integer ChordProgression event to append to the end.
    """
    self._events.append(event)
    self._end_step += 1

  def from_quantized_sequence(self, quantized_sequence, start_step, end_step):
    """Populate self with the chords from the given QuantizedSequence object.

    A chord progression is extracted from the given sequence starting at time
    step `start_step` and ending at time step `end_step`.

    The number of time steps per bar is computed from the time signature in
    `quantized_sequence`.

    Args:
      quantized_sequence: A sequences_lib.QuantizedSequence instance.
      start_step: Start populating chords at this time step.
      end_step: Stop populating chords at this time step.

    Raises:
      NonIntegerStepsPerBarException: If `quantized_sequence`'s bar length
          (derived from its time signature) is not an integer number of time
          steps.
      CoincidentChordsException: If any of the chords start on the same step.
    """
    self._reset()

    steps_per_bar_float = quantized_sequence.steps_per_bar()
    if steps_per_bar_float % 1 != 0:
      raise NonIntegerStepsPerBarException(
          'There are %f timesteps per bar. Time signature: %d/%d' %
          (steps_per_bar_float, quantized_sequence.time_signature.numerator,
           quantized_sequence.time_signature.denominator))
    self._steps_per_bar = int(steps_per_bar_float)
    self._steps_per_quarter = quantized_sequence.steps_per_quarter

    # Sort track by chord times.
    chords = sorted(quantized_sequence.chords, key=lambda chord: chord.step)

    prev_step = None
    prev_figure = NO_CHORD

    for chord in chords:
      if chord.step >= end_step:
        # No more chords within range.
        break

      elif chord.step < start_step:
        # Chord is before start of range.
        prev_step = chord.step
        prev_figure = chord.figure
        continue

      if chord.step == prev_step:
        if chord.figure == prev_figure:
          # Identical coincident chords, just skip.
          continue
        else:
          # Two different chords start at the same time step.
          self._reset()
          raise CoincidentChordsException('chords %s and %s are coincident' %
                                          (prev_figure, chord.figure))

      if chord.step > start_step:
        # Add the previous chord.
        start_index = max(prev_step, start_step) - start_step
        end_index = chord.step - start_step
        self._add_chord(prev_figure, start_index, end_index)

      prev_step = chord.step
      prev_figure = chord.figure

    if prev_step < end_step:
      # Add the last chord active before end_step.
      start_index = max(prev_step, start_step) - start_step
      end_index = end_step - start_step
      self._add_chord(prev_figure, start_index, end_index)

    self._start_step = start_step
    self._end_step = end_step

  def from_event_list(self, events, start_step=0,
                      steps_per_bar=DEFAULT_STEPS_PER_BAR,
                      steps_per_quarter=DEFAULT_STEPS_PER_QUARTER):
    """Initialies with a list of event values and sets attributes appropriately.

    Args:
      events: List of ChordProgression events.
      start_step: The integer starting step offset.
      steps_per_bar: The number of steps in a bar.
      steps_per_quarter: The number of steps in a quarter note.
    """
    self._events = list(events)
    self._start_step = start_step
    self._end_step = start_step + len(self)
    self._steps_per_bar = steps_per_bar
    self._steps_per_quarter = steps_per_quarter

  def to_sequence(self,
                  sequence_start_time=0.0,
                  qpm=120.0):
    """Converts the ChordProgression to NoteSequence proto.

    This doesn't generate actual notes, but text annotations specifying the
    chord changes when they occur.

    Args:
      sequence_start_time: A time in seconds (float) that the first chord in
          the sequence will land on.
      qpm: Quarter notes per minute (float).

    Returns:
      A NoteSequence proto encoding the given chords as text annotations.
    """
    seconds_per_step = 60.0 / qpm / self.steps_per_quarter

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().qpm = qpm
    sequence.ticks_per_quarter = STANDARD_PPQ

    current_figure = NO_CHORD
    for step, figure in enumerate(self):
      if figure != current_figure:
        current_figure = figure
        chord = sequence.text_annotations.add()
        chord.time = step * seconds_per_step + sequence_start_time
        chord.text = figure
        chord.annotation_type = CHORD_SYMBOL

    return sequence

  def transpose(self, transpose_amount, chord_symbol_functions=
                chord_symbols_lib.ChordSymbolFunctions.get()):
    """Transpose chords in this ChordProgression.

    Args:
      transpose_amount: The number of half steps to transpose this
          ChordProgression. Positive values transpose up. Negative values
          transpose down.
      chord_symbol_functions: ChordSymbolFunctions object with which to perform
          the actual transposition of chord symbol strings.

    Raises:
      ChordSymbolException: If a chord (other than "no chord") fails to be
          interpreted by the ChordSymbolFunctions object.
    """
    for i in xrange(len(self._events)):
      if self._events[i] != NO_CHORD:
        self._events[i] = chord_symbol_functions.transpose_chord_symbol(
            self._events[i], transpose_amount % NOTES_PER_OCTAVE)

  def set_length(self, steps, from_left=False):
    """Sets the length of the progression to the specified number of steps.

    If the chord progression is not long enough, ends any sustained notes and
    adds NO_CHORD steps for padding. If it is too long, it will be truncated to
    the requested length.

    Args:
      steps: How many steps long the chord progression should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if steps > len(self):
      if from_left:
        self._events[:0] = [NO_CHORD] * (steps - len(self))
      else:
        self._events.extend([NO_CHORD] * (steps - len(self)))
    else:
      if from_left:
        del self._events[0:-steps]
      else:
        del self._events[steps:]

    if from_left:
      self._start_step = self._end_step - steps
    else:
      self._end_step = self._start_step + steps


def extract_chords_for_melodies(quantized_sequence, melodies):
  """Extracts from the QuantizedSequence a chord progression for each melody.

  This function will extract the underlying chord progression (encoded as text
  annotations) from `quantized_sequence` for each monophonic melody in
  `melodies`.  Each chord progression will be the same length as its
  corresponding melody.

  Args:
    quantized_sequence: A sequences_lib.QuantizedSequence object.
    melodies: A python list of MonophonicMelody instances.

  Returns:
    A python list of ChordProgression instances, the same length as `melodies`.
        If a progression fails to be extracted for a melody, the corresponding
        list entry will be None.
  """
  chord_progressions = []
  stats = dict([('coincident_chords', statistics.Counter('coincident_chords'))])
  for melody in melodies:
    try:
      chords = ChordProgression()
      chords.from_quantized_sequence(
          quantized_sequence, melody.start_step, melody.end_step)
    except CoincidentChordsException:
      stats['coincident_chords'].increment()
      chords = None
    chord_progressions.append(chords)

  return chord_progressions, stats.values()


class ChordRenderer(object):
  """An abstract class for rendering NoteSequence chord symbols as notes."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def render(self, sequence):
    """Renders the chord symbols of a NoteSequence.

    This function renders chord symbol annotations in a NoteSequence as actual
    notes. Notes are added to the NoteSequence object, and the chord symbols
    remain also.

    Args:
      sequence: The NoteSequence for which to render chord symbols.
    """
    pass


class BasicChordRenderer(ChordRenderer):
  """A chord renderer that holds each note for the duration of the chord."""

  def __init__(self,
               velocity=100,
               instrument=1,
               program=88,
               chord_symbol_functions=
               chord_symbols_lib.ChordSymbolFunctions.get()):
    """Initialize a BasicChordRenderer object.

    Args:
      velocity: The MIDI note velocity to use.
      instrument: The MIDI instrument to use.
      program: The MIDI program to use.
      chord_symbol_functions: ChordSymbolFunctions object with which to perform
          the actual transposition of chord symbol strings.
    """
    self._velocity = velocity
    self._instrument = instrument
    self._program = program
    self._chord_symbol_functions = chord_symbol_functions

  def _render_notes(self, sequence, pitches, start_time, end_time):
    for pitch in pitches:
      # Add a note.
      note = sequence.notes.add()
      note.start_time = start_time
      note.end_time = end_time
      note.pitch = pitch
      note.velocity = self._velocity
      note.instrument = self._instrument
      note.program = self._program

  def render(self, sequence):
    # Sort text annotations by time.
    annotations = sorted(sequence.text_annotations, key=lambda a: a.time)

    prev_time = 0.0
    prev_figure = NO_CHORD

    for annotation in annotations:
      if annotation.time >= sequence.total_time:
        break

      if annotation.annotation_type == CHORD_SYMBOL:
        if prev_figure != NO_CHORD:
          # Render the previous chord.
          pitches = self._chord_symbol_functions.chord_symbol_midi_pitches(
              prev_figure)
          self._render_notes(sequence=sequence,
                             pitches=pitches,
                             start_time=prev_time,
                             end_time=annotation.time)

        prev_time = annotation.time
        prev_figure = annotation.text

    if (prev_time < sequence.total_time and
        prev_figure != NO_CHORD):
      # Render the last chord.
      pitches = self._chord_symbol_functions.chord_symbol_midi_pitches(
          prev_figure)
      self._render_notes(sequence=sequence,
                         pitches=pitches,
                         start_time=prev_time,
                         end_time=sequence.total_time)


class SingleChordEncoderDecoder(object):
  """An abstract class for encoding and decoding individual chords.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def num_classes(self):
    """The number of distinct chord encodings.

    Returns:
        An int, the range of ints that can be returned by self.encode_chord.
    """
    pass

  @abc.abstractmethod
  def encode_chord(self, figure):
    """Convert from a chord symbol string to a chord encoding int.

    Args:
      figure: A chord symbol string representing the chord.

    Returns:
        An integer representing the encoded chord, in range
            [0, self.num_classes).
    """
    pass

  @abc.abstractmethod
  def decode_chord(self, index):
    """Convert from a chord encoding integer to a chord symbol string.

    Args:
      index: The encoded chord, an integer in the range [0, self.num_classes).

    Returns:
        A chord symbol string representing the decoded chord.
    """
    pass


class MajorMinorEncoderDecoder(SingleChordEncoderDecoder):
  """Encodes chords as root + major/minor, with zero index for "no chord".

  Encodes chords as follows:
    0:     "no chord"
    1-12:  chords with a major triad, where 1 is C major, 2 is C# major, etc.
    13-24: chords with a minor triad, where 13 is C minor, 14 is C# minor, etc.
  """

  # Mapping from pitch class index to name.  Eventually this should be defined
  # more globally, but right now only `decode_chord` needs it.
  _PITCH_CLASS_MAPPING = ['C', 'C#', 'D', 'E-', 'E', 'F',
                          'F#', 'G', 'A-', 'A', 'B-', 'B']

  def __init__(self, chord_symbol_functions=
               chord_symbols_lib.ChordSymbolFunctions.get()):
    """Initialize the MajorMinorEncoderDecoder object.

    Args:
      chord_symbol_functions: ChordSymbolFunctions object with which to perform
          the actual transposition of chord symbol strings.
    """
    self._chord_symbol_functions = chord_symbol_functions

  @property
  def num_classes(self):
    return 2 * NOTES_PER_OCTAVE + 1

  def encode_chord(self, figure):
    if figure == NO_CHORD:
      return 0

    root = self._chord_symbol_functions.chord_symbol_root(figure)
    quality = self._chord_symbol_functions.chord_symbol_quality(figure)

    if quality == chord_symbols_lib.CHORD_QUALITY_MAJOR:
      return root + 1
    elif quality == chord_symbols_lib.CHORD_QUALITY_MINOR:
      return root + NOTES_PER_OCTAVE + 1
    else:
      raise ChordEncodingException('chord is neither major nor minor: %s'
                                   % figure)

  def decode_chord(self, index):
    if index == 0:
      return NO_CHORD
    elif index - 1 < 12:
      # major
      return self._PITCH_CLASS_MAPPING[index - 1]
    else:
      # minor
      return self._PITCH_CLASS_MAPPING[index - NOTES_PER_OCTAVE - 1] + 'm'


class ChordsEncoderDecoder(object):
  """An abstract class for translating between chords and model data.

  When building your dataset, the `encode` method takes in a chord progression
  and returns a SequenceExample of inputs and labels. These SequenceExamples
  are fed into the model during training and evaluation.

  During generation, the `get_inputs_batch` method takes in a list of the
  current chord progressions and returns an inputs batch which is fed into the
  model to predict what the next chord should be for each progression. The
  `extend_chord_progressions` method takes in the list of chord progressions
  and the softmax returned by the model and extends each progression by one
  step by sampling from the softmax probabilities. This loop
  (`get_inputs_batch` -> inputs batch is fed through the model to get a
  softmax -> `extend_chord_progressions`) is repeated until the generated
  chord progressions have reached the desired length.

  The `chord_to_input`, `chord_to_label`, and `class_index_to_chord_event`
  methods must be overwritten to be specific to your model.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    """Initializes a ChordsEncoderDecoder object.

    Unlike the MelodyEncoderDecoder, this object does not detect and control
    key for the chord progressions.  Instead, when encoding each chord
    progression a transpose amount must be passed.

    Properties:
      input_size: The length of the list returned by self.chords_to_input.
      num_classes: The range of ints that can be returned by
          self.chords_to_label.
    """
    pass

  @abc.abstractproperty
  def input_size(self):
    """The size of the input vector used by this model.

    Returns:
        An int, the length of the list returned by self.chords_to_input.
    """
    pass

  @abc.abstractproperty
  def num_classes(self):
    """The range of labels used by this model.

    Returns:
        An int, the range of ints that can be returned by self.chords_to_label.
    """
    pass

  @abc.abstractmethod
  def chords_to_input(self, chords, position):
    """Returns the input vector for the chord event at the given position.

    Args:
      chords: A ChordProgression object.
      position: An integer event position in the chord progression.

    Returns:
      An input vector, a self.input_size length list of floats.
    """
    pass

  @abc.abstractmethod
  def chords_to_label(self, chords, position):
    """Returns the label for the chord progression event at the given position.

    Args:
      chords: A ChordProgression object.
      position: An integer event position in the chord progression.

    Returns:
      A label, an integer in the range [0, self.num_classes).
    """
    pass

  def encode(self, chords, transpose_amount):
    """Returns a SequenceExample for the given chord progression.

    Args:
      chords: A ChordProgression object.
      transpose_amount: The number of half steps to transpose the chords.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    chords.transpose(transpose_amount)
    inputs = []
    labels = []
    for i in range(len(chords) - 1):
      inputs.append(self.chords_to_input(chords, i))
      labels.append(self.chords_to_label(chords, i + 1))
    return sequence_example_lib.make_sequence_example(inputs, labels)

  def get_inputs_batch(self, chord_progressions, full_length=False):
    """Returns an inputs batch for the given chord progressions.

    Args:
      chord_progressions: A list of ChordProgression objects.
      full_length: If True, the inputs batch will be for the full length of
          each chord progression. If False, the inputs batch will only be for
          the last event of each chord progression. A full-length inputs batch
          is used for the first step of extending the chord progressions, since
          the rnn cell state needs to be initialized with the priming sequence.
          For subsequent generation steps, only a last-event inputs batch is
          used.

    Returns:
      An inputs batch. If `full_length` is True, the shape will be
      [len(chord_progressions), len(chord_progressions[0]), INPUT_SIZE]. If
      `full_length` is False, the shape will be
      [len(chord_progressions), 1, INPUT_SIZE].
    """
    inputs_batch = []
    for chords in chord_progressions:
      inputs = []
      if full_length and len(chord_progressions):
        for i in range(len(chords)):
          inputs.append(self.chords_to_input(chords, i))
      else:
        inputs.append(self.chords_to_input(chords, len(chords) - 1))
      inputs_batch.append(inputs)
    return inputs_batch

  @abc.abstractmethod
  def class_index_to_chord_event(self, class_index, chords):
    """Returns the chords event for the given class index.

    This is the reverse process of the self.chords_to_label method.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      chords: A ChordProgression object. This object is not used in this
          implementation.

    Returns:
      A ChordProgression event value, a chord figure string or NO_CHORD.
    """
    pass

  def extend_chord_progressions(self, chord_progressions, softmax):
    """Extends the chord progressions by sampling the softmax probabilities.

    Args:
      chord_progressions: A list of ChordProgression objects.
      softmax: A list of softmax probability vectors. The list of softmaxes
          should be the same length as the list of chord progressions.
    """
    num_classes = len(softmax[0][0])
    for i in xrange(len(chord_progressions)):
      chosen_class = np.random.choice(num_classes, p=softmax[i][-1])
      chord_event = self.class_index_to_chord_event(chosen_class,
                                                    chord_progressions[i])
      chord_progressions[i].append_event(chord_event)
