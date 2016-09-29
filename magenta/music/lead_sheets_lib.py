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
"""Utility functions for working with lead sheets."""

import abc
import copy
import itertools

from magenta.music import chords_lib
from magenta.music import constants
from magenta.music import events_lib
from magenta.music import melodies_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class MelodyChordsMismatchException(Exception):
  pass


class LeadSheet(events_lib.EventSequence):
  """A wrapper around MonophonicMelody and ChordProgression.

  Attributes:
    melody: A MonophonicMelody object, the lead sheet melody.
    chords: A ChordProgression object, the underlying chords.
  """

  def __init__(self):
    """Construct an empty LeadSheet."""
    self._reset()

  def _reset(self):
    """Clear events and reset object state."""
    self._melody = melodies_lib.MonophonicMelody()
    self._chords = chords_lib.ChordProgression()

  def __iter__(self):
    """Return an iterator over (melody, chord) tuples in this LeadSheet.

    Returns:
      Python iterator over (melody, chord) event tuples.
    """
    return itertools.izip(self._melody, self._chords)

  def __getitem__(self, i):
    """Returns the melody-chord tuple at the given index."""
    return self._melody[i], self._chords[i]

  def __getslice__(self, i, j):
    """Returns the melody-chord tuples in the given slice range."""
    return zip(self._melody[i:j], self._chords[i:j])

  def __len__(self):
    """How many events (melody-chord tuples) are in this LeadSheet.

    Returns:
      Number of events as an integer.
    """
    return len(self._melody)

  def __deepcopy__(self, unused_memo=None):
    new_copy = type(self)()
    new_copy.from_melody_and_chords(copy.deepcopy(self._melody),
                                    copy.deepcopy(self._chords))
    return new_copy

  def __eq__(self, other):
    if not isinstance(other, LeadSheet):
      return False
    return (self._melody == other.melody and
            self._chords == other.chords)

  @property
  def start_step(self):
    return self._melody.start_step

  @property
  def end_step(self):
    return self._melody.end_step

  @property
  def steps_per_bar(self):
    return self._melody.steps_per_bar

  @property
  def steps_per_quarter(self):
    return self._melody.steps_per_quarter

  @property
  def melody(self):
    """Return the melody of the lead sheet.

    Returns:
        The lead sheet melody, a MonophonicMelody object.
    """
    return self._melody

  @property
  def chords(self):
    """Return the chord progression of the lead sheet.

    Returns:
        The lead sheet chords, a ChordProgression object.
    """
    return self._chords

  def append_event(self, event):
    """Appends event to the end of the sequence and increments the end step.

    Args:
      event: The event (a melody-chord tuple) to append to the end.
    """
    melody_event, chord_event = event
    self._melody.append_event(melody_event)
    self._chords.append_event(chord_event)

  def from_event_list(self, events, start_step=0,
                      steps_per_bar=DEFAULT_STEPS_PER_BAR,
                      steps_per_quarter=DEFAULT_STEPS_PER_QUARTER):
    """Initializes with a list of event values (melody-chord tuples).

    Args:
      events: List of melody-chord tuples.
      start_step: The integer starting step offset.
      steps_per_bar: The number of steps in a bar.
      steps_per_quarter: The number of steps in a quarter note.
    """
    melody_events, chord_events = zip(*events)
    self._melody.from_event_list(melody_events, start_step=start_step,
                                 steps_per_bar=steps_per_bar,
                                 steps_per_quarter=steps_per_quarter)
    self._chords.from_event_list(chord_events, start_step=start_step,
                                 steps_per_bar=steps_per_bar,
                                 steps_per_quarter=steps_per_quarter)

  def from_melody_and_chords(self, melody, chords):
    """Initializes a LeadSheet with a given melody and chords.

    Args:
      melody: A MonophonicMelody object.
      chords: A ChordProgression object.

    Raises:
      MelodyChordsMismatchException: If the melody and chord progression differ
          in temporal resolution or position in the source sequence.
    """
    if (len(melody) != len(chords) or
        melody.steps_per_bar != chords.steps_per_bar or
        melody.steps_per_quarter != chords.steps_per_quarter or
        melody.start_step != chords.start_step or
        melody.end_step != chords.end_step):
      raise MelodyChordsMismatchException()
    self._melody = melody
    self._chords = chords

  def to_sequence(self,
                  velocity=100,
                  instrument=0,
                  sequence_start_time=0.0,
                  qpm=120.0):
    """Converts the LeadSheet to NoteSequence proto.

    Args:
      velocity: Midi velocity to give each melody note. Between 1 and 127
          (inclusive).
      instrument: Midi instrument to give each melody note.
      sequence_start_time: A time in seconds (float) that the first note (and
          chord) in the sequence will land on.
      qpm: Quarter notes per minute (float).

    Returns:
      A NoteSequence proto encoding the melody and chords from the lead sheet.
    """
    sequence = self._melody.to_sequence(
        velocity=velocity, instrument=instrument,
        sequence_start_time=sequence_start_time, qpm=qpm)
    chord_sequence = self._chords.to_sequence(
        sequence_start_time=sequence_start_time, qpm=qpm)
    # A little ugly, but just add the chord annotations to the melody sequence.
    for text_annotation in chord_sequence.text_annotations:
      if text_annotation.annotation_type == CHORD_SYMBOL:
        chord = sequence.text_annotations.add()
        chord.CopyFrom(text_annotation)
    return sequence

  def transpose(self, transpose_amount, min_note=0, max_note=128):
    """Transpose notes and chords in this LeadSheet.

    All notes and chords are transposed the specified amount. Additionally,
    all notes are octave shifted to lie within the [min_note, max_note) range.

    Args:
      transpose_amount: The number of half steps to transpose this
          LeadSheet. Positive values transpose up. Negative values
          transpose down.
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
    """
    self._melody.transpose(transpose_amount, min_note, max_note)
    self._chords.transpose(transpose_amount)

  def squash(self, min_note, max_note, transpose_to_key):
    """Transpose and octave shift the notes and chords in this LeadSheet.

    Args:
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
      transpose_to_key: The lead sheet is transposed to be in this key.

    Returns:
      The transpose amount, in half steps.
    """
    transpose_amount = self._melody.squash(min_note, max_note,
                                           transpose_to_key)
    self._chords.transpose(transpose_amount)
    return transpose_amount

  def set_length(self, steps):
    """Sets the length of the lead sheet to the specified number of steps.

    Args:
      steps: How many steps long the lead sheet should be.
    """
    self._melody.set_length(steps)
    self._chords.set_length(steps)


def extract_lead_sheet_fragments(quantized_sequence,
                                 min_bars=7,
                                 gap_bars=1.0,
                                 min_unique_pitches=5,
                                 ignore_polyphonic_notes=True,
                                 require_chords=False):
  """Extracts a list of lead sheet fragments from the given QuantizedSequence.

  This function first extracts melodies using melodies_lib.extract_melodies,
  then extracts the chords underlying each melody using
  chords_lib.extract_chords_for_melodies.

  Args:
    quantized_sequence: A sequences_lib.QuantizedSequence object.
    min_bars: Minimum length of melodies in number of bars. Shorter melodies are
        discarded.
    gap_bars: A melody comes to an end when this number of bars (measures) of
        silence is encountered.
    min_unique_pitches: Minimum number of unique notes with octave equivalence.
        Melodies with too few unique notes are discarded.
    ignore_polyphonic_notes: If True, melodies will be extracted from
        `quantized_sequence` tracks that contain polyphony (notes start at the
        same time). If False, tracks with polyphony will be ignored.
    require_chords: If True, only return lead sheets that have at least one
        chord other than NO_CHORD. If False, lead sheets with only melody will
        also be returned.

  Returns:
    A python list of LeadSheet instances.

  Raises:
    NonIntegerStepsPerBarException: If `quantized_sequence`'s bar length
        (derived from its time signature) is not an integer number of time
        steps.
  """
  stats = dict([('empty_chord_progressions',
                 statistics.Counter('empty_chord_progressions'))])
  melodies, melody_stats = melodies_lib.extract_melodies(
      quantized_sequence, min_bars=min_bars, gap_bars=gap_bars,
      min_unique_pitches=min_unique_pitches,
      ignore_polyphonic_notes=ignore_polyphonic_notes)
  chord_progressions, chord_stats = chords_lib.extract_chords_for_melodies(
      quantized_sequence, melodies)
  lead_sheets = []
  for melody, chords in zip(melodies, chord_progressions):
    if chords is not None:
      if require_chords and all(chord == chords_lib.NO_CHORD
                                for chord in chords):
        stats['empty_chord_progressions'].increment()
      else:
        lead_sheet = LeadSheet()
        lead_sheet.from_melody_and_chords(melody, chords)
        lead_sheets.append(lead_sheet)
  return lead_sheets, stats.values() + melody_stats + chord_stats


class LeadSheetEncoderDecoder(events_lib.EventsEncoderDecoder):
  """An abstract class for translating between lead sheets and model data.

  When building your dataset, the `encode` method takes in a lead sheet and
  returns a SequenceExample of inputs and labels. These SequenceExamples are
  fed into the model during training and evaluation.

  During lead sheet generation, the `get_inputs_batch` method takes in a list
  of the current lead sheets and returns an inputs batch which is fed into the
  model to predict what the next note and chord should be for each lead sheet.
  The `extend_event_sequences` method takes in the list of lead sheets and the
  softmax returned by the model and extends each lead sheet by one step by
  sampling from the softmax probabilities. This loop (`get_inputs_batch` ->
  inputs batch is fed through the model to get a softmax ->
  `extend_event_sequences`) is repeated until the generated lead sheets have
  reached the desired length.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def min_note(self):
    """The min pitch value to allow for melodies.

    Returns:
        An integer, the min pitch value to allow for melodies.
    """
    pass

  @abc.abstractproperty
  def max_note(self):
    """The max pitch value to allow for melodies.

    Returns:
        An integer, the max pitch value to allow for melodies.
    """
    pass

  @abc.abstractproperty
  def transpose_to_key(self):
    """The key, an integer from 0 to 11 inclusive, into which to transpose.

    Returns:
        An integer, the key into which to transpose.
    """
    pass

  def squash_and_encode(self, lead_sheet):
    """Returns a SequenceExample for the given lead sheet.

    Args:
      lead_sheet: A LeadSheet object.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    lead_sheet.squash(self.min_note, self.max_note, self.transpose_to_key)
    return self._encode(lead_sheet)


class LeadSheetProductEncoderDecoder(LeadSheetEncoderDecoder):
  """A LeadSheetEncoderDecoder that trivially encodes/decodes melody & chords.

  The encoder/decoder uses a MelodyEncoderDecoder and a ChordsEncoderDecoder
  and trivially combines them. The input is a concatenation of the melody and
  chords inputs, and the output label is a product of the melody and chords
  labels.
  """

  def __init__(self, melody_encoder_decoder, chords_encoder_decoder):
    self._melody_encoder_decoder = melody_encoder_decoder
    self._chords_encoder_decoder = chords_encoder_decoder

  @property
  def min_note(self):
    return self._melody_encoder_decoder.min_note

  @property
  def max_note(self):
    return self._melody_encoder_decoder.max_note

  @property
  def transpose_to_key(self):
    return self._melody_encoder_decoder.transpose_to_key

  @property
  def input_size(self):
    return (self._melody_encoder_decoder.input_size +
            self._chords_encoder_decoder.input_size)

  @property
  def num_classes(self):
    return (self._melody_encoder_decoder.num_classes *
            self._chords_encoder_decoder.num_classes)

  def events_to_input(self, events, position):
    """Returns the input vector for the lead sheet event at the given position.

    The input vector is the concatenation of the input vectors for the melody
    and chords, using their respective encoder-decoders.

    Args:
      events: A LeadSheet object.
      position: An integer event position in the lead sheet.

    Returns:
      An input vector, a self.input_size length list of floats.
    """
    melody_input = self._melody_encoder_decoder.events_to_input(
        events.melody, position)
    chords_input = self.chords_encoder_decoder.events_to_input(
        events.chords, position)
    return melody_input + chords_input

  def events_to_label(self, events, position):
    """Returns the label for the lead sheet event at the given position.

    The label is a cartesian product of the melody label and chord label at the
    given position, mapped to a single integer.

    Args:
      events: A LeadSheet object.
      position: An integer event position in the lead sheet.

    Returns:
      A label, an integer in the range [0, self.num_classes).
    """
    melody_label = self._melody_encoder_decoder.events_to_label(
        events.melody, position)
    chords_label = self._chords_encoder_decoder.events_to_label(
        events.chords, position)
    return melody_label + (self._melody_encoder_decoder.num_classes *
                           chords_label)

  def class_index_to_event(self, class_index, events):
    """Returns the lead sheet event for the given class index.

    This is the reverse process of the self.events_to_label method. The lead
    sheet event will be a tuple, the first element of which is the melody event
    and the second element of which is the chord event.

    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A LeadSheet object.

    Returns:
      A lead sheet event value, a tuple containing the melody event and chord
          event.
    """
    melody_index = class_index % self._melody_encoder_decoder.num_classes
    chord_index = class_index / self._melody_encoder_decoder.num_classes
    return (
        self._melody_encoder_decoder.class_index_to_event(melody_index,
                                                          events.melody),
        self._chords_encoder_decoder.class_index_to_event(chord_index,
                                                          events.chords))
