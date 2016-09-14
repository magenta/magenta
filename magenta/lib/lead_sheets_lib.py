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
import itertools
import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin

from magenta.lib import chords_lib
from magenta.lib import melodies_lib
from magenta.lib import sequence_example_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2


# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class MelodyChordsMismatchException(Exception):
  pass


class LeadSheet(object):
  """A wrapper around MonophonicMelody and ChordProgression.

  Attributes:
    start_step: The offset of the first step of the lead sheet relative to the
        beginning of the source sequence. Will always be the first step of a
        bar.
    end_step: The offset to the beginning of the bar following the last step
        of the lead sheet relative the beginning of the source sequence. Will
        always be the first step of a bar.
    steps_per_quarter: Number of steps in in a quarter note.
    steps_per_bar: Number of steps in a bar (measure) of music.
  """

  def __init__(self, melody, chords):
    """Construct a LeadSheet from a given melody and chords.

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
    self.melody = melody
    self.chords = chords
    self.steps_per_bar = melody.steps_per_bar
    self.steps_per_quarter = melody.steps_per_quarter
    self.start_step = melody.start_step
    self.end_step = melody.end_step

  def __iter__(self):
    """Return an iterator over (melody, chord) tuples in this LeadSheet.

    Returns:
      Python iterator over (melody, chord) event tuples.
    """
    return itertools.izip(self.melody, self.chords)

  def __len__(self):
    """How many events are in this LeadSheet.

    Returns:
      Number of events as an int.
    """
    return len(self.melody)

  def __eq__(self, other):
    if not isinstance(other, LeadSheet):
      return False
    return (self.melody == other.melody and
            self.chords == other.chords and
            self.steps_per_bar == other.steps_per_bar and
            self.start_step == other.start_step and
            self.end_step == other.end_step)

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
    sequence = self.melody.to_sequence(velocity=velocity,
                                       instrument=instrument,
                                       sequence_start_time=sequence_start_time,
                                       qpm=qpm)
    chord_sequence = self.chords.to_sequence(
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
    self.melody.transpose(transpose_amount, min_note, max_note)
    self.chords.transpose(transpose_amount)

  def squash(self, min_note, max_note, transpose_to_key):
    """Transpose and octave shift the notes and chords in this LeadSheet.

    Args:
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
      transpose_to_key: The lead sheet is transposed to be in this key.

    Returns:
      The transpose amount, in half steps.
    """
    transpose_amount = self.melody.squash(min_note, max_note, transpose_to_key)
    self.chords.transpose(transpose_amount)
    return transpose_amount

  def set_length(self, steps):
    """Sets the length of the lead sheet to the specified number of steps.

    Args:
      steps: how many steps long the lead sheet should be.
    """
    self.melody.set_length(steps)
    self.chords.set_length(steps)

    assert self.melody.steps_per_bar == self.chords.steps_per_bar
    assert self.melody.steps_per_quarter == self.chords.steps_per_quarter
    assert self.melody.start_step == self.chords.start_step
    assert self.melody.end_step == self.chords.end_step

    self.steps_per_bar = self.melody.steps_per_bar
    self.steps_per_quarter = self.melody.steps_per_quarter
    self.start_step = self.melody.start_step
    self.end_step = self.melody.end_step


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
        lead_sheets.append(LeadSheet(melody, chords))
  return lead_sheets, stats.values() + melody_stats + chord_stats


class LeadSheetEncoderDecoder(object):
  """An abstract class for translating between lead sheets and model data.

  When building your dataset, the `encode` method takes in a lead sheet and
  returns a SequenceExample of inputs and labels. These SequenceExamples are
  fed into the model during training and evaluation.

  During lead sheet generation, the `get_inputs_batch` method takes in a list
  of the current lead sheets and returns an inputs batch which is fed into the
  model to predict what the next note and chord should be for each lead sheet.
  The `extend_lead_sheets` method takes in the list of lead sheets and the
  softmax returned by the model and extends each lead sheet by one step by
  sampling from the softmax probabilities. This loop (`get_inputs_batch` ->
  inputs batch is fed through the model to get a softmax ->
  `extend_lead_sheets`) is repeated until the generated lead sheets have
  reached the desired length.

  The `lead_sheet_to_input`, `lead_sheet_to_label`, and
  `class_index_to_melody_event` methods must be overwritten to be specific to
  your model. See chords_and_melody/basic_rnn/basic_rnn_encoder_decoder.py
  for an example of this.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def min_note(self):
    """The min pitch value to allow for melodies.

    Returns:
        An int, the min pitch value to allow for melodies.
    """
    pass

  @abc.abstractproperty
  def max_note(self):
    """The max pitch value to allow for melodies.

    Returns:
        An int, the max pitch value to allow for melodies.
    """
    pass

  @abc.abstractproperty
  def transpose_to_key(self):
    """The key, an int from 0 to 11 inclusive, into which to transpose.

    Returns:
        An int, the key into which to transpose.
    """
    pass

  @abc.abstractproperty
  def input_size(self):
    """The size of the input vector used by this model.

    Returns:
        An int, the length of the list returned by self.lead_sheet_to_input.
    """
    pass

  @abc.abstractproperty
  def num_classes(self):
    """The range of labels used by this model.

    Returns:
        An int, the range of ints that can be returned by
        self.lead_sheet_to_label.
    """
    pass

  @abc.abstractmethod
  def lead_sheet_to_input(self, lead_sheet, position):
    """Returns the input vector for the lead sheet event at the given position.

    Args:
      lead_sheet: A LeadSheet object.
      position: An integer event position in the lead sheet.

    Returns:
      An input vector, a self.input_size length list of floats.
    """
    pass

  @abc.abstractmethod
  def lead_sheet_to_label(self, lead_sheet, position):
    """Returns the label for the lead sheet event at the given position.

    Args:
      lead_sheet: A LeadSheet object.
      position: An integer event position in the lead sheet.

    Returns:
      A label, an int in the range [0, self.num_classes).
    """
    pass

  def encode(self, lead_sheet):
    """Returns a SequenceExample for the given lead sheet.

    Args:
      lead_sheet: A LeadSheet object.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    lead_sheet.squash(self.min_note, self.max_note, self.transpose_to_key)
    inputs = []
    labels = []
    for i in range(len(lead_sheet) - 1):
      inputs.append(self.lead_sheet_to_input(lead_sheet, i))
      labels.append(self.lead_sheet_to_label(lead_sheet, i + 1))
    return sequence_example_lib.make_sequence_example(inputs, labels)

  def get_inputs_batch(self, lead_sheets, full_length=False):
    """Returns an inputs batch for the given lead sheets.

    Args:
      lead_sheets: A list of LeadSheets objects.
      full_length: If True, the inputs batch will be for the full length of
          each lead sheet. If False, the inputs batch will only be for the last
          event of each lead sheet. A full-length inputs batch is used for the
          first step of extending the lead sheets, since the rnn cell state
          needs to be initialized with the priming sequence. For subsequent
          generation steps, only a last-event inputs batch is used.

    Returns:
      An inputs batch. If `full_length` is True, the shape will be
      [len(lead_sheets), len(lead_sheets[0]), INPUT_SIZE]. If `full_length` is
      False, the shape will be [len(lead_sheets), 1, INPUT_SIZE].
    """
    inputs_batch = []
    for lead_sheet in lead_sheets:
      inputs = []
      if full_length and len(lead_sheet):
        for i in range(len(lead_sheet)):
          inputs.append(self.lead_sheet_to_input(lead_sheet, i))
      else:
        inputs.append(self.lead_sheet_to_input(lead_sheet, len(lead_sheet) - 1))
      inputs_batch.append(inputs)
    return inputs_batch

  @abc.abstractmethod
  def class_index_to_melody_and_chord_event(self, class_index, lead_sheet):
    """Returns the melody event and chord event for the given class index.

    This is the reverse process of the self.lead_sheet_to_label method.

    Args:
      class_index: An int in the range [0, self.num_classes).
      lead_sheet: A LeadSheet object. This object is not used in this
          implementation.

    Returns:
      A (melody event, chord event) tuple.
    """
    pass

  def extend_lead_sheets(self, lead_sheets, softmax):
    """Extends the lead sheets by sampling from the softmax probabilities.

    Args:
      lead_sheets: A list of LeadSheet objects.
      softmax: A list of softmax probability vectors. The list of softmaxes
          should be the same length as the list of lead sheets.
    """
    num_classes = len(softmax[0][0])
    for i in xrange(len(lead_sheets)):
      chosen_class = np.random.choice(num_classes, p=softmax[i][-1])
      melody_event, chord_event = self.class_index_to_melody_and_chord_event(
          chosen_class, lead_sheets[i])
      lead_sheets[i].melody.append_event(melody_event)
      lead_sheets[i].chords.append_event(chord_event)


class LeadSheetProductEncoderDecoder(LeadSheetEncoderDecoder):
  """A LeadSheetEncoderDecoder that trivially encodes/decodes melody & chords.

  The encoder/decoder uses a MelodyEncoderDecoder and a ChordsEncoderDecoder
  and trivially combines them. The input is a concatenation of the melody and
  chords inputs, and the output label is a product of the melody and chords
  labels.

  Attributes:
    melody_encoder_decoder: A MelodyEncoderDecoder object.
    chords_encoder_decoder: A ChordsEncoderDecoder object.
  """

  def __init__(self, melody_encoder_decoder, chords_encoder_decoder):
    self.melody_encoder_decoder = melody_encoder_decoder
    self.chords_encoder_decoder = chords_encoder_decoder

  @property
  def min_note(self):
    return self.melody_encoder_decoder.min_note

  @property
  def max_note(self):
    return self.melody_encoder_decoder.max_note

  @property
  def transpose_to_key(self):
    return self.melody_encoder_decoder.transpose_to_key

  @property
  def input_size(self):
    return (self.melody_encoder_decoder.input_size +
            self.chords_encoder_decoder.input_size)

  @property
  def num_classes(self):
    return (self.melody_encoder_decoder.num_classes *
            self.chords_encoder_decoder.num_classes)

  def lead_sheet_to_input(self, lead_sheet, position):
    melody_input = self.melody_encoder_decoder.melody_to_input(
        lead_sheet.melody, position)
    chords_input = self.chords_encoder_decoder.chords_to_input(
        lead_sheet.chords, position)
    return melody_input + chords_input

  def lead_sheet_to_label(self, lead_sheet, position):
    melody_label = self.melody_encoder_decoder.melody_to_label(
        lead_sheet.melody, position)
    chords_label = self.chords_encoder_decoder.chords_to_label(
        lead_sheet.chords, position)
    return melody_label + self.melody_encoder_decoder.num_classes * chords_label

  def class_index_to_melody_and_chord_event(self, class_index, lead_sheet):
    melody_index = class_index % self.melody_encoder_decoder.num_classes
    chord_index = class_index / self.melody_encoder_decoder.num_classes
    return (
        self.melody_encoder_decoder.class_index_to_melody_event(
            melody_index, lead_sheet.melody),
        self.chords_encoder_decoder.class_index_to_chord_event(
            chord_index, lead_sheet.chords))
