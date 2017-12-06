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
"""MusicVAE data library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import copy
import functools
import itertools
import random

# internal imports
import numpy as np
import tensorflow as tf

import magenta.music as mm
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = 128

MAX_INSTRUMENT_NUMBER = 127

MEL_PROGRAMS = range(0, 31)  # piano, chromatic percussion, organ, guitar
BASS_PROGRAMS = range(32, 39)
ELECTRIC_BASS_PROGRAM = 33


def _maybe_pad_seqs(seqs, dtype):
  """Pads sequences to match the longest and returns as a numpy array."""
  if not len(seqs):  # pylint:disable=g-explicit-length-test
    return np.zeros((0, 0, 0), dtype), np.zeros((0, 1), np.int32)
  lengths = [len(s) for s in seqs]
  if len(set(lengths)) == 1:
    return np.array(seqs, dtype), np.array(lengths, np.int32)
  else:
    length = max(lengths)
    return (np.array([np.pad(s, [(0, length - len(s)), (0, 0)], mode='constant')
                      for s in seqs], dtype),
            np.array(lengths, np.int32))


def np_onehot(indices, depth, dtype=np.bool):
  """Converts 1D array of indices to a one-hot 2D array with given depth."""
  onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
  onehot_seq[np.arange(len(indices)), indices] = 1.0
  return onehot_seq


class NoteSequenceAugmenter(object):
  """Class for augmenting NoteSequences.

  Args:
    transpose_range: A tuple containing the inclusive, integer range of
        transpose amounts to sample from. If None, no transposition is applied.
    stretch_range: A tuple containing the inclusive, float range of stretch
        amounts to sample from.
  Returns:
    The augmented NoteSequence.
  """

  def __init__(self, transpose_range=None, stretch_range=None):
    self._transpose_range = transpose_range
    self._stretch_range = stretch_range

  def augment(self, note_sequence):
    """Python implementation that augments the NoteSequence."""
    trans_amt = (random.randint(*self._transpose_range)
                 if self._transpose_range else 0)
    stretch_factor = (random.uniform(*self._stretch_range)
                      if self._stretch_range else 1.0)
    augmented_ns = copy.deepcopy(note_sequence)
    del augmented_ns.notes[:]
    for note in note_sequence.notes:
      aug_pitch = note.pitch + trans_amt
      if MIN_MIDI_PITCH <= aug_pitch <= MAX_MIDI_PITCH:
        augmented_ns.notes.add().CopyFrom(note)
        augmented_ns.notes[-1].pitch = aug_pitch

    augmented_ns = sequences_lib.stretch_note_sequence(
        augmented_ns, stretch_factor)
    return augmented_ns

  def tf_augment(self, note_sequence_scalar):
    """TF op that augments the NoteSequence."""
    def _augment_str(note_sequence_str):
      note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)
      augmented_ns = self.augment(note_sequence)
      return [augmented_ns.SerializeToString()]

    augmented_note_sequence_scalar = tf.py_func(
        _augment_str,
        [note_sequence_scalar],
        tf.string,
        name='augment')
    augmented_note_sequence_scalar.set_shape(())
    return augmented_note_sequence_scalar


class BaseNoteSequenceConverter(object):
  """Base class for NoteSequence data converters.

  Inheriting classes must implement the following abstract methods:
    -`_to_tensors`
    -`_to_notesequences`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, input_depth, input_dtype, output_depth, output_dtype,
               end_token, presplit_on_time_changes=True,
               max_tensors_per_notesequence=None):
    """Initializes BaseNoteSequenceConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      end_token: Optional end token.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
        before converting.
      max_tensors_per_notesequence: The maximum number of outputs to return
        for each NoteSequence.
    """
    self._input_depth = input_depth
    self._input_dtype = input_dtype
    self._output_depth = output_depth
    self._output_dtype = output_dtype
    self._end_token = end_token
    self._presplit_on_time_changes = presplit_on_time_changes
    self._max_tensors_per_notesequence = max_tensors_per_notesequence
    self._is_training = False

  @property
  def is_training(self):
    return self._is_training

  @is_training.setter
  def is_training(self, value):
    self._is_training = value

  @property
  def max_tensors_per_notesequence(self):
    return self._max_tensors_per_notesequence

  @max_tensors_per_notesequence.setter
  def max_tensors_per_notesequence(self, value):
    self._max_tensors_per_notesequence = value

  @property
  def end_token(self):
    """End token, or None."""
    return self._end_token

  @property
  def input_depth(self):
    """Dimension of inputs (to encoder) at each timestep of the sequence."""
    return self._input_depth

  @property
  def input_dtype(self):
    """DType of inputs (to encoder)."""
    return self._input_dtype

  @property
  def output_depth(self):
    """Dimension of outputs (from decoder) at each timestep of the sequence."""
    return self._output_depth

  @property
  def output_dtype(self):
    """DType of outputs (from decoder)."""
    return self._output_dtype

  @abc.abstractmethod
  def _to_tensors(self, note_sequence):
    """Implementation that converts `note_sequence` into list of tensors."""
    pass

  @abc.abstractmethod
  def _to_notesequences(self, samples):
    """Implementation that decodes model samples into list of NoteSequences."""
    pass

  def _maybe_sample_outputs(self, outputs):
    """If should limit outputs, returns up to limit (randomly if training)."""
    if (not self.max_tensors_per_notesequence or
        len(outputs) <= self.max_tensors_per_notesequence):
      return outputs
    if self.is_training:
      indices = set(np.random.choice(
          len(outputs), size=self.max_tensors_per_notesequence, replace=False))
      return [outputs[i] for i in indices]
    else:
      return outputs[:self.max_tensors_per_notesequence]

  def to_tensors(self, note_sequence):
    """Python method that converts `note_sequence` into list of tensors."""
    if self._presplit_on_time_changes:
      note_sequences = sequences_lib.split_note_sequence_on_time_changes(
          note_sequence)
    else:
      note_sequences = [note_sequence]
    results = []
    for ns in note_sequences:
      r = self._to_tensors(ns)
      if r[0]:
        results.extend(zip(*r))
    sampled_results = self._maybe_sample_outputs(results)
    return list(zip(*sampled_results)) if sampled_results else ([], [])

  def to_notesequences(self, samples):
    """Python method that decodes samples into list of NoteSequences."""
    return self._to_notesequences(samples)

  def tf_to_tensors(self, note_sequence_scalar):
    """TensorFlow op that converts NoteSequence into output tensors.

    Sequences will be padded to match the length of the longest.

    Args:
      note_sequence_scalar: A scalar of type tf.String containing a serialized
         NoteSequence to be encoded.

    Returns:
      outputs: A Tensor, shaped [num encoded seqs, max(lengths), output_depth],
        containing the padded output encodings resulting from the input
        NoteSequence.
      lengths: A tf.int32 Tensor, shaped [num encoded seqs], containing the
        unpadded lengths of the tensor sequences resulting from the input
        NoteSequence.
    """
    def _convert_and_pad(note_sequence_str):
      note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)
      inputs, outputs = self.to_tensors(note_sequence)
      inputs, lengths = _maybe_pad_seqs(inputs, self.output_dtype)
      outputs, _ = _maybe_pad_seqs(outputs, self.output_dtype)
      return inputs, outputs, lengths
    inputs, outputs, lengths = tf.py_func(
        _convert_and_pad,
        [note_sequence_scalar],
        [self.input_dtype, self.output_dtype, tf.int32],
        name='convert_and_pad')
    inputs.set_shape([None, None, self.input_depth])
    outputs.set_shape([None, None, self.output_depth])
    lengths.set_shape([None])
    return inputs, outputs, lengths


class LegacyEventListOneHotConverter(BaseNoteSequenceConverter):
  """Converts NoteSequences using legacy EventSequenceEncoderDecoder framework.

  Quantizes the sequences, extracts event lists in the requested size range,
  uniquifies, and converts to encoding. Uses the EventSequenceEncoderDecoder's
  output encoding for both the input and output.

  Args:
    event_list_fn: A function that returns a new EventList.
    event_extractor_fn: A function for extracing events into EventLists. The
      sole input should be the quantized NoteSequence.
    legacy_encoder_decoder: An instantiated EventSequenceEncoderDecoder to use.
    add_end_token: Whether or not to add the end token. Recommended to be False
      for fixed-length outputs.
    slice_bars: Optional size of window to slide over raw event lists after
      extraction.
    steps_per_quarter: The number of quantization steps per quarter note.
      Mututally exclusive with `steps_per_second`.
    steps_per_second: The number of quantization steps per second.
      Mututally exclusive with `steps_per_quarter`.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
     before converting.
  """

  def __init__(self, event_list_fn, event_extractor_fn,
               legacy_encoder_decoder, add_end_token=False, slice_bars=None,
               slice_steps=None, steps_per_quarter=None, steps_per_second=None,
               quarters_per_bar=4, pad_to_total_time=False,
               max_tensors_per_notesequence=None,
               presplit_on_time_changes=True):
    if (steps_per_quarter, steps_per_second).count(None) != 1:
      raise ValueError(
          'Exactly one of `steps_per_quarter` and `steps_per_second` should be '
          'provided.')
    if (slice_bars, slice_steps).count(None) == 0:
      raise ValueError(
          'At most one of `slice_bars` and `slice_steps` should be provided.')
    self._event_list_fn = event_list_fn
    self._event_extractor_fn = event_extractor_fn
    self._legacy_encoder_decoder = legacy_encoder_decoder
    self._steps_per_quarter = steps_per_quarter
    if steps_per_quarter:
      self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._steps_per_second = steps_per_second
    if slice_bars:
      self._slice_steps = self._steps_per_bar * slice_bars
    else:
      self._slice_steps = slice_steps
    self._pad_to_total_time = pad_to_total_time

    depth = legacy_encoder_decoder.num_classes + add_end_token
    super(LegacyEventListOneHotConverter, self).__init__(
        input_depth=depth,
        input_dtype=np.bool,
        output_depth=depth,
        output_dtype=np.bool,
        end_token=legacy_encoder_decoder.num_classes if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors(self, note_sequence):
    """Converts NoteSequence to unique, one-hot tensor sequences."""
    try:
      if self._steps_per_quarter:
        quantized_sequence = mm.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
        if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
            self._steps_per_bar):
          return [], []
      else:
        quantized_sequence = mm.quantize_note_sequence_absolute(
            note_sequence, self._steps_per_second)
    except (mm.BadTimeSignatureException, mm.NonIntegerStepsPerBarException,
            mm.NegativeTimeException) as e:
      return [], []

    event_lists, unused_stats = self._event_extractor_fn(quantized_sequence)
    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_tuples = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_tuples.append(tuple(l[i - self._slice_steps: i]))
    else:
      sliced_event_tuples = [tuple(l) for l in event_lists]

    # TODO(adarob): Consider handling the fact that different event lists can
    # be mapped to identical tensors by the encoder_decoder (e.g., Drums).

    unique_event_tuples = list(set(sliced_event_tuples))
    unique_event_tuples = self._maybe_sample_outputs(unique_event_tuples)

    seqs = []
    for t in unique_event_tuples:
      seqs.append(np_onehot(
          [self._legacy_encoder_decoder.encode_event(e) for e in t] +
          ([] if self._end_token is None else [self._end_token]),
          self.output_depth, self.output_dtype))

    return seqs, seqs

  def _to_notesequences(self, samples):
    sample_labels = np.argmax(samples, axis=-1)
    output_sequences = []
    for s in sample_labels:
      if self._end_token is not None and self._end_token in s:
        s = s[:s.tolist().index(self._end_token)]
      event_list = self._event_list_fn()
      for e in s:
        assert e != self._end_token
        event_list.append(self._legacy_encoder_decoder.decode_event(e))
      output_sequences.append(event_list.to_sequence(velocity=80))
    return output_sequences


class OneHotMelodyConverter(LegacyEventListOneHotConverter):
  """Converter for legacy MelodyOneHotEncoding.

  Args:
    min_pitch: The minimum pitch to model. Those below this value will be
      ignored.
    max_pitch: The maximum pitch to model. Those above this value will be
      ignored.
    valid_programs: Optional set of program numbers to allow.
    skip_polyphony: Whether to skip polyphonic instruments. If False, the
      highest pitch will be taken in polyphonic sections.
    max_bars: Optional maximum number of bars per extracted melody, before
      slicing.
    slice_bars: Optional size of window to slide over raw Melodies after
      extraction.
    gap_bars: If this many bars or more of non-events follow a note event, the
       melody is ended. Disabled when set to 0 or None.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    add_end_token: Whether to add an end token at the end of each sequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
  """

  def __init__(self, min_pitch=PIANO_MIN_MIDI_PITCH,
               max_pitch=PIANO_MAX_MIDI_PITCH, valid_programs=None,
               skip_polyphony=False, max_bars=None, slice_bars=None,
               gap_bars=1.0, steps_per_quarter=4, quarters_per_bar=4,
               add_end_token=False, pad_to_total_time=False,
               max_tensors_per_notesequence=5, presplit_on_time_changes=True):
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch
    self._valid_programs = valid_programs
    steps_per_bar = steps_per_quarter * quarters_per_bar
    max_steps_truncate = steps_per_bar * max_bars if max_bars else None

    def melody_fn():
      return mm.Melody(
          steps_per_bar=steps_per_bar, steps_per_quarter=steps_per_quarter)
    melody_extractor_fn = functools.partial(
        mm.extract_melodies,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=max_steps_truncate,
        min_unique_pitches=1,
        ignore_polyphonic_notes=not skip_polyphony,
        pad_end=True)
    super(OneHotMelodyConverter, self).__init__(
        melody_fn,
        melody_extractor_fn,
        mm.MelodyOneHotEncoding(min_pitch, max_pitch + 1),
        add_end_token=add_end_token,
        slice_bars=slice_bars,
        pad_to_total_time=pad_to_total_time,
        steps_per_quarter=steps_per_quarter,
        quarters_per_bar=quarters_per_bar,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        presplit_on_time_changes=presplit_on_time_changes)

  def _to_tensors(self, note_sequence):
    def is_valid(note):
      if (self._valid_programs is not None and
          note.program not in self._valid_programs):
        return False
      return self._min_pitch <= note.pitch <= self._max_pitch
    notes = list(note_sequence.notes)
    del note_sequence.notes[:]
    note_sequence.notes.extend([n for n in notes if is_valid(n)])
    return super(OneHotMelodyConverter, self)._to_tensors(note_sequence)


class OneHotDrumsConverter(LegacyEventListOneHotConverter):
  """Converter for legacy MultiDrumOneHotEncoding with binary inputs.

  Outputs are one-hot encoded labels representing up to 9 possible drum hits at
  a given time, as defined in legacy MultiDrumOneHotEncoding. Inputs are
  optionally the separate drum hits represented by a binary array at each
  timestep.

  Args:
    max_bars: Optional maximum number of bars per extracted drums, before
      slicing.
    slice_bars: Optional size of window to slide over raw Melodies after
      extraction.
    gap_bars: If this many bars or more follow a non-empty drum event, the
      drum track is ended. Disabled when set to 0 or None.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    add_end_token: Whether to add an end token at the end of each sequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    binary_input: Whether to use the separate drum hits represented by a binary
      array at each timestep as the input, instead of the one-hot output
      representation.
  """

  def __init__(self, max_bars=None, slice_bars=None, gap_bars=1.0,
               steps_per_quarter=4, quarters_per_bar=4, pad_to_total_time=False,
               add_end_token=False, max_tensors_per_notesequence=5,
               binary_input=False, presplit_on_time_changes=True):
    steps_per_bar = steps_per_quarter * quarters_per_bar
    max_steps_truncate = steps_per_bar * max_bars if max_bars else None
    self._binary_input = binary_input

    def drumtrack_fn():
      return mm.DrumTrack(
          steps_per_bar=steps_per_bar,
          steps_per_quarter=steps_per_quarter)
    drums_extractor_fn = functools.partial(
        mm.extract_drum_tracks,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=max_steps_truncate,
        pad_end=True)
    super(OneHotDrumsConverter, self).__init__(
        drumtrack_fn,
        drums_extractor_fn,
        mm.MultiDrumOneHotEncoding(),
        add_end_token=add_end_token,
        slice_bars=slice_bars,
        pad_to_total_time=pad_to_total_time,
        steps_per_quarter=steps_per_quarter,
        quarters_per_bar=quarters_per_bar,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        presplit_on_time_changes=presplit_on_time_changes)

    if binary_input:
      # Binary input tensor is log_2 of the output depth after removing the end
      # token plus an additional dimension to represent the "empty" label.
      self._input_depth = (
          int(np.log2(self._output_depth - add_end_token)) + 1 + add_end_token)

  def _to_tensors(self, note_sequence):
    input_seqs, output_seqs = super(OneHotDrumsConverter, self)._to_tensors(
        note_sequence)
    if not self._binary_input:
      return input_seqs, output_seqs
    input_seqs = []
    has_end_token = self.end_token is not None
    for out_seq in output_seqs:
      labels = np.argmax(out_seq, axis=-1)
      in_seq = np.zeros((len(labels), self.input_depth), self.input_dtype)
      for i, l in enumerate(labels):
        if has_end_token and l == self.end_token:
          in_seq[i, -1] = 1
        elif l == 0:
          in_seq[i, -(1 + has_end_token)] = 1
        else:
          for j in range(self.input_depth - has_end_token - 1):
            in_seq[i, j] = (l >> j) % 2
        assert np.any(in_seq[i]), '%s %d' % (labels, i)
      input_seqs.append(in_seq)
    return input_seqs, output_seqs


class TrioConverter(BaseNoteSequenceConverter):
  """Converts to/from 3-part (mel, drums, bass) multi-one-hot events.

  Extracts overlapping segments with melody, drums, and bass (determined by
  program number) and concatenates one-hot tensors from OneHotMelodyConverter
  and OneHotDrumsConverter. Takes the cross products from the sets of
  instruments of each type.

  Args:
    slice_bars: Optional size of window to slide over full converted tensor.
    gap_bars: The number of consecutive empty bars to allow for any given
      instrument. Note that this number is effectively doubled for internal
      gaps.
    max_bars: Optional maximum number of bars per extracted sequence, before
      slicing.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
  """

  class InstrumentType(object):
    UNK = 0
    MEL = 1
    BASS = 2
    DRUMS = 3
    INVALID = 4

  def __init__(
      self, slice_bars=None, gap_bars=2, max_bars=1024, steps_per_quarter=4,
      quarters_per_bar=4, max_tensors_per_notesequence=5):
    self._melody_converter = OneHotMelodyConverter(
        gap_bars=None, steps_per_quarter=steps_per_quarter,
        pad_to_total_time=True, presplit_on_time_changes=False,
        max_tensors_per_notesequence=None)
    self._drums_converter = OneHotDrumsConverter(
        gap_bars=None, steps_per_quarter=steps_per_quarter,
        pad_to_total_time=True, presplit_on_time_changes=False,
        max_tensors_per_notesequence=None)
    self._slice_bars = slice_bars
    self._gap_bars = gap_bars
    self._max_bars = max_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar

    self._split_output_depths = (
        self._melody_converter.output_depth,
        self._melody_converter.output_depth,
        self._drums_converter.output_depth)
    output_depth = sum(self._split_output_depths)

    self._program_map = dict(
        [(i, TrioConverter.InstrumentType.MEL) for i in MEL_PROGRAMS] +
        [(i, TrioConverter.InstrumentType.BASS) for i in BASS_PROGRAMS])

    super(TrioConverter, self).__init__(
        input_depth=output_depth,
        input_dtype=np.bool,
        output_depth=output_depth,
        output_dtype=np.bool,
        end_token=False,
        presplit_on_time_changes=True,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors(self, note_sequence):
    try:
      quantized_sequence = mm.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return []
    except (mm.BadTimeSignatureException, mm.NonIntegerStepsPerBarException,
            mm.NegativeTimeException):
      return []

    total_bars = int(
        np.ceil(quantized_sequence.total_quantized_steps / self._steps_per_bar))
    total_bars = min(total_bars, self._max_bars)

    # Assign an instrument class for each instrument, and compute its coverage.
    # If an instrument has multiple classes, it is considered INVALID.
    instrument_type = np.zeros(MAX_INSTRUMENT_NUMBER + 1, np.uint8)
    coverage = np.zeros((total_bars, MAX_INSTRUMENT_NUMBER + 1), np.bool)
    for note in quantized_sequence.notes:
      i = note.instrument
      if i > MAX_INSTRUMENT_NUMBER:
        tf.logging.warning('Skipping invalid instrument number: %d', i)
        continue
      inferred_type = (
          self.InstrumentType.DRUMS if note.is_drum else
          self._program_map.get(note.program, self.InstrumentType.INVALID))
      if not instrument_type[i]:
        instrument_type[i] = inferred_type
      elif instrument_type[i] != inferred_type:
        instrument_type[i] = self.InstrumentType.INVALID

      start_bar = note.quantized_start_step // self._steps_per_bar
      end_bar = int(np.ceil(note.quantized_end_step / self._steps_per_bar))

      if start_bar >= total_bars:
        continue
      coverage[start_bar:min(end_bar, total_bars), i] = True

    # Group instruments by type.
    instruments_by_type = collections.defaultdict(list)
    for i, type_ in enumerate(instrument_type):
      if type_ not in (self.InstrumentType.UNK, self.InstrumentType.INVALID):
        instruments_by_type[type_].append(i)
    if len(instruments_by_type) < 3:
      # This NoteSequence doesn't have all 3 types.
      return [], []

    def _extract_instrument(note_sequence, instrument):
      extracted_ns = copy.copy(note_sequence)
      del extracted_ns.notes[:]
      extracted_ns.notes.extend(
          n for n in note_sequence.notes if n.instrument == instrument)
      return extracted_ns

    # Encode individual instruments.
    # Set total time so that instruments will be padded correctly.
    note_sequence.total_time = (
        total_bars * self._steps_per_bar *
        60 / note_sequence.tempos[0].qpm / self._steps_per_quarter)
    encoded_instruments = {}
    for i in (instruments_by_type[self.InstrumentType.MEL] +
              instruments_by_type[self.InstrumentType.BASS]):
      _, t = self._melody_converter.to_tensors(
          _extract_instrument(note_sequence, i))
      if t:
        encoded_instruments[i] = t[0]
      else:
        coverage[:, i] = False
    for i in instruments_by_type[self.InstrumentType.DRUMS]:
      _, t = self._drums_converter.to_tensors(
          _extract_instrument(note_sequence, i))
      if t:
        encoded_instruments[i] = t[0]
      else:
        coverage[:, i] = False

    # Fill in coverage gaps up to self._gap_bars.
    og_coverage = coverage.copy()
    for j in range(total_bars):
      coverage[j] = np.any(
          og_coverage[
              max(0, j-self._gap_bars):min(total_bars, j+self._gap_bars) + 1],
          axis=0)

    # Take cross product of instruments from each class and compute combined
    # encodings where they overlap.
    seqs = []
    for grp in itertools.product(
        instruments_by_type[self.InstrumentType.MEL],
        instruments_by_type[self.InstrumentType.BASS],
        instruments_by_type[self.InstrumentType.DRUMS]):
      # Consider an instrument covered within gap_bars from the end if any of
      # the other instruments are. This allows more leniency when re-encoding
      # slices.
      grp_coverage = np.all(coverage[:, grp], axis=1)
      grp_coverage[:self._gap_bars] = np.any(coverage[:self._gap_bars, grp])
      grp_coverage[-self._gap_bars:] = np.any(coverage[-self._gap_bars:, grp])
      for j in range(total_bars - self._slice_bars + 1):
        if np.all(grp_coverage[j:j + self._slice_bars]):
          start_step = j * self._steps_per_bar
          end_step = (j + self._slice_bars) * self._steps_per_bar
          seqs.append(np.concatenate(
              [encoded_instruments[i][start_step:end_step] for i in grp],
              axis=-1))

    return seqs, seqs

  def _to_notesequences(self, samples):
    output_sequences = []
    dim_ranges = np.cumsum(self._split_output_depths)
    for s in samples:
      mel_ns = self._melody_converter.to_notesequences(
          [s[:, :dim_ranges[0]]])[0]
      bass_ns = self._melody_converter.to_notesequences(
          [s[:, dim_ranges[0]:dim_ranges[1]]])[0]
      drums_ns = self._drums_converter.to_notesequences(
          [s[:, dim_ranges[1]:]])[0]

      for n in bass_ns.notes:
        n.instrument = 1
        n.program = ELECTRIC_BASS_PROGRAM
      for n in drums_ns.notes:
        n.instrument = 9

      ns = mel_ns
      ns.notes.extend(bass_ns.notes)
      ns.notes.extend(drums_ns.notes)
      ns.total_time = max(
          mel_ns.total_time, bass_ns.total_time, drums_ns.total_time)
      output_sequences.append(ns)
    return output_sequences


def count_examples(examples_path, note_sequence_converter):
  """Counts the number of examples produced by the converter from files."""
  filenames = tf.gfile.Glob(examples_path)

  num_examples = 0

  for f in filenames:
    tf.logging.info('Counting examples in %s.', f)
    with tf.python_io.tf_record_iterator(f) as reader:
      for note_sequence_str in reader:
        note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)
        seqs, _ = note_sequence_converter.to_tensors(note_sequence)
        num_examples += len(seqs)
  tf.logging.info('Total examples: %d', num_examples)
  return num_examples


def get_dataset(config, file_reader_class=tf.data.TFRecordDataset,
                num_threads=1, is_training=False):
  """Returns a Dataset object that encodes raw serialized NoteSequences."""
  examples_path = (
      config.train_examples_path if is_training else config.eval_examples_path)
  note_sequence_augmenter = (
      config.note_sequence_augmenter if is_training else None)
  note_sequence_converter = config.note_sequence_converter
  note_sequence_converter.is_training = is_training

  filenames = tf.gfile.Glob(examples_path)
  if is_training:
    random.shuffle(filenames)
  tf.logging.info('Reading examples from: %s', filenames)
  reader = file_reader_class(filenames)

  def _remove_pad_fn(padded_seq_1, padded_seq_2, length):
    return padded_seq_1[0:length], padded_seq_2[0:length], length

  dataset = reader
  if note_sequence_augmenter is not None:
    dataset = dataset.map(note_sequence_augmenter.tf_augment)
  dataset = (dataset
             .map(note_sequence_converter.tf_to_tensors,
                  num_parallel_calls=num_threads)
             .flat_map(
                 lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)))
             .map(_remove_pad_fn))
  return dataset
