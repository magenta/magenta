# Copyright 2020 The Magenta Authors.
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

"""MusicVAE data library for hierarchical converters."""
import abc
import random

from magenta.models.music_vae import data
from magenta.pipelines import performance_pipeline
import note_seq
from note_seq import chords_lib
from note_seq import performance_lib
import numpy as np
import tensorflow.compat.v1 as tf

CHORD_SYMBOL = note_seq.NoteSequence.TextAnnotation.CHORD_SYMBOL


def split_performance(performance, steps_per_segment, new_performance_fn,
                      clip_tied_notes=False):
  """Splits a performance into multiple fixed-length segments.

  Args:
    performance: A Performance (or MetricPerformance) object to split.
    steps_per_segment: The number of quantized steps per segment.
    new_performance_fn: A function to create new Performance (or
        MetricPerformance objects). Takes `quantized_sequence` and `start_step`
        arguments.
    clip_tied_notes: If True, clip tied notes across segments by converting each
        segment to NoteSequence and back.

  Returns:
    A list of performance segments.
  """
  segments = []
  cur_segment = new_performance_fn(quantized_sequence=None, start_step=0)
  cur_step = 0
  for e in performance:
    if e.event_type != performance_lib.PerformanceEvent.TIME_SHIFT:
      if cur_step == steps_per_segment:
        # At a segment boundary, note-offs happen before the cutoff.
        # Everything else happens after.
        if e.event_type != performance_lib.PerformanceEvent.NOTE_OFF:
          segments.append(cur_segment)
          cur_segment = new_performance_fn(
              quantized_sequence=None,
              start_step=len(segments) * steps_per_segment)
          cur_step = 0
        cur_segment.append(e)
      else:
        # We're not at a segment boundary.
        cur_segment.append(e)
    else:
      if cur_step + e.event_value <= steps_per_segment:
        # If it's a time shift, but we're still within the current segment,
        # just append to current segment.
        cur_segment.append(e)
        cur_step += e.event_value
      else:
        # If it's a time shift that goes beyond the current segment, possibly
        # split the time shift into two events and create a new segment.
        cur_segment_steps = steps_per_segment - cur_step
        if cur_segment_steps > 0:
          cur_segment.append(performance_lib.PerformanceEvent(
              event_type=performance_lib.PerformanceEvent.TIME_SHIFT,
              event_value=cur_segment_steps))

        segments.append(cur_segment)
        cur_segment = new_performance_fn(
            quantized_sequence=None,
            start_step=len(segments) * steps_per_segment)
        cur_step = 0

        new_segment_steps = e.event_value - cur_segment_steps
        if new_segment_steps > 0:
          cur_segment.append(performance_lib.PerformanceEvent(
              event_type=performance_lib.PerformanceEvent.TIME_SHIFT,
              event_value=new_segment_steps))
          cur_step += new_segment_steps

  segments.append(cur_segment)

  # There may be a final segment with zero duration. If so, remove it.
  if segments and segments[-1].num_steps == 0:
    segments = segments[:-1]

  if clip_tied_notes:
    # Convert each segment to NoteSequence and back to remove notes that are
    # held across segment boundaries.
    for i in range(len(segments)):
      sequence = segments[i].to_sequence()
      if isinstance(segments[i], performance_lib.MetricPerformance):
        # Performance is quantized relative to meter.
        quantized_sequence = note_seq.quantize_note_sequence(
            sequence, steps_per_quarter=segments[i].steps_per_quarter)
      else:
        # Performance is quantized with absolute timing.
        quantized_sequence = note_seq.quantize_note_sequence_absolute(
            sequence, steps_per_second=segments[i].steps_per_second)
      segments[i] = new_performance_fn(
          quantized_sequence=quantized_sequence,
          start_step=segments[i].start_step)
      segments[i].set_length(steps_per_segment)

  return segments


def remove_padding(max_lengths, samples, controls=None):
  """Remove padding."""
  if not max_lengths:
    return samples, controls
  unpadded_samples = [sample.reshape(max_lengths + [-1])
                      for sample in samples]
  unpadded_controls = None
  if controls is not None:
    unpadded_controls = [control.reshape(max_lengths + [-1])
                         for control in controls]
  return unpadded_samples, unpadded_controls


class TooLongError(Exception):
  """Exception for when an array is too long."""
  pass


def pad_with_element(nested_list, max_lengths, element):
  """Pads a nested list of elements up to `max_lengths`.

  For example, `pad_with_element([[0, 1, 2], [3, 4]], [3, 4], 5)` produces
  `[[0, 1, 2, 5], [3, 4, 5, 5], [5, 5, 5, 5]]`.

  Args:
    nested_list: A (potentially nested) list.
    max_lengths: The maximum length at each level of the nested list to pad to.
    element: The element to pad with at the lowest level. If an object, a copy
      is not made, and the same instance will be used multiple times.

  Returns:
    `nested_list`, padded up to `max_lengths` with `element`.

  Raises:
    TooLongError: If any of the nested lists are already longer than the
      maximum length at that level given by `max_lengths`.
  """
  if not max_lengths:
    return nested_list

  max_length = max_lengths[0]
  delta = max_length - len(nested_list)
  if delta < 0:
    raise TooLongError

  if len(max_lengths) == 1:
    return nested_list + [element] * delta
  else:
    return [pad_with_element(l, max_lengths[1:], element)
            for l in nested_list + [[] for _ in range(delta)]]


def pad_with_value(array, length, pad_value):
  """Pad numpy array so that its first dimension is length.

  Args:
    array: A 2D numpy array.
    length: Desired length of the first dimension.
    pad_value: Value to pad with.
  Returns:
    array, padded to shape `[length, array.shape[1]]`.
  Raises:
    TooLongError: If the array is already longer than length.
  """
  if array.shape[0] > length:
    raise TooLongError
  return np.pad(array, ((0, length - array.shape[0]), (0, 0)), 'constant',
                constant_values=pad_value)


def hierarchical_pad_tensors(tensors, sample_size, randomize, max_lengths,
                             end_token, input_depth, output_depth,
                             control_depth, control_pad_token):
  """Converts to tensors and adds hierarchical padding, if needed."""
  sampled_results = data.maybe_sample_items(
      list(zip(*tensors)), sample_size, randomize)
  if sampled_results:
    unpadded_results = data.ConverterTensors(*zip(*sampled_results))
  else:
    unpadded_results = data.ConverterTensors()
  if not max_lengths:
    return unpadded_results

  # TODO(iansimon): The way control tensors are set in ConverterTensors is
  # ugly when using a hierarchical converter. Figure out how to clean this up.

  def _hierarchical_pad(input_, output, control):
    """Pad and flatten hierarchical inputs, outputs, and controls."""
    # Pad empty segments with end tokens and flatten hierarchy.
    input_ = tf.nest.flatten(
        pad_with_element(input_, max_lengths[:-1],
                         data.np_onehot([end_token], input_depth)))
    output = tf.nest.flatten(
        pad_with_element(output, max_lengths[:-1],
                         data.np_onehot([end_token], output_depth)))
    length = np.squeeze(np.array([len(x) for x in input_], np.int32))

    # Pad and concatenate flatten hierarchy.
    input_ = np.concatenate(
        [pad_with_value(x, max_lengths[-1], 0) for x in input_])
    output = np.concatenate(
        [pad_with_value(x, max_lengths[-1], 0) for x in output])

    if np.size(control):
      control = tf.nest.flatten(
          pad_with_element(control, max_lengths[:-1],
                           data.np_onehot([control_pad_token], control_depth)))
      control = np.concatenate(
          [pad_with_value(x, max_lengths[-1], 0) for x in control])

    return input_, output, control, length

  padded_results = []
  for i, o, c, _ in zip(*unpadded_results):
    try:
      padded_results.append(_hierarchical_pad(i, o, c))
    except TooLongError:
      continue

  if padded_results:
    return data.ConverterTensors(*zip(*padded_results))
  else:
    return data.ConverterTensors()


class BaseHierarchicalNoteSequenceConverter(data.BaseNoteSequenceConverter):
  """Base class for data converters for hierarchical sequences.

  Output sequences will be padded hierarchically and flattened if `max_lengths`
  is defined. For example, if `max_lengths = [3, 2, 4]`, `end_token=5`, and the
  underlying `_to_tensors` implementation returns an example
  (before one-hot conversion) [[[1, 5]], [[2, 3, 5]]], `to_tensors` will
  convert it to:
    `[[1, 5, 0, 0], [5, 0, 0, 0],
      [2, 3, 5, 0], [5, 0, 0, 0],
      [5, 0, 0, 0], [5, 0, 0, 0]]`
  If any of the lengths are beyond `max_lengths`, the tensor will be filtered.

  Inheriting classes must implement the following abstract methods:
    -`_to_tensors`
    -`_to_items`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, input_depth, input_dtype, output_depth, output_dtype,
               control_depth=None, control_dtype=np.bool,
               control_pad_token=None, end_token=None,
               max_lengths=None, presplit_on_time_changes=True,
               max_tensors_per_notesequence=None, flat_output=False):
    self._control_pad_token = control_pad_token
    self._max_lengths = [] if max_lengths is None else max_lengths
    if max_lengths and not flat_output:
      length_shape = (np.prod(max_lengths[:-1]),)
    else:
      length_shape = ()
    super(BaseHierarchicalNoteSequenceConverter, self).__init__(
        input_depth=input_depth,
        input_dtype=input_dtype,
        output_depth=output_depth,
        output_dtype=output_dtype,
        control_depth=control_depth,
        control_dtype=control_dtype,
        end_token=end_token,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        length_shape=length_shape,
        presplit_on_time_changes=presplit_on_time_changes)


class MultiInstrumentPerformanceConverter(
    BaseHierarchicalNoteSequenceConverter):
  """Converts to/from multiple-instrument metric performances.

  Attributes:
    num_velocity_bins: Number of velocity bins.
    max_tensors_per_notesequence: The maximum number of outputs to return
        for each NoteSequence.
    hop_size_bars: How many bars each sequence should be.
    chunk_size_bars: Chunk size used for hierarchically decomposing sequence.
    steps_per_quarter: Number of time steps per quarter note.
    quarters_per_bar: Number of quarter notes per bar.
    min_num_instruments: Minimum number of instruments per sequence.
    max_num_instruments: Maximum number of instruments per sequence.
    min_total_events: Minimum total length of all performance tracks, in events.
    max_events_per_instrument: Maximum length of a single-instrument
        performance, in events.
    first_subsequence_only: If True, only use the very first hop and discard all
        sequences longer than the hop size.
    chord_encoding: An instantiated OneHotEncoding object to use for encoding
        chords on which to condition, or None if not conditioning on chords.
    drop_tracks_and_truncate: Randomly drop extra tracks and truncate the event
        sequence.
  """

  def __init__(self,
               num_velocity_bins=0,
               max_tensors_per_notesequence=None,
               hop_size_bars=1,
               chunk_size_bars=1,
               steps_per_quarter=24,
               quarters_per_bar=4,
               min_num_instruments=2,
               max_num_instruments=8,
               min_total_events=8,
               max_events_per_instrument=64,
               min_pitch=performance_lib.MIN_MIDI_PITCH,
               max_pitch=performance_lib.MAX_MIDI_PITCH,
               first_subsequence_only=False,
               chord_encoding=None,
               drop_tracks_and_truncate=False):
    max_shift_steps = (performance_lib.DEFAULT_MAX_SHIFT_QUARTERS *
                       steps_per_quarter)

    self._performance_encoding = note_seq.PerformanceOneHotEncoding(
        num_velocity_bins=num_velocity_bins,
        max_shift_steps=max_shift_steps,
        min_pitch=min_pitch,
        max_pitch=max_pitch)
    self._chord_encoding = chord_encoding

    self._num_velocity_bins = num_velocity_bins
    self._hop_size_bars = hop_size_bars
    self._chunk_size_bars = chunk_size_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._min_num_instruments = min_num_instruments
    self._max_num_instruments = max_num_instruments
    self._min_total_events = min_total_events
    self._max_events_per_instrument = max_events_per_instrument
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch
    self._first_subsequence_only = first_subsequence_only
    self._drop_tracks_and_truncate = drop_tracks_and_truncate

    self._max_num_chunks = hop_size_bars // chunk_size_bars
    self._max_steps_truncate = (
        steps_per_quarter * quarters_per_bar * hop_size_bars)

    # Each encoded track will begin with a program specification token
    # (with one extra program for drums).
    num_program_tokens = (
        note_seq.MAX_MIDI_PROGRAM - note_seq.MIN_MIDI_PROGRAM + 2)
    end_token = self._performance_encoding.num_classes + num_program_tokens
    depth = end_token + 1

    max_lengths = [
        self._max_num_chunks, max_num_instruments, max_events_per_instrument]
    if chord_encoding is None:
      control_depth = 0
      control_pad_token = None
    else:
      control_depth = chord_encoding.num_classes
      control_pad_token = chord_encoding.encode_event(note_seq.NO_CHORD)

    super(MultiInstrumentPerformanceConverter, self).__init__(
        input_depth=depth,
        input_dtype=np.bool,
        output_depth=depth,
        output_dtype=np.bool,
        control_depth=control_depth,
        control_dtype=np.bool,
        control_pad_token=control_pad_token,
        end_token=end_token,
        max_lengths=max_lengths,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _quantized_subsequence_to_tensors(self, quantized_subsequence):
    # Reject sequences with out-of-range pitches.
    if any(note.pitch < self._min_pitch or note.pitch > self._max_pitch
           for note in quantized_subsequence.notes):
      return [], []

    # Extract all instruments.
    tracks, _ = performance_pipeline.extract_performances(
        quantized_subsequence,
        max_steps_truncate=self._max_steps_truncate,
        num_velocity_bins=self._num_velocity_bins,
        split_instruments=True)

    if (self._drop_tracks_and_truncate and
        len(tracks) > self._max_num_instruments):
      tracks = random.sample(tracks, self._max_num_instruments)

    # Reject sequences with too few instruments.
    if not (self._min_num_instruments <= len(tracks) <=
            self._max_num_instruments):
      return [], []

    # Sort tracks by program, with drums at the end.
    tracks = sorted(tracks, key=lambda t: (t.is_drum, t.program))

    chunk_size_steps = self._steps_per_bar * self._chunk_size_bars
    chunks = [[] for _ in range(self._max_num_chunks)]

    total_length = 0

    for track in tracks:
      # Make sure the track is the proper number of time steps.
      track.set_length(self._max_steps_truncate)

      # Split this track into chunks.
      def new_performance(quantized_sequence, start_step, track=track):
        steps_per_quarter = (
            self._steps_per_quarter if quantized_sequence is None else None)
        return performance_lib.MetricPerformance(
            quantized_sequence=quantized_sequence,
            steps_per_quarter=steps_per_quarter,
            start_step=start_step,
            num_velocity_bins=self._num_velocity_bins,
            program=track.program, is_drum=track.is_drum)
      track_chunks = split_performance(
          track, chunk_size_steps, new_performance, clip_tied_notes=True)

      assert len(track_chunks) == self._max_num_chunks

      if self._drop_tracks_and_truncate:
        for i in range(len(track_chunks)):
          track_chunks[i].truncate(self._max_events_per_instrument - 2)

      track_chunk_lengths = [len(track_chunk) for track_chunk in track_chunks]
      # Each track chunk needs room for program token and end token.
      if not all(l <= self._max_events_per_instrument - 2
                 for l in track_chunk_lengths):
        return [], []
      if not all(
          note_seq.MIN_MIDI_PROGRAM <= t.program <= note_seq.MAX_MIDI_PROGRAM
          for t in track_chunks
          if not t.is_drum):
        return [], []

      total_length += sum(track_chunk_lengths)

      # Aggregate by chunk.
      for i, track_chunk in enumerate(track_chunks):
        chunks[i].append(track_chunk)

    # Reject sequences that are too short (in events).
    if total_length < self._min_total_events:
      return [], []

    num_programs = note_seq.MAX_MIDI_PROGRAM - note_seq.MIN_MIDI_PROGRAM + 1

    chunk_tensors = []
    chunk_chord_tensors = []

    for chunk_tracks in chunks:
      track_tensors = []

      for track in chunk_tracks:
        # Add a special token for program at the beginning of each track.
        track_tokens = [self._performance_encoding.num_classes + (
            num_programs if track.is_drum else track.program)]
        # Then encode the performance events.
        for event in track:
          track_tokens.append(self._performance_encoding.encode_event(event))
        # Then add the end token.
        track_tokens.append(self.end_token)

        encoded_track = data.np_onehot(
            track_tokens, self.output_depth, self.output_dtype)
        track_tensors.append(encoded_track)

      if self._chord_encoding:
        # Extract corresponding chords for each track. The chord sequences may
        # be different for different tracks even though the underlying chords
        # are the same, as the performance event times will generally be
        # different.
        try:
          track_chords = chords_lib.event_list_chords(
              quantized_subsequence, chunk_tracks)
        except chords_lib.CoincidentChordsError:
          return [], []

        track_chord_tensors = []

        try:
          # Chord encoding for all tracks is inside this try block. If any
          # track fails we need to skip the whole subsequence.

          for chords in track_chords:
            # Start with a pad token corresponding to the track program token.
            track_chord_tokens = [self._control_pad_token]
            # Then encode the chords.
            for chord in chords:
              track_chord_tokens.append(
                  self._chord_encoding.encode_event(chord))
            # Then repeat the final chord for the track end token.
            track_chord_tokens.append(track_chord_tokens[-1])

            encoded_track_chords = data.np_onehot(
                track_chord_tokens, self.control_depth, self.control_dtype)
            track_chord_tensors.append(encoded_track_chords)

        except (note_seq.ChordSymbolError, note_seq.ChordEncodingError):
          return [], []

        chunk_chord_tensors.append(track_chord_tensors)

      chunk_tensors.append(track_tensors)

    return chunk_tensors, chunk_chord_tensors

  def _to_tensors_fn(self, note_sequence):
    # Performance sequences require sustain to be correctly interpreted.
    note_sequence = note_seq.apply_sustain_control_changes(note_sequence)

    if self._chord_encoding and not any(
        ta.annotation_type == CHORD_SYMBOL
        for ta in note_sequence.text_annotations):
      try:
        # Quantize just for the purpose of chord inference.
        # TODO(iansimon): Allow chord inference in unquantized sequences.
        quantized_sequence = note_seq.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
        if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
            self._steps_per_bar):
          return data.ConverterTensors()

        # Infer chords in quantized sequence.
        note_seq.infer_chords_for_sequence(quantized_sequence)

      except (note_seq.BadTimeSignatureError,
              note_seq.NonIntegerStepsPerBarError, note_seq.NegativeTimeError,
              note_seq.ChordInferenceError):
        return data.ConverterTensors()

      # Copy inferred chords back to original sequence.
      for qta in quantized_sequence.text_annotations:
        if qta.annotation_type == CHORD_SYMBOL:
          ta = note_sequence.text_annotations.add()
          ta.annotation_type = CHORD_SYMBOL
          ta.time = qta.time
          ta.text = qta.text

    if note_sequence.tempos:
      quarters_per_minute = note_sequence.tempos[0].qpm
    else:
      quarters_per_minute = note_seq.DEFAULT_QUARTERS_PER_MINUTE
    quarters_per_bar = self._steps_per_bar / self._steps_per_quarter
    hop_size_quarters = quarters_per_bar * self._hop_size_bars
    hop_size_seconds = 60.0 * hop_size_quarters / quarters_per_minute

    # Split note sequence by bar hop size (in seconds).
    subsequences = note_seq.split_note_sequence(note_sequence, hop_size_seconds)

    if self._first_subsequence_only and len(subsequences) > 1:
      return data.ConverterTensors()

    sequence_tensors = []
    sequence_chord_tensors = []

    for subsequence in subsequences:
      # Quantize this subsequence.
      try:
        quantized_subsequence = note_seq.quantize_note_sequence(
            subsequence, self._steps_per_quarter)
        if (note_seq.steps_per_bar_in_quantized_sequence(quantized_subsequence)
            != self._steps_per_bar):
          return data.ConverterTensors()
      except (note_seq.BadTimeSignatureError,
              note_seq.NonIntegerStepsPerBarError, note_seq.NegativeTimeError):
        return data.ConverterTensors()

      # Convert the quantized subsequence to tensors.
      tensors, chord_tensors = self._quantized_subsequence_to_tensors(
          quantized_subsequence)
      if tensors:
        sequence_tensors.append(tensors)
        if self._chord_encoding:
          sequence_chord_tensors.append(chord_tensors)

    tensors = data.ConverterTensors(
        inputs=sequence_tensors, outputs=sequence_tensors,
        controls=sequence_chord_tensors)
    return hierarchical_pad_tensors(tensors, self.max_tensors_per_notesequence,
                                    self.is_training, self._max_lengths,
                                    self.end_token, self.input_depth,
                                    self.output_depth, self.control_depth,
                                    self._control_pad_token)

  def to_tensors(self, note_sequence):
    return data.split_process_and_combine(note_sequence,
                                          self._presplit_on_time_changes,
                                          self.max_tensors_per_notesequence,
                                          self.is_training, self._to_tensors_fn)

  def _to_single_notesequence(self, samples, controls):
    qpm = note_seq.DEFAULT_QUARTERS_PER_MINUTE
    seconds_per_step = 60.0 / (self._steps_per_quarter * qpm)
    chunk_size_steps = self._steps_per_bar * self._chunk_size_bars

    seq = note_seq.NoteSequence()
    seq.tempos.add().qpm = qpm
    seq.ticks_per_quarter = note_seq.STANDARD_PPQ

    tracks = [[] for _ in range(self._max_num_instruments)]
    all_timed_chords = []

    for chunk_index, encoded_chunk in enumerate(samples):
      chunk_step_offset = chunk_index * chunk_size_steps

      # Decode all tracks in this chunk into performance representation.
      # We don't immediately convert to NoteSequence as we first want to group
      # by track and concatenate.
      for instrument, encoded_track in enumerate(encoded_chunk):
        track_tokens = np.argmax(encoded_track, axis=-1)

        # Trim to end token.
        if self.end_token in track_tokens:
          idx = track_tokens.tolist().index(self.end_token)
          track_tokens = track_tokens[:idx]

        # Handle program token. If there are extra program tokens, just use the
        # first one.
        program_tokens = [token for token in track_tokens
                          if token >= self._performance_encoding.num_classes]
        track_token_indices = [idx for idx, t in enumerate(track_tokens)
                               if t < self._performance_encoding.num_classes]
        track_tokens = [track_tokens[idx] for idx in track_token_indices]
        if not program_tokens:
          program = 0
          is_drum = False
        else:
          program = program_tokens[0] - self._performance_encoding.num_classes
          if program == note_seq.MAX_MIDI_PROGRAM + 1:
            # This is the drum program.
            program = 0
            is_drum = True
          else:
            is_drum = False

        # Decode the tokens into a performance track.
        track = performance_lib.MetricPerformance(
            quantized_sequence=None,
            steps_per_quarter=self._steps_per_quarter,
            start_step=0,
            num_velocity_bins=self._num_velocity_bins,
            program=program,
            is_drum=is_drum)
        for token in track_tokens:
          track.append(self._performance_encoding.decode_event(token))

        if controls is not None:
          # Get the corresponding chord and time for each event in the track.
          # This is a little tricky since we removed extraneous program tokens
          # when constructing the track.
          track_chord_tokens = np.argmax(controls[chunk_index][instrument],
                                         axis=-1)
          track_chord_tokens = [track_chord_tokens[idx]
                                for idx in track_token_indices]
          chords = [self._chord_encoding.decode_event(token)
                    for token in track_chord_tokens]
          chord_times = [(chunk_step_offset + step) * seconds_per_step
                         for step in track.steps if step < chunk_size_steps]
          all_timed_chords += zip(chord_times, chords)

        # Make sure the track has the proper length in time steps.
        track.set_length(chunk_size_steps)

        # Aggregate by instrument.
        tracks[instrument].append(track)

    # Concatenate all of the track chunks for each instrument.
    for instrument, track_chunks in enumerate(tracks):
      if track_chunks:
        track = track_chunks[0]
        for t in track_chunks[1:]:
          for e in t:
            track.append(e)

      track_seq = track.to_sequence(instrument=instrument, qpm=qpm)
      seq.notes.extend(track_seq.notes)

    # Set total time.
    if seq.notes:
      seq.total_time = max(note.end_time for note in seq.notes)

    if self._chord_encoding:
      # Sort chord times from all tracks and add to the sequence.
      all_chord_times, all_chords = zip(*sorted(all_timed_chords))
      chords_lib.add_chords_to_sequence(seq, all_chords, all_chord_times)

    return seq

  def from_tensors(self, samples, controls=None):
    samples, controls = remove_padding(self._max_lengths, samples, controls)
    output_sequences = []
    for i in range(len(samples)):
      seq = self._to_single_notesequence(
          samples[i], controls[i]
          if self._chord_encoding and controls is not None else None)
      output_sequences.append(seq)
    return output_sequences
