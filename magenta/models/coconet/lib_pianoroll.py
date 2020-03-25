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

# Lint as: python3
"""Utilities for converting between NoteSequences and pianorolls."""
import numpy as np
import pretty_midi
import tensorflow.compat.v1 as tf


class PitchOutOfEncodeRangeError(Exception):
  """Exception for when pitch of note is out of encodings range."""
  pass


def get_pianoroll_encoder_decoder(hparams):
  encoder_decoder = PianorollEncoderDecoder(
      shortest_duration=hparams.shortest_duration,
      min_pitch=hparams.min_pitch,
      max_pitch=hparams.max_pitch,
      separate_instruments=hparams.separate_instruments,
      num_instruments=hparams.num_instruments,
      quantization_level=hparams.quantization_level)
  return encoder_decoder


class PianorollEncoderDecoder(object):
  """Encodes list/array format piece into pianorolls and decodes into midi."""

  qpm = 120
  # Oboe, English horn, clarinet, bassoon, sounds better on timidity.
  programs = [69, 70, 72, 71]

  def __init__(self,
               shortest_duration=0.125,
               min_pitch=36,
               max_pitch=81,
               separate_instruments=True,
               num_instruments=None,
               quantization_level=None):
    assert num_instruments is not None
    self.shortest_duration = shortest_duration
    self.min_pitch = min_pitch
    self.max_pitch = max_pitch
    self.separate_instruments = separate_instruments
    self.num_instruments = num_instruments
    self.quantization_level = quantization_level
    if quantization_level is None:
      quantization_level = self.shortest_duration

  def encode(self, sequence):
    """Encode sequence into pianoroll."""
    # Sequence can either be a 2D numpy array or a list of lists.
    if (isinstance(sequence, np.ndarray) and sequence.ndim == 2) or (
        isinstance(sequence, list) and
        isinstance(sequence[0], (list, tuple))):
      # If sequence is an numpy array should have shape (time, num_instruments).
      if (isinstance(sequence, np.ndarray) and
          sequence.shape[-1] != self.num_instruments):
        raise ValueError('Last dim of sequence should equal num_instruments.')
      if isinstance(sequence, np.ndarray) and not self.separate_instruments:
        raise ValueError('Only use numpy array if instruments are separated.')
      sequence = list(sequence)
      return self.encode_list_of_lists(sequence)
    else:
      raise TypeError('Type %s not yet supported.' % type(sequence))

  def encode_list_of_lists(self, sequence):
    """Encode 2d array or list of lists of midi note numbers into pianoroll."""
    # step_size larger than 1 means some notes will be skipped over.
    step_size = self.quantization_level / self.shortest_duration
    if not step_size.is_integer():
      raise ValueError(
          'quantization %r should be multiple of shortest_duration %r.' %
          (self.quantization_level, self.shortest_duration))
    step_size = int(step_size)

    if not (len(sequence) / step_size).is_integer():
      raise ValueError('step_size %r should fully divide length of seq %r.' %
                       (step_size, len(sequence)))
    tt = int(len(sequence) / step_size)
    pp = self.max_pitch - self.min_pitch + 1
    if self.separate_instruments:
      roll = np.zeros((tt, pp, self.num_instruments))
    else:
      roll = np.zeros((tt, pp, 1))
    for raw_t, chord in enumerate(sequence):
      # Only takes time steps that are on the quantization grid.
      if raw_t % step_size != 0:
        continue
      t = int(raw_t / step_size)
      for i in range(self.num_instruments):
        if i > len(chord):
          # Some instruments are silence in this time step.
          if self.separate_instruments:
            raise ValueError(
                'If instruments are separated must have all encoded.')
          continue
        pitch = chord[i]
        # Silences are sometimes encoded as NaN when instruments are separated.
        if np.isnan(pitch):
          continue
        if pitch > self.max_pitch or pitch < self.min_pitch:
          raise PitchOutOfEncodeRangeError(
              '%r is out of specified range [%r, %r].' % (pitch, self.min_pitch,
                                                          self.max_pitch))
        p = pitch - self.min_pitch
        if not float(p).is_integer():
          raise ValueError('Non integer pitches not yet supported.')
        p = int(p)
        if self.separate_instruments:
          roll[t, p, i] = 1
        else:
          roll[t, p, 0] = 0
    return roll

  def decode_to_midi(self, pianoroll):
    """Decodes pianoroll into midi."""
    # NOTE: Assumes four separate instruments ordered from high to low.
    midi_data = pretty_midi.PrettyMIDI()
    duration = self.qpm / 60 * self.shortest_duration
    tt, pp, ii = pianoroll.shape
    for i in range(ii):
      notes = []
      for p in range(pp):
        for t in range(tt):
          if pianoroll[t, p, i]:
            notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=self.min_pitch + p,
                    start=t * duration,
                    end=(t + 1) * duration))
      notes = merge_held(notes)

      instrument = pretty_midi.Instrument(program=self.programs[i] - 1)
      instrument.notes.extend(notes)
      midi_data.instruments.append(instrument)
    return midi_data

  def encode_midi_melody_to_pianoroll(self, midi):
    """Encodes midi into pianorolls."""
    if len(midi.instruments) != 1:
      raise ValueError('Only one melody/instrument allowed, %r given.' %
                       (len(midi.instruments)))
    unused_tempo_change_times, tempo_changes = midi.get_tempo_changes()
    assert len(tempo_changes) == 1
    fs = 4
    # Returns matrix of shape (128, time) with summed velocities.
    roll = midi.get_piano_roll(fs=fs)  # 16th notes
    roll = np.where(roll > 0, 1, 0)
    tf.logging.debug('Roll shape: %s', roll.shape)
    roll = roll.T
    tf.logging.debug('Roll argmax: %s', np.argmax(roll, 1))
    return roll

  def encode_midi_to_pianoroll(self, midi, requested_shape):
    """Encodes midi into pianorolls according to requested_shape."""
    # TODO(annahuang): Generalize to not requiring a requested shape.
    # TODO(annahuang): Assign instruments to SATB according to range of notes.
    bb, tt, pp, ii = requested_shape
    if not midi.instruments:
      return np.zeros(requested_shape)
    elif len(midi.instruments) > ii:
      raise ValueError('Max number of instruments allowed %d < %d given.' % ii,
                       (len(midi.instruments)))
    unused_tempo_change_times, tempo_changes = midi.get_tempo_changes()
    assert len(tempo_changes) == 1

    tf.logging.debug('# of instr %d', len(midi.instruments))
    # Encode each instrument separately.
    instr_rolls = [
        self.get_instr_pianoroll(instr, requested_shape)
        for instr in midi.instruments
    ]
    if len(instr_rolls) != ii:
      for unused_i in range(ii - len(instr_rolls)):
        instr_rolls.append(np.zeros_like(instr_rolls[0]))

    max_tt = np.max([roll.shape[0] for roll in instr_rolls])
    if tt < max_tt:
      tf.logging.warning(
          'WARNING: input midi is a longer sequence then the requested'
          'size (%d > %d)', max_tt, tt)
    elif max_tt < tt:
      max_tt = tt
    pianorolls = np.zeros((bb, max_tt, pp, ii))
    for i, roll in enumerate(instr_rolls):
      pianorolls[:, :roll.shape[0], :, i] = np.tile(roll[:, :], (bb, 1, 1))
    tf.logging.debug('Requested roll shape: %s', requested_shape)
    tf.logging.debug('Roll argmax: %s',
                     np.argmax(pianorolls, axis=2) + self.min_pitch)
    return pianorolls

  def get_instr_pianoroll(self, midi_instr, requested_shape):
    """Returns midi_instr as 2D (time, model pitch_range) pianoroll."""
    pianoroll = np.zeros(requested_shape[1:-1])
    if not midi_instr.notes:
      return pianoroll
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(midi_instr)
    # TODO(annahuang): Sampling frequency is dataset dependent.
    fs = 4
    # Returns matrix of shape (128, time) with summed velocities.
    roll = midi.get_piano_roll(fs=fs)
    roll = np.where(roll > 0, 1, 0)
    roll = roll.T
    out_of_range_pitch_count = (
        np.sum(roll[:, self.max_pitch + 1:]) + np.sum(roll[:, :self.min_pitch]))
    if out_of_range_pitch_count > 0:
      raise ValueError(
          '%d pitches out of the range (%d, %d) the model was trained on.' %
          (out_of_range_pitch_count, self.min_pitch, self.max_pitch))
    roll = roll[:, self.min_pitch:self.max_pitch + 1]
    return roll


def merge_held(notes):
  """Combine repeated notes into one sustained note."""
  notes = list(notes)
  i = 1
  while i < len(notes):
    if (notes[i].pitch == notes[i - 1].pitch and
        notes[i].start == notes[i - 1].end):
      notes[i - 1].end = notes[i].end
      del notes[i]
    else:
      i += 1
  return notes
