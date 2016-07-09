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

import collections
import math


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


class BadNoteException(Exception):
  pass


class BadTimeSignatureException(Exception):
  pass


def is_power_of_2(x):
  return x and not (x & (x - 1))


class QuantizedSequence(object):
  Note = collections.namedtuple('Note', ['pitch', 'velocity', 'start', 'end', 'instrument', 'program'])
  TimeSignature = collections.namedtuple('TimeSignature', ['numerator', 'denominator'])
  KeySignature = collections.namedtuple('KeySignature', ['key', 'mode'])

  def __init__(self):
    self._reset()

  def _reset(self):
    self.tracks = {}
    self.bpm = 120.0
    self.time_signature = self.TimeSignature(4, 4)  # numerator, denominator
    self.key_signature = self.KeySignature(0, 0)  # key, mode
    self.steps_per_beat = 4

  def from_note_sequence(self, note_sequence, steps_per_beat):
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
      ignore_polyphonic_notes: If true, any notes that come before or land on
          an already added note's start step will be ignored. If false,
          PolyphonicMelodyException will be raised.

    Raises:
      PolyphonicMelodyException: If any of the notes start on the same step when
      quantized and ignore_polyphonic_notes is False.
    """
    self._reset()

    self.steps_per_beat = steps_per_beat

    if note_sequence.time_signatures:
      self.time_signature = self.TimeSignature(note_sequence.time_signatures[0].numerator, note_sequence.time_signatures[0].denominator)

    if not is_power_of_2(self.time_signature.denominator):
      raise BadTimeSignatureException('Denominator is not a power of 2. Time signature: %d/%d' % (self.time_signature.numerator, self.time_signature.denominator))

    if note_sequence.key_signatures:
      self.key_signature = self.KeySignature(note_sequence.key_signatures[0].key, note_sequence.key_signatures[0].mode)

    bpm = note_sequence.tempos[0].bpm if note_sequence.tempos else 120.0

    # Compute quantization steps per second.
    steps_per_second = steps_per_beat * bpm / 60.0

    quantize = lambda x: int(math.ceil(x - QUANTIZE_CUTOFF))

    for note in note_sequence.notes:
      # Quantize the start and end times of the note.
      start_step = quantize(note.start_time * steps_per_second)
      end_step = quantize(note.end_time * steps_per_second)
      if end_step == start_step:
        end_step += 1

      # Do not allow notes to start or end in negative time.
      if start_step < 0 or end_step < 0:
        raise BadNoteException(
            'Got negative note time: start_step = %s, start_step = %s'
            % (start_step, end_step))

      if note.instrument not in self.tracks:
        self.tracks[note.instrument] = []
      self.tracks[note.instrument].append(self.Note(pitch=note.pitch, velocity=note.velocity, start=start_step, end=end_step, instrument=note.instrument, program=note.program))