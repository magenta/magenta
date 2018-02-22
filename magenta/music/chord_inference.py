# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Chord inference for NoteSequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

# internal imports
import numpy as np
import tensorflow as tf

from magenta.music import constants
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

# MIDI programs that typically sound unpitched.
_UNPITCHED_PROGRAMS = (
    list(range(96, 104)) + list(range(112, 120)) + list(range(120, 128)))

# Names of pitch classes to use (mostly ignoring spelling).
_PITCH_CLASS_NAMES = [
    'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Pitch classes in a key (rooted at zero).
_KEY_PITCHES = [0, 2, 4, 5, 7, 9, 11]

# Pitch classes in each chord kind (rooted at zero).
_CHORD_KIND_PITCHES = {
    '': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}
_CHORD_KINDS = _CHORD_KIND_PITCHES.keys()

# All usable chords, including no-chord.
_CHORDS = [constants.NO_CHORD] + list(
    itertools.product(range(12), _CHORD_KINDS))

# All key-chord pairs.
_KEY_CHORDS = list(itertools.product(range(12), _CHORDS))

# Maximum length of chord sequence to infer.
_MAX_NUM_CHORDS = 1000

# Mapping from time signature to number of chords to infer per bar.
_DEFAULT_TIME_SIGNATURE_CHORDS_PER_BAR = {
    (2, 2): 1,
    (2, 4): 1,
    (3, 4): 1,
    (4, 4): 2,
    (6, 8): 2,
}


def _key_chord_distribution(chord_pitch_out_of_key_prob):
  """Probability distribution over chords for each key."""
  num_pitches_in_key = np.zeros([12, len(_CHORDS)], dtype=np.int32)
  num_pitches_out_of_key = np.zeros([12, len(_CHORDS)], dtype=np.int32)

  # For each key and chord, compute the number of chord notes in the key and the
  # number of chord notes outside the key.
  for key in range(12):
    key_pitches = set((key + offset) % 12 for offset in _KEY_PITCHES)
    for i, chord in enumerate(_CHORDS[1:]):
      root, kind = chord
      chord_pitches = set((root + offset) % 12
                          for offset in _CHORD_KIND_PITCHES[kind])
      num_pitches_in_key[key, i + 1] = len(chord_pitches & key_pitches)
      num_pitches_out_of_key[key, i + 1] = len(chord_pitches - key_pitches)

  # Compute the probability of each chord under each key, normalizing to sum to
  # one for each key.
  mat = ((1 - chord_pitch_out_of_key_prob) ** num_pitches_in_key *
         chord_pitch_out_of_key_prob ** num_pitches_out_of_key)
  mat /= mat.sum(axis=1)[:, np.newaxis]
  return mat


def _key_chord_transition_distribution(
    key_chord_distribution, key_change_prob, chord_change_prob):
  """Transition distribution between key-chord pairs."""
  mat = np.zeros([len(_KEY_CHORDS), len(_KEY_CHORDS)])

  for i, key_chord_1 in enumerate(_KEY_CHORDS):
    key_1, chord_1 = key_chord_1
    chord_index_1 = i % len(_CHORDS)

    for j, key_chord_2 in enumerate(_KEY_CHORDS):
      key_2, chord_2 = key_chord_2
      chord_index_2 = j % len(_CHORDS)

      if key_1 != key_2:
        # Key change. Chord probability depends only on key and not previous
        # chord.
        mat[i, j] = (key_change_prob / 11)
        mat[i, j] *= key_chord_distribution[key_2, chord_index_2]

      else:
        # No key change.
        mat[i, j] = 1 - key_change_prob
        if chord_1 != chord_2:
          # Chord probability depends on key, but we have to redistribute the
          # probability mass on the previous chord since we know the chord
          # changed.
          mat[i, j] *= (
              chord_change_prob * (
                  key_chord_distribution[key_2, chord_index_2] +
                  key_chord_distribution[key_2, chord_index_1] / (len(_CHORDS) -
                                                                  1)))
        else:
          # No chord change.
          mat[i, j] *= 1 - chord_change_prob

  return mat


def _chord_pitch_vectors():
  """Unit vectors over pitch classes for all chords."""
  x = np.zeros([len(_CHORDS), 12])
  for i, chord in enumerate(_CHORDS[1:]):
    root, kind = chord
    for offset in _CHORD_KIND_PITCHES[kind]:
      x[i + 1, (root + offset) % 12] = 1
  x[1:, :] /= np.linalg.norm(x[1:, :], axis=1)[:, np.newaxis]
  return x


def sequence_note_pitch_vectors(sequence, seconds_per_frame):
  """Compute pitch class vectors for fixed-size frames across a sequence.

  Args:
    sequence: The NoteSequence for which to compute pitch class vectors.
    seconds_per_frame: The size of the frame corresponding to each pitch class
        vector, in seconds.

  Returns:
    A numpy array with shape `[num_frames, 12]` where each row is a unit-
    normalized pitch class vector for the corresponding frame in `sequence`.
  """
  num_frames = int(math.ceil(sequence.total_time / seconds_per_frame))
  x = np.zeros([num_frames, 12])

  for note in sequence.notes:
    if note.is_drum:
      continue
    if note.program in _UNPITCHED_PROGRAMS:
      continue

    start_frame = int(math.floor(note.start_time / seconds_per_frame))
    end_frame = int(math.ceil(note.end_time / seconds_per_frame)) - 1
    pitch_class = note.pitch % 12

    start_frame = min(start_frame, num_frames - 1)
    end_frame = max(end_frame, 0)

    if start_frame >= end_frame:
      x[start_frame, pitch_class] += note.end_time - note.start_time
    else:
      x[start_frame, pitch_class] += (
          (start_frame + 1) * seconds_per_frame - note.start_time)
      for frame in range(start_frame + 1, end_frame):
        x[frame, pitch_class] += seconds_per_frame
      x[end_frame, pitch_class] += note.end_time - end_frame * seconds_per_frame

  x_norm = np.linalg.norm(x, axis=1)
  nonzero_frames = x_norm > 0
  x[nonzero_frames, :] /= x_norm[nonzero_frames, np.newaxis]

  return x


def _chord_frame_log_likelihood(note_pitch_vectors, chord_note_concentration):
  """Log-likelihood of observing each frame of note pitches under each chord."""
  return chord_note_concentration * np.dot(note_pitch_vectors,
                                           _chord_pitch_vectors().T)


def _key_chord_viterbi(chord_frame_loglik,
                       key_chord_loglik,
                       key_chord_transition_loglik):
  """Use the Viterbi algorithm to infer a sequence of key-chord pairs."""
  num_frames, num_chords = chord_frame_loglik.shape
  num_key_chords = len(key_chord_transition_loglik)

  loglik_matrix = np.zeros([num_frames, num_key_chords])
  path_matrix = np.zeros([num_frames, num_key_chords], dtype=np.int32)

  # Initialize with a uniform distribution over keys.
  for i, key_chord in enumerate(_KEY_CHORDS):
    key, unused_chord = key_chord
    chord_index = i % len(_CHORDS)
    loglik_matrix[0, i] = (
        -np.log(12) + key_chord_loglik[key, chord_index] +
        chord_frame_loglik[0, chord_index])

  for frame in range(1, num_frames):
    # At each frame, store the log-likelihood of the best sequence ending in
    # each key-chord pair, along with the index of the parent key-chord pair
    # from the previous frame.
    mat = (np.tile(loglik_matrix[frame - 1][:, np.newaxis],
                   [1, num_key_chords]) +
           key_chord_transition_loglik)
    path_matrix[frame, :] = mat.argmax(axis=0)
    loglik_matrix[frame, :] = (
        mat[path_matrix[frame, :], range(num_key_chords)] +
        np.tile(chord_frame_loglik[frame], 12))

  # Reconstruct the most likely sequence of key-chord pairs.
  path = [np.argmax(loglik_matrix[-1])]
  for frame in range(num_frames, 1, -1):
    path.append(path_matrix[frame - 1, path[-1]])

  return [(index // num_chords, _CHORDS[index % num_chords])
          for index in path[::-1]]


class ChordInferenceException(Exception):
  pass


class SequenceAlreadyHasChordsException(ChordInferenceException):
  pass


class UncommonTimeSignatureException(ChordInferenceException):
  pass


class NonIntegerStepsPerChordException(ChordInferenceException):
  pass


class EmptySequenceException(ChordInferenceException):
  pass


class SequenceTooLongException(ChordInferenceException):
  pass


def infer_chords_for_sequence(quantized_sequence,
                              chords_per_bar=None,
                              key_change_prob=0.001,
                              chord_change_prob=0.5,
                              chord_pitch_out_of_key_prob=0.01,
                              chord_note_concentration=100.0):
  """Infer chords for a quantized NoteSequence using the Viterbi algorithm.

  This uses some heuristics to infer chords for a quantized NoteSequence. At
  each chord position a key and chord will be inferred, and the chords will be
  added (as text annotations) to the sequence.

  Args:
    quantized_sequence: The quantized NoteSequence for which to infer chords.
        This NoteSequence will be modified in place.
    chords_per_bar: The number of chords per bar to infer. If None, use a
        default number of chords based on the time signature of
        `quantized_sequence`.
    key_change_prob: Probability of a key change between two adjacent frames.
    chord_change_prob: Probability of a chord change between two adjacent
        frames.
    chord_pitch_out_of_key_prob: Probability of a pitch in a chord not belonging
        to the current key.
    chord_note_concentration: Concentration parameter for the distribution of
        observed pitches played over a chord. At zero, all pitches are equally
        likely. As concentration increases, observed pitches must match the
        chord pitches more closely.

  Raises:
    SequenceAlreadyHasChordsException: If `quantized_sequence` already has
        chords.
    UncommonTimeSignatureException: If `chords_per_bar` is not specified and
        `quantized_sequence` has an uncommon time signature.
    NonIntegerStepsPerChordException: If the number of quantized steps per chord
        is not an integer.
    EmptySequenceException: If `quantized_sequence` is empty.
    SequenceTooLongException: If the number of chords to be inferred is too
        large.
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
  for ta in quantized_sequence.text_annotations:
    if ta.annotation_type == music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL:
      raise SequenceAlreadyHasChordsException(
          'NoteSequence already has chord(s): %s' % ta.text)

  if chords_per_bar is None:
    time_signature = (quantized_sequence.time_signatures[0].numerator,
                      quantized_sequence.time_signatures[0].denominator)
    if time_signature not in _DEFAULT_TIME_SIGNATURE_CHORDS_PER_BAR:
      raise UncommonTimeSignatureException(
          'No default chords per bar for time signature: (%d, %d)' %
          time_signature)
    chords_per_bar = _DEFAULT_TIME_SIGNATURE_CHORDS_PER_BAR[time_signature]

  # Determine the number of seconds (and steps) each chord is held.
  steps_per_bar_float = sequences_lib.steps_per_bar_in_quantized_sequence(
      quantized_sequence)
  steps_per_chord_float = steps_per_bar_float / chords_per_bar
  if steps_per_chord_float != round(steps_per_chord_float):
    raise NonIntegerStepsPerChordException(
        'Non-integer number of steps per chord: %f' % steps_per_chord_float)
  steps_per_chord = int(steps_per_chord_float)
  steps_per_second = sequences_lib.steps_per_quarter_to_steps_per_second(
      quantized_sequence.quantization_info.steps_per_quarter,
      quantized_sequence.tempos[0].qpm)
  seconds_per_chord = steps_per_chord / steps_per_second

  num_chords = int(math.ceil(quantized_sequence.total_time / seconds_per_chord))
  if num_chords == 0:
    raise EmptySequenceException('NoteSequence is empty.')
  if num_chords > _MAX_NUM_CHORDS:
    raise SequenceTooLongException(
        'NoteSequence too long for chord inference: %d frames' % num_chords)

  # Compute pitch vectors for each chord frame, then compute log-likelihood of
  # observing those pitch vectors under each possible chord.
  note_pitch_vectors = sequence_note_pitch_vectors(
      quantized_sequence, seconds_per_frame=seconds_per_chord)
  chord_frame_loglik = _chord_frame_log_likelihood(
      note_pitch_vectors, chord_note_concentration)

  # Compute distribution over chords for each key, and transition distribution
  # between key-chord pairs.
  key_chord_distribution = _key_chord_distribution(
      chord_pitch_out_of_key_prob=chord_pitch_out_of_key_prob)
  key_chord_transition_distribution = _key_chord_transition_distribution(
      key_chord_distribution,
      key_change_prob=key_change_prob,
      chord_change_prob=chord_change_prob)
  key_chord_loglik = np.log(key_chord_distribution)
  key_chord_transition_loglik = np.log(key_chord_transition_distribution)

  key_chords = _key_chord_viterbi(
      chord_frame_loglik, key_chord_loglik, key_chord_transition_loglik)

  # Add the inferred chord changes to the sequence, logging any key changes.
  current_key_name = None
  current_chord_name = None
  for frame, (key, chord) in enumerate(key_chords):
    if _PITCH_CLASS_NAMES[key] != current_key_name:
      if current_key_name is not None:
        tf.logging.info('Sequence has key change from %s to %s at %f seconds.',
                        current_key_name, _PITCH_CLASS_NAMES[key],
                        frame * seconds_per_chord)
      current_key_name = _PITCH_CLASS_NAMES[key]

    if chord == constants.NO_CHORD:
      figure = constants.NO_CHORD
    else:
      root, kind = chord
      figure = '%s%s' % (_PITCH_CLASS_NAMES[root], kind)

    if figure != current_chord_name:
      ta = quantized_sequence.text_annotations.add()
      ta.time = frame * seconds_per_chord
      ta.quantized_step = frame * steps_per_chord
      ta.text = figure
      ta.annotation_type = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
      current_chord_name = figure
