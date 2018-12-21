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
"""Infer melody from polyphonic NoteSequence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect

import numpy as np
import scipy

from magenta.music import constants
from magenta.music import sequences_lib

REST = -1
MELODY_VELOCITY = 127

# Maximum number of melody frames to infer.
MAX_NUM_FRAMES = 10000


def _melody_transition_distribution(rest_prob, interval_prob_fn):
  """Compute the transition distribution between melody pitches (and rest).

  Args:
    rest_prob: Probability that a note will be followed by a rest.
    interval_prob_fn: Function from pitch interval (value between -127 and 127)
        to weight. Will be normalized so that outgoing probabilities (including
        rest) from each pitch sum to one.

  Returns:
    A 257-by-257 melody event transition matrix. Row/column zero represents
    rest. Rows/columns 1-128 represent MIDI note onsets for all pitches.
    Rows/columns 129-256 represent MIDI note continuations (i.e. non-onsets) for
    all pitches.
  """
  pitches = np.arange(constants.MIN_MIDI_PITCH, constants.MAX_MIDI_PITCH + 1)
  num_pitches = len(pitches)

  # Evaluate the probability of each possible pitch interval.
  max_interval = constants.MAX_MIDI_PITCH - constants.MIN_MIDI_PITCH
  intervals = np.arange(-max_interval, max_interval + 1)
  interval_probs = np.vectorize(interval_prob_fn)(intervals)

  # Form the note onset transition matrix.
  interval_probs_mat = scipy.linalg.toeplitz(
      interval_probs[max_interval::-1],
      interval_probs[max_interval::])
  interval_probs_mat /= interval_probs_mat.sum(axis=1)[:, np.newaxis]
  interval_probs_mat *= 1 - rest_prob

  num_melody_events = 1 + 2 * num_pitches
  mat = np.zeros([num_melody_events, num_melody_events])

  # Continuing a rest is a non-event.
  mat[0, 0] = 1

  # All note onsets are equally likely after rest.
  mat[0, 1:num_pitches+1] = np.ones([1, num_pitches]) / num_pitches

  # Transitioning to rest/onset follows user-specified distribution.
  mat[1:num_pitches+1, 0] = rest_prob
  mat[1:num_pitches+1, 1:num_pitches+1] = interval_probs_mat

  # Sustaining a note after onset is a non-event. Transitioning to a different
  # note (without onset) is forbidden.
  mat[1:num_pitches+1, num_pitches+1:] = np.eye(num_pitches)

  # Transitioning to rest/onset follows user-specified distribution.
  mat[num_pitches+1:, 0] = rest_prob
  mat[num_pitches+1:, 1:num_pitches+1] = interval_probs_mat

  # Sustaining a note is a non-event. Transitioning to a different note (without
  # onset) is forbidden.
  mat[num_pitches+1:, num_pitches+1:] = np.eye(num_pitches)

  return mat


def sequence_note_frames(sequence):
  """Split a NoteSequence into frame summaries separated by onsets/offsets.

  Args:
    sequence: The NoteSequence for which to compute frame summaries.

  Returns:
    pitches: A list of MIDI pitches present in `sequence`, in ascending order.
    has_onsets: A Boolean matrix with shape `[num_frames, num_pitches]` where
        entry (i,j) indicates whether pitch j has a note onset in frame i.
    has_notes: A Boolean matrix with shape `[num_frames, num_pitches]` where
        entry (i,j) indicates whether pitch j is present in frame i, either as
        an onset or a sustained note.
    event_times: A list of length `num_frames - 1` containing the event times
        separating adjacent frames.
  """
  notes = [note for note in sequence.notes
           if not note.is_drum
           and note.program not in constants.UNPITCHED_PROGRAMS]

  onset_times = [note.start_time for note in notes]
  offset_times = [note.end_time for note in notes]
  event_times = set(onset_times + offset_times)

  event_times.discard(0.0)
  event_times.discard(sequence.total_time)

  event_times = sorted(event_times)
  num_frames = len(event_times) + 1

  pitches = sorted(set(note.pitch for note in notes))
  pitch_map = dict((p, i) for i, p in enumerate(pitches))
  num_pitches = len(pitches)

  has_onsets = np.zeros([num_frames, num_pitches], dtype=bool)
  has_notes = np.zeros([num_frames, num_pitches], dtype=bool)

  for note in notes:
    start_frame = bisect.bisect_right(event_times, note.start_time)
    end_frame = bisect.bisect_left(event_times, note.end_time)

    has_onsets[start_frame, pitch_map[note.pitch]] = True
    has_notes[start_frame:end_frame+1, pitch_map[note.pitch]] = True

  return pitches, has_onsets, has_notes, event_times


def _melody_frame_log_likelihood(pitches, has_onsets, has_notes, durations,
                                 instantaneous_non_max_pitch_prob,
                                 instantaneous_non_empty_rest_prob,
                                 instantaneous_missing_pitch_prob):
  """Compute the log-likelihood of each frame given each melody state."""
  num_frames = len(has_onsets)
  num_pitches = len(pitches)

  # Whether or not each frame has any notes present at all.
  any_notes = np.sum(has_notes, axis=1, dtype=bool)

  # Whether or not each note has the maximum pitch in each frame.
  if num_pitches > 1:
    has_higher_notes = np.concatenate([
        np.cumsum(has_notes[:, ::-1], axis=1, dtype=bool)[:, num_pitches-2::-1],
        np.zeros([num_frames, 1], dtype=bool)
    ], axis=1)
  else:
    has_higher_notes = np.zeros([num_frames, 1], dtype=bool)

  # Initialize the log-likelihood matrix. There are two melody states for each
  # pitch (onset vs. non-onset) and one rest state.
  mat = np.zeros([num_frames, 1 + 2 * num_pitches])

  # Log-likelihood of each frame given rest. Depends only on presence of any
  # notes.
  mat[:, 0] = (
      any_notes * instantaneous_non_empty_rest_prob +
      ~any_notes * (1 - instantaneous_non_empty_rest_prob))

  # Log-likelihood of each frame given onset. Depends on presence of onset and
  # whether or not it is the maximum pitch. Probability of no observed onset
  # given melody onset is zero.
  mat[:, 1:num_pitches+1] = has_onsets * (
      ~has_higher_notes * (1 - instantaneous_non_max_pitch_prob) +
      has_higher_notes * instantaneous_non_max_pitch_prob)

  # Log-likelihood of each frame given non-onset. Depends on absence of onset
  # and whether note is present and the maximum pitch. Probability of observed
  # onset given melody non-onset is zero; this is to prevent Viterbi from being
  # "lazy" and always treating repeated notes as sustain.
  mat[:, num_pitches+1:] = ~has_onsets * (
      ~has_higher_notes * (1 - instantaneous_non_max_pitch_prob) +
      has_higher_notes * instantaneous_non_max_pitch_prob) * (
          has_notes * (1 - instantaneous_missing_pitch_prob) +
          ~has_notes * instantaneous_missing_pitch_prob)

  # Take the log and scale by duration.
  mat = durations[:, np.newaxis] * np.log(mat)

  return mat


def _melody_viterbi(pitches, melody_frame_loglik, melody_transition_loglik):
  """Use the Viterbi algorithm to infer a sequence of melody events."""
  num_frames, num_melody_events = melody_frame_loglik.shape
  assert num_melody_events == 2 * len(pitches) + 1

  loglik_matrix = np.zeros([num_frames, num_melody_events])
  path_matrix = np.zeros([num_frames, num_melody_events], dtype=np.int32)

  # Assume the very first frame follows a rest.
  loglik_matrix[0, :] = (
      melody_transition_loglik[0, :] + melody_frame_loglik[0, :])

  for frame in range(1, num_frames):
    # At each frame, store the log-likelihood of the best sequence ending in
    # each melody event, along with the index of the parent melody event from
    # the previous frame.
    mat = (np.tile(loglik_matrix[frame - 1][:, np.newaxis],
                   [1, num_melody_events]) +
           melody_transition_loglik)
    path_matrix[frame, :] = mat.argmax(axis=0)
    loglik_matrix[frame, :] = (
        mat[path_matrix[frame, :], range(num_melody_events)] +
        melody_frame_loglik[frame])

  # Reconstruct the most likely sequence of melody events.
  path = [np.argmax(loglik_matrix[-1])]
  for frame in range(num_frames, 1, -1):
    path.append(path_matrix[frame - 1, path[-1]])

  # Mapping from melody event index to rest or (pitch, is-onset) tuple.
  def index_to_event(i):
    if i == 0:
      return REST
    elif i <= len(pitches):
      # Note onset.
      return pitches[i - 1], True
    else:
      # Note sustain.
      return pitches[i - len(pitches) - 1], False

  return [index_to_event(index) for index in path[::-1]]


class MelodyInferenceException(Exception):
  pass


def infer_melody_for_sequence(sequence,
                              melody_interval_scale=2.0,
                              rest_prob=0.1,
                              instantaneous_non_max_pitch_prob=1e-15,
                              instantaneous_non_empty_rest_prob=0.0,
                              instantaneous_missing_pitch_prob=1e-15):
  """Infer melody for a NoteSequence.

  This is a work in progress and should not necessarily be expected to return
  reasonable results. It operates under two main assumptions:

  1) Melody onsets always coincide with actual note onsets from the polyphonic
     NoteSequence.
  2) When multiple notes are active, the melody note tends to be the note with
     the highest pitch.

  Args:
    sequence: The NoteSequence for which to infer melody. This NoteSequence will
        be modified in place, with inferred melody notes added as a new
        instrument.
    melody_interval_scale: The scale parameter for the prior distribution over
        melody intervals.
    rest_prob: The probability of rest after a melody note.
    instantaneous_non_max_pitch_prob: The instantaneous probability that the
        melody note will not have the maximum active pitch.
    instantaneous_non_empty_rest_prob: The instantaneous probability that at
        least one note will be active during a melody rest.
    instantaneous_missing_pitch_prob: The instantaneous probability that the
        melody note will not be active.

  Returns:
    The instrument number used for the added melody.

  Raises:
    MelodyInferenceException: If `sequence` is quantized, or if the number of
        frames is too large.
  """
  if sequences_lib.is_quantized_sequence(sequence):
    raise MelodyInferenceException(
        'Melody inference on quantized NoteSequence not supported.')

  pitches, has_onsets, has_notes, event_times = sequence_note_frames(sequence)

  melody_instrument = (0 if not sequence.notes else
                       max(note.instrument for note in sequence.notes) + 1)
  if melody_instrument == 9:
    # Avoid any confusion around drum channel.
    melody_instrument = 10

  if not pitches:
    # No pitches present in sequence.
    return melody_instrument

  if len(event_times) + 1 > MAX_NUM_FRAMES:
    raise MelodyInferenceException(
        'Too many frames for melody inference: %d' % (len(event_times) + 1))

  # Compute frame durations (times between consecutive note events).
  durations = np.array(
      [event_times[0]] +
      [t2 - t1 for (t1, t2) in zip(event_times[:-1], event_times[1:])] +
      [sequence.total_time - event_times[-1]]
  ) if event_times else np.array([sequence.total_time])

  # Interval distribution is Cauchy-like.
  interval_prob_fn = lambda d: 1 / (1 + (d / melody_interval_scale) ** 2)
  melody_transition_distribution = _melody_transition_distribution(
      rest_prob=rest_prob, interval_prob_fn=interval_prob_fn)

  # Remove all pitches absent from sequence from transition matrix; for most
  # sequences this will greatly reduce the state space.
  num_midi_pitches = constants.MAX_MIDI_PITCH - constants.MIN_MIDI_PITCH + 1
  pitch_indices = (
      [0] +
      [p - constants.MIN_MIDI_PITCH + 1 for p in pitches] +
      [num_midi_pitches + p - constants.MIN_MIDI_PITCH + 1 for p in pitches]
  )
  melody_transition_loglik = np.log(
      melody_transition_distribution[pitch_indices, :][:, pitch_indices])

  # Compute log-likelihood of each frame under each possibly melody event.
  melody_frame_loglik = _melody_frame_log_likelihood(
      pitches, has_onsets, has_notes, durations,
      instantaneous_non_max_pitch_prob=instantaneous_non_max_pitch_prob,
      instantaneous_non_empty_rest_prob=instantaneous_non_empty_rest_prob,
      instantaneous_missing_pitch_prob=instantaneous_missing_pitch_prob)

  # Compute the most likely sequence of melody events using Viterbi.
  melody_events = _melody_viterbi(
      pitches, melody_frame_loglik, melody_transition_loglik)

  def add_note(start_time, end_time, pitch):
    note = sequence.notes.add()
    note.start_time = start_time
    note.end_time = end_time
    note.pitch = pitch
    note.velocity = MELODY_VELOCITY
    note.instrument = melody_instrument

  note_pitch = None
  note_start_time = None

  for event, time in zip(melody_events, [0.0] + event_times):
    if event == REST:
      if note_pitch is not None:
        # A note has just ended.
        add_note(note_start_time, time, note_pitch)
        note_pitch = None

    else:
      pitch, is_onset = event
      if is_onset:
        # This is a new note onset.
        if note_pitch is not None:
          add_note(note_start_time, time, note_pitch)
        note_pitch = pitch
        note_start_time = time
      else:
        # This is a continuation of the current note.
        assert pitch == note_pitch

  if note_pitch is not None:
    # Add the final note.
    add_note(note_start_time, sequence.total_time, note_pitch)

  return melody_instrument
