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

"""Transcription metrics calculations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
import mir_eval
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2
import numpy as np
import pretty_midi
import tensorflow.compat.v1 as tf

# Disable for Numpy and Pandas containers.
# pylint: disable=g-explicit-length-test


def sequence_to_valued_intervals(note_sequence,
                                 min_midi_pitch=constants.MIN_MIDI_PITCH,
                                 max_midi_pitch=constants.MAX_MIDI_PITCH,
                                 restrict_to_pitch=None):
  """Convert a NoteSequence to valued intervals."""
  intervals = []
  pitches = []
  velocities = []

  for note in note_sequence.notes:
    if restrict_to_pitch and restrict_to_pitch != note.pitch:
      continue
    if note.pitch < min_midi_pitch or note.pitch > max_midi_pitch:
      continue
    # mir_eval does not allow notes that start and end at the same time.
    if note.end_time == note.start_time:
      continue
    intervals.append((note.start_time, note.end_time))
    pitches.append(note.pitch)
    velocities.append(note.velocity)

  # Reshape intervals to ensure that the second dim is 2, even if the list is
  # of size 0. mir_eval functions will complain if intervals is not shaped
  # appropriately.
  return (np.array(intervals).reshape((-1, 2)), np.array(pitches),
          np.array(velocities))


def f1_score(precision, recall):
  """Creates an op for calculating the F1 score.

  Args:
    precision: A tensor representing precision.
    recall: A tensor representing recall.

  Returns:
    A tensor with the result of the F1 calculation.
  """
  return tf.where(
      tf.greater(precision + recall, 0), 2 * (
          (precision * recall) / (precision + recall)), 0)


def accuracy_without_true_negatives(true_positives, false_positives,
                                    false_negatives):
  """Creates an op for calculating accuracy without true negatives.

  Args:
    true_positives: A tensor representing true_positives.
    false_positives: A tensor representing false_positives.
    false_negatives: A tensor representing false_negatives.

  Returns:
    A tensor with the result of the calculation.
  """
  return tf.where(
      tf.greater(true_positives + false_positives + false_negatives, 0),
      true_positives / (true_positives + false_positives + false_negatives), 0)


def calculate_frame_metrics(frame_labels, frame_predictions):
  """Calculate frame-based metrics."""
  frame_labels_bool = tf.cast(frame_labels, tf.bool)
  frame_predictions_bool = tf.cast(frame_predictions, tf.bool)

  frame_true_positives = tf.reduce_sum(tf.to_float(tf.logical_and(
      tf.equal(frame_labels_bool, True),
      tf.equal(frame_predictions_bool, True))))
  frame_false_positives = tf.reduce_sum(tf.to_float(tf.logical_and(
      tf.equal(frame_labels_bool, False),
      tf.equal(frame_predictions_bool, True))))
  frame_false_negatives = tf.reduce_sum(tf.to_float(tf.logical_and(
      tf.equal(frame_labels_bool, True),
      tf.equal(frame_predictions_bool, False))))
  frame_accuracy = (
      tf.reduce_sum(
          tf.to_float(tf.equal(frame_labels_bool, frame_predictions_bool))) /
      tf.cast(tf.size(frame_labels), tf.float32))

  frame_precision = tf.where(
      tf.greater(frame_true_positives + frame_false_positives, 0),
      tf.div(frame_true_positives,
             frame_true_positives + frame_false_positives),
      0)
  frame_recall = tf.where(
      tf.greater(frame_true_positives + frame_false_negatives, 0),
      tf.div(frame_true_positives,
             frame_true_positives + frame_false_negatives),
      0)
  frame_f1_score = f1_score(frame_precision, frame_recall)
  frame_accuracy_without_true_negatives = accuracy_without_true_negatives(
      frame_true_positives, frame_false_positives, frame_false_negatives)

  return {
      'true_positives': [frame_true_positives],
      'false_positives': [frame_false_positives],
      'false_negatives': [frame_false_negatives],
      'accuracy': [frame_accuracy],
      'accuracy_without_true_negatives': [
          frame_accuracy_without_true_negatives],
      'precision': [frame_precision],
      'recall': [frame_recall],
      'f1_score': [frame_f1_score],
  }


def _calculate_metrics_py(frame_probs,
                          onset_probs,
                          frame_predictions,
                          onset_predictions,
                          offset_predictions,
                          velocity_values,
                          sequence_label_str,
                          frame_labels,
                          sequence_id,
                          hparams,
                          min_pitch,
                          max_pitch,
                          onsets_only,
                          restrict_to_pitch=None):
  """Python logic for calculating metrics on a single example."""
  tf.logging.info('Calculating metrics for %s with length %d', sequence_id,
                  frame_labels.shape[0])

  sequence_prediction = infer_util.predict_sequence(
      frame_probs=frame_probs,
      onset_probs=onset_probs,
      frame_predictions=frame_predictions,
      onset_predictions=onset_predictions,
      offset_predictions=offset_predictions,
      velocity_values=velocity_values,
      min_pitch=min_pitch,
      hparams=hparams,
      onsets_only=onsets_only)

  note_density = len(sequence_prediction.notes) / sequence_prediction.total_time

  sequence_label = music_pb2.NoteSequence.FromString(sequence_label_str)

  if hparams.backward_shift_amount_ms:

    def shift_notesequence(ns_time):
      return ns_time + hparams.backward_shift_amount_ms / 1000.

    shifted_sequence_label, skipped_notes = (
        sequences_lib.adjust_notesequence_times(sequence_label,
                                                shift_notesequence))
    assert skipped_notes == 0
    sequence_label = shifted_sequence_label

  est_intervals, est_pitches, est_velocities = (
      sequence_to_valued_intervals(
          sequence_prediction, restrict_to_pitch=restrict_to_pitch))

  ref_intervals, ref_pitches, ref_velocities = (
      sequence_to_valued_intervals(
          sequence_label, restrict_to_pitch=restrict_to_pitch))

  processed_frame_predictions = sequences_lib.sequence_to_pianoroll(
      sequence_prediction,
      frames_per_second=data.hparams_frames_per_second(hparams),
      min_pitch=min_pitch,
      max_pitch=max_pitch).active

  if processed_frame_predictions.shape[0] < frame_labels.shape[0]:
    # Pad transcribed frames with silence.
    pad_length = frame_labels.shape[0] - processed_frame_predictions.shape[0]
    processed_frame_predictions = np.pad(processed_frame_predictions,
                                         [(0, pad_length), (0, 0)], 'constant')
  elif processed_frame_predictions.shape[0] > frame_labels.shape[0]:
    # Truncate transcribed frames.
    processed_frame_predictions = (
        processed_frame_predictions[:frame_labels.shape[0], :])

  if len(ref_pitches) == 0:
    tf.logging.info(
        'Reference pitches were length 0, returning empty metrics for %s:',
        sequence_id)
    return tuple([[]] * 13 + [processed_frame_predictions])

  note_precision, note_recall, note_f1, _ = (
      mir_eval.transcription.precision_recall_f1_overlap(
          ref_intervals,
          pretty_midi.note_number_to_hz(ref_pitches),
          est_intervals,
          pretty_midi.note_number_to_hz(est_pitches),
          offset_ratio=None))

  (note_with_velocity_precision, note_with_velocity_recall,
   note_with_velocity_f1, _) = (
       mir_eval.transcription_velocity.precision_recall_f1_overlap(
           ref_intervals=ref_intervals,
           ref_pitches=pretty_midi.note_number_to_hz(ref_pitches),
           ref_velocities=ref_velocities,
           est_intervals=est_intervals,
           est_pitches=pretty_midi.note_number_to_hz(est_pitches),
           est_velocities=est_velocities,
           offset_ratio=None))

  (note_with_offsets_precision, note_with_offsets_recall, note_with_offsets_f1,
   _) = (
       mir_eval.transcription.precision_recall_f1_overlap(
           ref_intervals, pretty_midi.note_number_to_hz(ref_pitches),
           est_intervals, pretty_midi.note_number_to_hz(est_pitches)))

  (note_with_offsets_velocity_precision, note_with_offsets_velocity_recall,
   note_with_offsets_velocity_f1, _) = (
       mir_eval.transcription_velocity.precision_recall_f1_overlap(
           ref_intervals=ref_intervals,
           ref_pitches=pretty_midi.note_number_to_hz(ref_pitches),
           ref_velocities=ref_velocities,
           est_intervals=est_intervals,
           est_pitches=pretty_midi.note_number_to_hz(est_pitches),
           est_velocities=est_velocities))

  tf.logging.info(
      'Metrics for %s: Note F1 %f, Note w/ velocity F1 %f, Note w/ offsets F1 '
      '%f, Note w/ offsets & velocity: %f', sequence_id, note_f1,
      note_with_velocity_f1, note_with_offsets_f1,
      note_with_offsets_velocity_f1)
  # Return 1-d tensors for the metrics
  return ([note_precision], [note_recall], [note_f1], [note_density],
          [note_with_velocity_precision], [note_with_velocity_recall],
          [note_with_velocity_f1], [note_with_offsets_precision],
          [note_with_offsets_recall], [note_with_offsets_f1
                                      ], [note_with_offsets_velocity_precision],
          [note_with_offsets_velocity_recall], [note_with_offsets_velocity_f1
                                               ], [processed_frame_predictions])


def calculate_metrics(frame_probs,
                      onset_probs,
                      frame_predictions,
                      onset_predictions,
                      offset_predictions,
                      velocity_values,
                      sequence_label,
                      frame_labels,
                      sequence_id,
                      hparams,
                      min_pitch,
                      max_pitch,
                      onsets_only=False,
                      pitch_map=None):
  """Calculate metrics for a single example."""

  def make_metrics(note_precision,
                   note_recall,
                   note_f1,
                   note_density,
                   note_with_velocity_precision,
                   note_with_velocity_recall,
                   note_with_velocity_f1,
                   note_with_offsets_precision,
                   note_with_offsets_recall,
                   note_with_offsets_f1,
                   note_with_offsets_velocity_precision,
                   note_with_offsets_velocity_recall,
                   note_with_offsets_velocity_f1,
                   processed_frame_predictions,
                   frame_labels,
                   onsets_only=False,
                   prefix=''):
    """Create a dict of onset, offset, frame and velocity metrics."""
    def _add_prefix(name):
      return '_'.join(x for x in [prefix, name] if x)

    def _metrics(precision, recall, f1, name):
      """Create and return a dict of metrics."""
      metrics = {
          _add_prefix(name) + '_precision': precision,
          _add_prefix(name) + '_recall': recall,
          _add_prefix(name) + '_f1_score': f1,
      }
      return metrics

    frame_metrics = calculate_frame_metrics(
        frame_labels=frame_labels,
        frame_predictions=processed_frame_predictions)
    metrics = _metrics(frame_metrics['precision'], frame_metrics['recall'],
                       frame_metrics['f1_score'], 'frame')
    metrics.update({
        _add_prefix('frame_accuracy'): frame_metrics['accuracy'],
        _add_prefix('frame_accuracy_without_true_negatives'):
            frame_metrics['accuracy_without_true_negatives'],
        _add_prefix('note_density'): note_density,
    })

    metrics.update(_metrics(note_precision, note_recall, note_f1, 'note'))
    metrics.update(
        _metrics(note_with_velocity_precision, note_with_velocity_recall,
                 note_with_velocity_f1, 'note_with_velocity'))
    if not onsets_only:
      metrics.update(
          _metrics(note_with_offsets_precision, note_with_offsets_recall,
                   note_with_offsets_f1, 'note_with_offsets'))
      metrics.update(
          _metrics(
              note_with_offsets_velocity_precision,
              note_with_offsets_velocity_recall, note_with_offsets_velocity_f1,
              'note_with_offsets_velocity'))
    return metrics

  (note_precision, note_recall, note_f1, note_density,
   note_with_velocity_precision, note_with_velocity_recall,
   note_with_velocity_f1, note_with_offsets_precision, note_with_offsets_recall,
   note_with_offsets_f1, note_with_offsets_velocity_precision,
   note_with_offsets_velocity_recall, note_with_offsets_velocity_f1,
   processed_frame_predictions) = tf.py_func(
       functools.partial(
           _calculate_metrics_py,
           hparams=hparams,
           min_pitch=min_pitch,
           max_pitch=max_pitch,
           onsets_only=onsets_only),
       inp=[
           frame_probs, onset_probs, frame_predictions, onset_predictions,
           offset_predictions, velocity_values, sequence_label, frame_labels,
           sequence_id
       ],
       Tout=([tf.float64] * 13) + [tf.float32],
       stateful=False)
  metrics = make_metrics(
      note_precision,
      note_recall,
      note_f1,
      note_density,
      note_with_velocity_precision,
      note_with_velocity_recall,
      note_with_velocity_f1,
      note_with_offsets_precision,
      note_with_offsets_recall,
      note_with_offsets_f1,
      note_with_offsets_velocity_precision,
      note_with_offsets_velocity_recall,
      note_with_offsets_velocity_f1,
      processed_frame_predictions,
      frame_labels,
      onsets_only=onsets_only)

  if pitch_map:
    for pitch, name in pitch_map.items():
      (note_precision, note_recall, note_f1, note_density,
       note_with_velocity_precision, note_with_velocity_recall,
       note_with_velocity_f1, note_with_offsets_precision,
       note_with_offsets_recall, note_with_offsets_f1,
       note_with_offsets_velocity_precision, note_with_offsets_velocity_recall,
       note_with_offsets_velocity_f1, processed_frame_predictions) = tf.py_func(
           functools.partial(
               _calculate_metrics_py,
               hparams=hparams,
               min_pitch=min_pitch,
               max_pitch=max_pitch,
               onsets_only=onsets_only,
               restrict_to_pitch=pitch),
           inp=[
               frame_probs, onset_probs, frame_predictions, onset_predictions,
               offset_predictions, velocity_values, sequence_label,
               frame_labels, sequence_id + name
           ],
           Tout=([tf.float64] * 13) + [tf.float32],
           stateful=False)
      metrics.update(
          make_metrics(
              note_precision,
              note_recall,
              note_f1,
              note_density,
              note_with_velocity_precision,
              note_with_velocity_recall,
              note_with_velocity_f1,
              note_with_offsets_precision,
              note_with_offsets_recall,
              note_with_offsets_f1,
              note_with_offsets_velocity_precision,
              note_with_offsets_velocity_recall,
              note_with_offsets_velocity_f1,
              processed_frame_predictions,
              frame_labels,
              onsets_only=onsets_only,
              prefix='pitch/' + name))
  return metrics


def define_metrics(frame_probs,
                   onset_probs,
                   frame_predictions,
                   onset_predictions,
                   offset_predictions,
                   velocity_values,
                   length,
                   sequence_label,
                   frame_labels,
                   sequence_id,
                   hparams,
                   min_pitch=constants.MIN_MIDI_PITCH,
                   max_pitch=constants.MAX_MIDI_PITCH,
                   prefix='',
                   onsets_only=False,
                   pitch_map=None):
  """Create a metric name to tf.metric pair dict for transcription metrics."""
  with tf.device('/device:CPU:*'):
    metrics = collections.defaultdict(list)
    for i in range(hparams.eval_batch_size):
      for k, v in calculate_metrics(
          frame_probs=frame_probs[i][:length[i]],
          onset_probs=onset_probs[i][:length[i]],
          frame_predictions=frame_predictions[i][:length[i]],
          onset_predictions=onset_predictions[i][:length[i]],
          offset_predictions=offset_predictions[i][:length[i]],
          velocity_values=velocity_values[i][:length[i]],
          sequence_label=sequence_label[i],
          frame_labels=frame_labels[i][:length[i]],
          sequence_id=sequence_id[i],
          hparams=hparams,
          min_pitch=min_pitch,
          max_pitch=max_pitch,
          onsets_only=onsets_only,
          pitch_map=pitch_map).items():
        metrics[k].append(v)
    return {'metrics/' + prefix + k: v for k, v in metrics.items()}
