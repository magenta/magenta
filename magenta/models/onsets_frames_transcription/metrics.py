# Copyright 2019 The Magenta Authors.
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
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

import mir_eval
import numpy as np
import pretty_midi
import tensorflow as tf


def sequence_to_valued_intervals(note_sequence,
                                 min_midi_pitch=constants.MIN_MIDI_PITCH,
                                 max_midi_pitch=constants.MAX_MIDI_PITCH):
  """Convert a NoteSequence to valued intervals."""
  intervals = []
  pitches = []
  velocities = []

  for note in note_sequence.notes:
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
      'true_positives': frame_true_positives,
      'false_positives': frame_false_positives,
      'false_negatives': frame_false_negatives,
      'accuracy': frame_accuracy,
      'accuracy_without_true_negatives': frame_accuracy_without_true_negatives,
      'precision': frame_precision,
      'recall': frame_recall,
      'f1_score': frame_f1_score,
  }


def _calculate_metrics_py(
    frame_predictions, onset_predictions, offset_predictions, velocity_values,
    sequence_label_str, frame_labels, sequence_id, hparams, min_pitch,
    max_pitch, onsets_only):
  """Python logic for calculating metrics on a single example."""
  tf.logging.info('Calculating metrics for %s with length %d', sequence_id,
                  frame_labels.shape[0])

  sequence_prediction = infer_util.predict_sequence(
      frame_predictions=frame_predictions, onset_predictions=onset_predictions,
      offset_predictions=offset_predictions, velocity_values=velocity_values,
      min_pitch=min_pitch, hparams=hparams,
      onsets_only=onsets_only)

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
      sequence_to_valued_intervals(sequence_prediction))

  ref_intervals, ref_pitches, ref_velocities = (
      sequence_to_valued_intervals(sequence_label))

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

  processed_frame_predictions = sequences_lib.sequence_to_pianoroll(
      sequence_prediction,
      frames_per_second=data.hparams_frames_per_second(hparams),
      min_pitch=min_pitch, max_pitch=max_pitch).active

  if processed_frame_predictions.shape[0] < frame_labels.shape[0]:
    # Pad transcribed frames with silence.
    pad_length = frame_labels.shape[0] - processed_frame_predictions.shape[0]
    processed_frame_predictions = np.pad(processed_frame_predictions,
                                         [(0, pad_length), (0, 0)], 'constant')
  elif processed_frame_predictions.shape[0] > frame_labels.shape[0]:
    # Truncate transcribed frames.
    processed_frame_predictions = (
        processed_frame_predictions[:frame_labels.shape[0], :])

  tf.logging.info(
      'Metrics for %s: Note F1 %f, Note w/ offsets F1 %f, '
      'Note w/ offsets & velocity: %f', sequence_id, note_f1,
      note_with_offsets_f1, note_with_offsets_velocity_f1)
  return (note_precision, note_recall, note_f1, note_with_velocity_precision,
          note_with_velocity_recall, note_with_velocity_f1,
          note_with_offsets_precision, note_with_offsets_recall,
          note_with_offsets_f1, note_with_offsets_velocity_precision,
          note_with_offsets_velocity_recall, note_with_offsets_velocity_f1,
          processed_frame_predictions)


def calculate_metrics(frame_predictions,
                      onset_predictions,
                      offset_predictions,
                      velocity_values,
                      sequence_label,
                      frame_labels,
                      sequence_id,
                      hparams,
                      min_pitch,
                      max_pitch,
                      onsets_only=False):
  """Calculate metrics for a single example."""
  (note_precision, note_recall, note_f1, note_with_velocity_precision,
   note_with_velocity_recall, note_with_velocity_f1,
   note_with_offsets_precision, note_with_offsets_recall, note_with_offsets_f1,
   note_with_offsets_velocity_precision, note_with_offsets_velocity_recall,
   note_with_offsets_velocity_f1, processed_frame_predictions) = tf.py_func(
       functools.partial(
           _calculate_metrics_py,
           hparams=hparams,
           min_pitch=min_pitch,
           max_pitch=max_pitch,
           onsets_only=onsets_only),
       inp=[
           frame_predictions, onset_predictions, offset_predictions,
           velocity_values, sequence_label, frame_labels, sequence_id
       ],
       Tout=([tf.float64] * 12) + [tf.float32],
       stateful=False)

  metrics = {
      'note_precision': note_precision,
      'note_recall': note_recall,
      'note_f1_score': note_f1,
      'note_with_velocity_precision': note_with_velocity_precision,
      'note_with_velocity_recall': note_with_velocity_recall,
      'note_with_velocity_f1_score': note_with_velocity_f1,
  }

  if not onsets_only:
    metrics['note_with_offsets_precision'] = note_with_offsets_precision
    metrics['note_with_offsets_recall'] = note_with_offsets_recall
    metrics['note_with_offsets_f1_score'] = note_with_offsets_f1
    metrics[
        'note_with_offsets_velocity_precision'] = note_with_offsets_velocity_precision
    metrics[
        'note_with_offsets_velocity_recall'] = note_with_offsets_velocity_recall
    metrics[
        'note_with_offsets_velocity_f1_score'] = note_with_offsets_velocity_f1

    frame_metrics = calculate_frame_metrics(
        frame_labels=frame_labels,
        frame_predictions=processed_frame_predictions)
    metrics['frame_precision'] = frame_metrics['precision']
    metrics['frame_recall'] = frame_metrics['recall']
    metrics['frame_f1_score'] = frame_metrics['f1_score']
    metrics['frame_accuracy'] = frame_metrics['accuracy']
    metrics['frame_accuracy_without_true_negatives'] = frame_metrics[
        'accuracy_without_true_negatives']

  return metrics


def define_metrics(frame_predictions,
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
                   onsets_only=False):
  """Create a metric name to tf.metric pair dict for transcription metrics."""
  with tf.device('/device:CPU:*'):
    metrics = collections.defaultdict(list)
    for i in range(hparams.eval_batch_size):
      for k, v in calculate_metrics(
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
          onsets_only=onsets_only).items():
        metrics[k].append(v)
    return {'metrics/' + prefix + k: v for k, v in metrics.items()}
