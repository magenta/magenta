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

"""Utilities for inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.onsets_frames_transcription import constants
from magenta.music import sequences_lib
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


def frame_metrics(frame_labels, frame_predictions):
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


def define_metrics(num_dims):
  """Defines and creates metrics for inference."""
  with tf.variable_scope('metrics'):
    metric_frame_labels = tf.placeholder(
        tf.int32, (None, num_dims), name='metric_frame_labels')
    metric_frame_predictions = tf.placeholder(
        tf.int32, (None, num_dims), name='metric_frame_predictions')
    metric_note_precision = tf.placeholder(
        tf.float32, (), name='metric_note_precision')
    metric_note_recall = tf.placeholder(
        tf.float32, (), name='metric_note_recall')
    metric_note_f1 = tf.placeholder(
        tf.float32, (), name='metric_note_f1')
    metric_note_precision_with_offsets = tf.placeholder(
        tf.float32, (), name='metric_note_precision_with_offsets')
    metric_note_recall_with_offsets = tf.placeholder(
        tf.float32, (), name='metric_note_recall_with_offsets')
    metric_note_f1_with_offsets = tf.placeholder(
        tf.float32, (), name='metric_note_f1_with_offsets')
    metric_note_precision_with_offsets_velocity = tf.placeholder(
        tf.float32, (), name='metric_note_precision_with_offsets_velocity')
    metric_note_recall_with_offsets_velocity = tf.placeholder(
        tf.float32, (), name='metric_note_recall_with_offsets_velocity')
    metric_note_f1_with_offsets_velocity = tf.placeholder(
        tf.float32, (), name='metric_note_f1_with_offsets_velocity')

    frame = frame_metrics(metric_frame_labels, metric_frame_predictions)

    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map({
            'metrics/note_precision':
                tf.metrics.mean(metric_note_precision),
            'metrics/note_recall':
                tf.metrics.mean(metric_note_recall),
            'metrics/note_f1_score':
                tf.metrics.mean(metric_note_f1),
            'metrics/note_precision_with_offsets':
                tf.metrics.mean(metric_note_precision_with_offsets),
            'metrics/note_recall_with_offsets':
                tf.metrics.mean(metric_note_recall_with_offsets),
            'metrics/note_f1_score_with_offsets':
                tf.metrics.mean(metric_note_f1_with_offsets),
            'metrics/note_precision_with_offsets_velocity':
                tf.metrics.mean(metric_note_precision_with_offsets_velocity),
            'metrics/note_recall_with_offsets_velocity':
                tf.metrics.mean(metric_note_recall_with_offsets_velocity),
            'metrics/note_f1_score_with_offsets_velocity':
                tf.metrics.mean(metric_note_f1_with_offsets_velocity),
            'metrics/frame_precision':
                tf.metrics.mean(frame['precision']),
            'metrics/frame_recall':
                tf.metrics.mean(frame['recall']),
            'metrics/frame_f1_score':
                tf.metrics.mean(frame['f1_score']),
            'metrics/frame_accuracy':
                tf.metrics.mean(frame['accuracy']),
            'metrics/frame_true_positives':
                tf.metrics.mean(frame['true_positives']),
            'metrics/frame_false_positives':
                tf.metrics.mean(frame['false_positives']),
            'metrics/frame_false_negatives':
                tf.metrics.mean(frame['false_negatives']),
            'metrics/frame_accuracy_without_true_negatives':
                tf.metrics.mean(frame['accuracy_without_true_negatives']),
        }))

    for metric_name, metric_value in metrics_to_values.items():
      tf.summary.scalar(metric_name, metric_value)

    return (metrics_to_updates, metric_note_precision, metric_note_recall,
            metric_note_f1, metric_note_precision_with_offsets,
            metric_note_recall_with_offsets, metric_note_f1_with_offsets,
            metric_note_precision_with_offsets_velocity,
            metric_note_recall_with_offsets_velocity,
            metric_note_f1_with_offsets_velocity, metric_frame_labels,
            metric_frame_predictions)


def score_sequence(session, global_step_increment, metrics_to_updates,
                   metric_note_precision, metric_note_recall, metric_note_f1,
                   metric_note_precision_with_offsets,
                   metric_note_recall_with_offsets, metric_note_f1_with_offsets,
                   metric_note_precision_with_offsets_velocity,
                   metric_note_recall_with_offsets_velocity,
                   metric_note_f1_with_offsets_velocity, metric_frame_labels,
                   metric_frame_predictions, frame_labels, sequence_prediction,
                   frames_per_second, sequence_label, sequence_id):
  """Calculate metrics on the inferred sequence."""
  est_intervals, est_pitches, est_velocities = sequence_to_valued_intervals(
      sequence_prediction)

  ref_intervals, ref_pitches, ref_velocities = sequence_to_valued_intervals(
      sequence_label)

  sequence_note_precision, sequence_note_recall, sequence_note_f1, _ = (
      mir_eval.transcription.precision_recall_f1_overlap(
          ref_intervals,
          pretty_midi.note_number_to_hz(ref_pitches),
          est_intervals,
          pretty_midi.note_number_to_hz(est_pitches),
          offset_ratio=None))

  (sequence_note_precision_with_offsets,
   sequence_note_recall_with_offsets,
   sequence_note_f1_with_offsets, _) = (
       mir_eval.transcription.precision_recall_f1_overlap(
           ref_intervals,
           pretty_midi.note_number_to_hz(ref_pitches),
           est_intervals,
           pretty_midi.note_number_to_hz(est_pitches)))

  (sequence_note_precision_with_offsets_velocity,
   sequence_note_recall_with_offsets_velocity,
   sequence_note_f1_with_offsets_velocity, _) = (
       mir_eval.transcription_velocity.precision_recall_f1_overlap(
           ref_intervals=ref_intervals,
           ref_pitches=pretty_midi.note_number_to_hz(ref_pitches),
           ref_velocities=ref_velocities,
           est_intervals=est_intervals,
           est_pitches=pretty_midi.note_number_to_hz(est_pitches),
           est_velocities=est_velocities))

  frame_predictions = sequences_lib.sequence_to_pianoroll(
      sequence_prediction,
      frames_per_second=frames_per_second,
      min_pitch=constants.MIN_MIDI_PITCH,
      max_pitch=constants.MAX_MIDI_PITCH).active

  if frame_predictions.shape[0] < frame_labels.shape[0]:
    # Pad transcribed frames with silence.
    pad_length = frame_labels.shape[0] - frame_predictions.shape[0]
    frame_predictions = np.pad(
        frame_predictions, [(0, pad_length), (0, 0)], 'constant')
  elif frame_predictions.shape[0] > frame_labels.shape[0]:
    # Truncate transcribed frames.
    frame_predictions = frame_predictions[:frame_labels.shape[0], :]

  global_step, _ = session.run(
      [global_step_increment, metrics_to_updates], {
          metric_frame_predictions:
              frame_predictions,
          metric_frame_labels:
              frame_labels,
          metric_note_precision:
              sequence_note_precision,
          metric_note_recall:
              sequence_note_recall,
          metric_note_f1:
              sequence_note_f1,
          metric_note_precision_with_offsets:
              sequence_note_precision_with_offsets,
          metric_note_recall_with_offsets:
              sequence_note_recall_with_offsets,
          metric_note_f1_with_offsets:
              sequence_note_f1_with_offsets,
          metric_note_precision_with_offsets_velocity:
              sequence_note_precision_with_offsets_velocity,
          metric_note_recall_with_offsets_velocity:
              sequence_note_recall_with_offsets_velocity,
          metric_note_f1_with_offsets_velocity:
              sequence_note_f1_with_offsets_velocity,
      })

  tf.logging.info('Updating scores for %s: Step= %d, Note F1=%f', sequence_id,
                  global_step, sequence_note_f1)


def posterior_pianoroll_image(frame_probs, sequence_prediction,
                              frame_labels, frames_per_second, overlap=False):
  """Create a pianoroll image showing frame posteriors, predictions & labels."""
  frame_predictions = sequences_lib.sequence_to_pianoroll(
      sequence_prediction,
      frames_per_second=frames_per_second,
      min_pitch=constants.MIN_MIDI_PITCH,
      max_pitch=constants.MAX_MIDI_PITCH).active

  if frame_predictions.shape[0] < frame_labels.shape[0]:
    # Pad transcribed frames with silence.
    pad_length = frame_labels.shape[0] - frame_predictions.shape[0]
    frame_predictions = np.pad(
        frame_predictions, [(0, pad_length), (0, 0)], 'constant')
  elif frame_predictions.shape[0] > frame_labels.shape[0]:
    # Truncate transcribed frames.
    frame_predictions = frame_predictions[:frame_labels.shape[0], :]

  pianoroll_img = np.zeros([len(frame_probs), 3 * len(frame_probs[0]), 3])

  if overlap:
    # Show overlap in yellow
    pianoroll_img[:, :, 0] = np.concatenate(
        [np.array(frame_labels),
         np.array(frame_predictions),
         np.array(frame_probs)],
        axis=1)
    pianoroll_img[:, :, 1] = np.concatenate(
        [np.array(frame_labels),
         np.array(frame_labels),
         np.array(frame_labels)],
        axis=1)
    pianoroll_img[:, :, 2] = np.concatenate(
        [np.array(frame_labels),
         np.zeros_like(frame_predictions),
         np.zeros_like(np.array(frame_probs))],
        axis=1)
  else:
    # Show only red and green
    pianoroll_img[:, :, 0] = np.concatenate(
        [np.array(frame_labels),
         np.array(frame_predictions) * (1.0 - np.array(frame_labels)),
         np.array(frame_probs) * (1.0 - np.array(frame_labels))],
        axis=1)
    pianoroll_img[:, :, 1] = np.concatenate(
        [np.array(frame_labels),
         np.array(frame_predictions) * np.array(frame_labels),
         np.array(frame_probs) * np.array(frame_labels)],
        axis=1)
    pianoroll_img[:, :, 2] = np.concatenate(
        [np.array(frame_labels),
         np.zeros_like(frame_predictions),
         np.zeros_like(np.array(frame_probs))],
        axis=1)

  return np.flipud(np.transpose(pianoroll_img, [1, 0, 2]))
