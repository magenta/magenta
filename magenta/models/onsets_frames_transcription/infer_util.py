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

"""Utilities for inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports

from . import constants
from . import data

import mir_eval
import numpy as np
import pretty_midi
import tensorflow as tf

from magenta.music import constants as mm_constants
from magenta.protobuf import music_pb2


def sequence_to_valued_intervals(note_sequence,
                                 min_duration_ms,
                                 min_midi_pitch=constants.MIN_MIDI_PITCH,
                                 max_midi_pitch=constants.MAX_MIDI_PITCH):
  """Convert a NoteSequence to valued intervals."""
  intervals = []
  pitches = []

  for note in note_sequence.notes:
    if note.pitch < min_midi_pitch or note.pitch > max_midi_pitch:
      continue
    # mir_eval does not allow notes that start and end at the same time.
    if note.end_time == note.start_time:
      continue
    if (note.end_time - note.start_time) * 1000 >= min_duration_ms:
      intervals.append((note.start_time, note.end_time))
      pitches.append(note.pitch)

  return np.array(intervals), np.array(pitches)


def pianoroll_to_note_sequence(
    frames,
    frames_per_second,
    min_duration_ms,
    velocity=70,
    instrument=0,
    program=0,
    qpm=mm_constants.DEFAULT_QUARTERS_PER_MINUTE,
    min_midi_pitch=constants.MIN_MIDI_PITCH,
    onset_predictions=None,
    velocity_values=None):
  """Convert frames to a NoteSequence."""
  frame_length_seconds = 1 / frames_per_second

  sequence = music_pb2.NoteSequence()
  sequence.tempos.add().qpm = qpm
  sequence.ticks_per_quarter = mm_constants.STANDARD_PPQ

  pitch_start_step = {}
  onset_velocities = velocity * np.ones(
      constants.MAX_MIDI_PITCH, dtype=np.int32)

  # Add silent frame at the end so we can do a final loop and terminate any
  # notes that are still active.
  frames = np.append(frames, [np.zeros(frames[0].shape)], 0)
  if velocity_values is None:
    velocity_values = velocity * np.ones_like(frames, dtype=np.int32)

  if onset_predictions is not None:
    onset_predictions = np.append(
        onset_predictions, [np.zeros(onset_predictions[0].shape)], 0)
    # Ensure that any frame with an onset prediction is considered active.
    frames = np.logical_or(frames, onset_predictions)

  def end_pitch(pitch, end_frame):
    """End an active pitch."""
    start_time = pitch_start_step[pitch] * frame_length_seconds
    end_time = end_frame * frame_length_seconds

    if (end_time - start_time) * 1000 >= min_duration_ms:
      note = sequence.notes.add()
      note.start_time = start_time
      note.end_time = end_time
      note.pitch = pitch + min_midi_pitch
      note.velocity = onset_velocities[pitch]
      note.instrument = instrument
      note.program = program

    del pitch_start_step[pitch]

  def unscale_velocity(velocity):
    """Translates a velocity estimate to a MIDI velocity value."""
    return int(max(min(velocity, 1.), 0) * 80. + 10.)

  def process_active_pitch(pitch, i):
    """Process a pitch being active in a given frame."""
    if pitch not in pitch_start_step:
      if onset_predictions is not None:
        # If onset predictions were supplied, only allow a new note to start
        # if we've predicted an onset.
        if onset_predictions[i, pitch]:
          pitch_start_step[pitch] = i
          onset_velocities[pitch] = unscale_velocity(velocity_values[i, pitch])
        else:
          # Even though the frame is active, the onset predictor doesn't
          # say there should be an onset, so ignore it.
          pass
      else:
        pitch_start_step[pitch] = i
    else:
      if onset_predictions is not None:
        # pitch is already active, but if this is a new onset, we should end
        # the note and start a new one.
        if (onset_predictions[i, pitch] and
            not onset_predictions[i - 1, pitch]):
          end_pitch(pitch, i)
          pitch_start_step[pitch] = i
          onset_velocities[pitch] = unscale_velocity(velocity_values[i, pitch])

  for i, frame in enumerate(frames):
    for pitch, active in enumerate(frame):
      if active:
        process_active_pitch(pitch, i)
      elif pitch in pitch_start_step:
        end_pitch(pitch, i)

  sequence.total_time = len(frames) * frame_length_seconds
  if sequence.notes:
    assert sequence.total_time >= sequence.notes[-1].end_time

  return sequence


def safe_log(value):
  """Lower bounded log function."""
  return np.log(1e-6 + value)


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


def _frame_metrics(frame_labels, frame_predictions):
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
  frame_accuracy = tf.reduce_sum(tf.to_float(
      tf.equal(frame_labels_bool, frame_predictions_bool)))

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

    frame = _frame_metrics(metric_frame_labels, metric_frame_predictions)

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
            'metrics/frame_precision': tf.metrics.mean(frame['precision']),
            'metrics/frame_recall': tf.metrics.mean(frame['recall']),
            'metrics/frame_f1_score': tf.metrics.mean(frame['f1_score']),
            'metrics/frame_accuracy': tf.metrics.mean(frame['accuracy']),
            'metrics/frame_true_positives':
                tf.metrics.mean(frame['true_positives']),
            'metrics/frame_false_positives':
                tf.metrics.mean(frame['false_positives']),
            'metrics/frame_false_negatives':
                tf.metrics.mean(frame['false_negatives']),
            'metrics/frame_accuracy_without_true_negatives':
                tf.metrics.mean(frame['accuracy_without_true_negatives']),
        }))

    for metric_name, metric_value in metrics_to_values.iteritems():
      tf.summary.scalar(metric_name, metric_value)

    return (metrics_to_updates, metric_note_precision, metric_note_recall,
            metric_note_f1, metric_note_precision_with_offsets,
            metric_note_recall_with_offsets, metric_note_f1_with_offsets,
            metric_frame_labels, metric_frame_predictions)


def score_sequence(session, global_step_increment, summary_op, summary_writer,
                   metrics_to_updates, metric_note_precision,
                   metric_note_recall, metric_note_f1,
                   metric_note_precision_with_offsets,
                   metric_note_recall_with_offsets,
                   metric_note_f1_with_offsets, metric_frame_labels,
                   metric_frame_predictions, frame_labels, sequence_prediction,
                   frames_per_second, note_sequence_str_label, min_duration_ms,
                   sequence_id):
  """Calculate metrics on the inferred sequence."""
  est_intervals, est_pitches = sequence_to_valued_intervals(
      sequence_prediction,
      min_duration_ms=min_duration_ms)

  sequence_label = music_pb2.NoteSequence.FromString(note_sequence_str_label)
  ref_intervals, ref_pitches = sequence_to_valued_intervals(
      sequence_label,
      min_duration_ms=min_duration_ms)

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

  frame_predictions, _, _, _ = data.sequence_to_pianoroll(
      sequence_prediction,
      frames_per_second=frames_per_second,
      min_pitch=constants.MIN_MIDI_PITCH,
      max_pitch=constants.MAX_MIDI_PITCH)

  if frame_predictions.shape[0] < frame_labels.shape[0]:
    # Pad transcribed frames with silence.
    pad_length = frame_labels.shape[0] - frame_predictions.shape[0]
    frame_predictions = np.pad(
        frame_predictions, [(0, pad_length), (0, 0)], 'constant')
  elif frame_predictions.shape[0] > frame_labels.shape[0]:
    # Truncate transcribed frames.
    frame_predictions = frame_predictions[:frame_labels.shape[0], :]

  global_step, _ = session.run([global_step_increment, metrics_to_updates], {
      metric_frame_predictions: frame_predictions,
      metric_frame_labels: frame_labels,
      metric_note_precision: sequence_note_precision,
      metric_note_recall: sequence_note_recall,
      metric_note_f1: sequence_note_f1,
      metric_note_precision_with_offsets: sequence_note_precision_with_offsets,
      metric_note_recall_with_offsets: sequence_note_recall_with_offsets,
      metric_note_f1_with_offsets: sequence_note_f1_with_offsets
  })
  # Running the summary op separately ensures that all of the metrics have been
  # updated before we try to query them.
  summary = session.run(summary_op)

  tf.logging.info(
      'Writing score summary for %s: Step= %d, Note F1=%f',
      sequence_id, global_step, sequence_note_f1)
  summary_writer.add_summary(summary, global_step)
  summary_writer.flush()

  return sequence_label


def posterior_pianoroll_image(frame_probs, sequence_prediction,
                              frame_labels, frames_per_second, overlap=False):
  """Create a pianoroll image showing frame posteriors, predictions & labels."""
  frame_predictions, _, _, _ = data.sequence_to_pianoroll(
      sequence_prediction,
      frames_per_second=frames_per_second,
      min_pitch=constants.MIN_MIDI_PITCH,
      max_pitch=constants.MAX_MIDI_PITCH)

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
