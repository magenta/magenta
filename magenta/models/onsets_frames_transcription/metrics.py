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

# TODO(fjord): Combine redundant functionality between this file and infer_util.


def _calculate_metrics_py(
    frame_predictions, onset_predictions, offset_predictions, velocity_values,
    sequence_label_str, frame_labels, sequence_id, hparams):
  """Python logic for calculating metrics on a single example."""
  tf.logging.info('Calculating metrics for %s with length %d', sequence_id,
                  frame_labels.shape[0])
  if not hparams.predict_onset_threshold:
    onset_predictions = None
  if not hparams.predict_offset_threshold:
    offset_predictions = None

  sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
      frames=frame_predictions,
      frames_per_second=data.hparams_frames_per_second(hparams),
      min_duration_ms=0,
      min_midi_pitch=constants.MIN_MIDI_PITCH,
      onset_predictions=onset_predictions,
      offset_predictions=offset_predictions,
      velocity_values=velocity_values)

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
      infer_util.sequence_to_valued_intervals(sequence_prediction))

  ref_intervals, ref_pitches, ref_velocities = (
      infer_util.sequence_to_valued_intervals(sequence_label))

  note_precision, note_recall, note_f1, _ = (
      mir_eval.transcription.precision_recall_f1_overlap(
          ref_intervals,
          pretty_midi.note_number_to_hz(ref_pitches),
          est_intervals,
          pretty_midi.note_number_to_hz(est_pitches),
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
      min_pitch=constants.MIN_MIDI_PITCH,
      max_pitch=constants.MAX_MIDI_PITCH).active

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
  return (note_precision, note_recall, note_f1, note_with_offsets_precision,
          note_with_offsets_recall, note_with_offsets_f1,
          note_with_offsets_velocity_precision,
          note_with_offsets_velocity_recall, note_with_offsets_velocity_f1,
          processed_frame_predictions)


def calculate_metrics(frame_predictions, onset_predictions, offset_predictions,
                      velocity_values, sequence_label, frame_labels,
                      sequence_id, hparams):
  """Calculate metrics for a single example."""
  (note_precision, note_recall, note_f1, note_with_offsets_precision,
   note_with_offsets_recall, note_with_offsets_f1,
   note_with_offsets_velocity_precision, note_with_offsets_velocity_recall,
   note_with_offsets_velocity_f1, processed_frame_predictions) = tf.py_func(
       functools.partial(_calculate_metrics_py, hparams=hparams),
       inp=[
           frame_predictions, onset_predictions, offset_predictions,
           velocity_values, sequence_label, frame_labels, sequence_id
       ],
       Tout=([tf.float64] * 9) + [tf.float32],
       stateful=False)

  frame_metrics = infer_util.frame_metrics(
      frame_labels=frame_labels, frame_predictions=processed_frame_predictions)

  return {
      'note_precision':
          note_precision,
      'note_recall':
          note_recall,
      'note_f1_score':
          note_f1,
      'note_with_offsets_precision':
          note_with_offsets_precision,
      'note_with_offsets_recall':
          note_with_offsets_recall,
      'note_with_offsets_f1_score':
          note_with_offsets_f1,
      'note_with_offsets_velocity_precision':
          note_with_offsets_velocity_precision,
      'note_with_offsets_velocity_recall':
          note_with_offsets_velocity_recall,
      'note_with_offsets_velocity_f1_score':
          note_with_offsets_velocity_f1,
      'frame_precision':
          frame_metrics['precision'],
      'frame_recall':
          frame_metrics['recall'],
      'frame_f1_score':
          frame_metrics['f1_score'],
      'frame_accuracy':
          frame_metrics['accuracy'],
      'frame_accuracy_without_true_negatives':
          frame_metrics['accuracy_without_true_negatives'],
  }


def define_metrics(frame_predictions, onset_predictions, offset_predictions,
                   velocity_values, length, sequence_label, frame_labels,
                   sequence_id, hparams):
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
          hparams=hparams).items():
        metrics[k].append(v)
    return {'metrics/' + k: tf.metrics.mean(v) for k, v in metrics.items()}
