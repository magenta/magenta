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

"""Utilities for inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.models.onsets_frames_transcription import data
from note_seq import sequences_lib
import numpy as np


def probs_to_pianoroll_viterbi(frame_probs, onset_probs, alpha=0.5):
  """Viterbi decoding of frame & onset probabilities to pianoroll.

  Args:
    frame_probs: A numpy array (num-frames-by-num-pitches) of frame
      probabilities.
    onset_probs: A numpy array (num-frames-by-num-pitches) of onset
      probabilities.
    alpha: Relative weight of onset and frame loss, a float between 0 and 1.
      With alpha = 0, onset probabilities will be ignored. With alpha = 1, frame
      probabilities will be ignored.

  Returns:
    A numpy array (num-frames-by-num-pitches) representing the boolean-valued
    pianoroll.
  """
  n, d = onset_probs.shape

  loss_matrix = np.zeros([n, d, 2], dtype=np.float)
  path_matrix = np.zeros([n, d, 2], dtype=np.bool)

  frame_losses = (1 - alpha) * -np.log(np.stack([1 - frame_probs,
                                                 frame_probs], axis=-1))
  onset_losses = alpha * -np.log(np.stack([1 - onset_probs,
                                           onset_probs], axis=-1))

  loss_matrix[0, :, :] = frame_losses[0, :, :] + onset_losses[0, :, :]

  for i in range(1, n):
    transition_loss = np.tile(loss_matrix[i - 1, :, :][:, :, np.newaxis],
                              [1, 1, 2])

    transition_loss[:, 0, 0] += onset_losses[i, :, 0]
    transition_loss[:, 0, 1] += onset_losses[i, :, 1]
    transition_loss[:, 1, 0] += onset_losses[i, :, 0]
    transition_loss[:, 1, 1] += onset_losses[i, :, 0]

    path_matrix[i, :, :] = np.argmin(transition_loss, axis=1)

    loss_matrix[i, :, 0] = transition_loss[
        np.arange(d), path_matrix[i, :, 0].astype(int), 0]
    loss_matrix[i, :, 1] = transition_loss[
        np.arange(d), path_matrix[i, :, 1].astype(int), 1]

    loss_matrix[i, :, :] += frame_losses[i, :, :]

  pianoroll = np.zeros([n, d], dtype=np.bool)
  pianoroll[n - 1, :] = np.argmin(loss_matrix[n - 1, :, :], axis=-1)
  for i in range(n - 2, -1, -1):
    pianoroll[i, :] = path_matrix[
        i + 1, np.arange(d), pianoroll[i + 1, :].astype(int)]

  return pianoroll


def predict_sequence(frame_probs,
                     onset_probs,
                     frame_predictions,
                     onset_predictions,
                     offset_predictions,
                     velocity_values,
                     min_pitch,
                     hparams,
                     onsets_only=False):
  """Predict sequence given model output."""
  if not hparams.predict_onset_threshold:
    onset_predictions = None
  if not hparams.predict_offset_threshold:
    offset_predictions = None

  if onsets_only:
    if onset_predictions is None:
      raise ValueError(
          'Cannot do onset only prediction if onsets are not defined.')
    sequence_prediction = sequences_lib.pianoroll_onsets_to_note_sequence(
        onsets=onset_predictions,
        frames_per_second=data.hparams_frames_per_second(hparams),
        note_duration_seconds=0.05,
        min_midi_pitch=min_pitch,
        velocity_values=velocity_values,
        velocity_scale=hparams.velocity_scale,
        velocity_bias=hparams.velocity_bias)
  else:
    if hparams.viterbi_decoding:
      pianoroll = probs_to_pianoroll_viterbi(
          frame_probs, onset_probs, alpha=hparams.viterbi_alpha)
      onsets = np.concatenate([
          pianoroll[:1, :], pianoroll[1:, :] & ~pianoroll[:-1, :]
      ], axis=0)
      sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
          frames=pianoroll,
          frames_per_second=data.hparams_frames_per_second(hparams),
          min_duration_ms=0,
          min_midi_pitch=min_pitch,
          onset_predictions=onsets,
          velocity_values=velocity_values,
          velocity_scale=hparams.velocity_scale,
          velocity_bias=hparams.velocity_bias)
    else:
      sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
          frames=frame_predictions,
          frames_per_second=data.hparams_frames_per_second(hparams),
          min_duration_ms=0,
          min_midi_pitch=min_pitch,
          onset_predictions=onset_predictions,
          offset_predictions=offset_predictions,
          velocity_values=velocity_values,
          velocity_scale=hparams.velocity_scale,
          velocity_bias=hparams.velocity_bias)

  return sequence_prediction


def labels_to_features_wrapper(data_fn):
  """Add wrapper to data_fn that add labels to features."""
  def wrapper(params, *args, **kwargs):
    """Wrapper for input_fn that adds contents of labels to features.labels."""
    # Workaround for Estimator API that forces 'labels' to be None when in
    # predict mode.
    # https://github.com/tensorflow/tensorflow/issues/17824
    # See also train_util.create_estimator
    assert not args
    dataset = data_fn(params=params, **kwargs)
    features_with_labels_type = collections.namedtuple(
        type(dataset.output_shapes[0]).__name__ + 'WithLabels',
        dataset.output_shapes[0]._fields + ('labels',))
    def add_labels_to_features(features, labels):
      features_dict = features._asdict()
      features_dict.update(labels=labels)
      return features_with_labels_type(**features_dict), labels
    return dataset.map(add_labels_to_features)
  return wrapper


def posterior_pianoroll_image(onset_probs, onset_labels, frame_probs,
                              frame_labels, sequence_frame_predictions):
  """Create a pianoroll image showing frame posteriors, predictions & labels."""
  def probs_and_labels_image(probs, labels, max_length):
    pianoroll_img = np.zeros([max_length, labels.shape[1], 3])

    # Show overlap in yellow
    pianoroll_img[:probs.shape[0], :, 0] = probs
    pianoroll_img[:labels.shape[0], :, 1] = labels
    return pianoroll_img
  max_length = np.max([onset_probs.shape[0], onset_labels.shape[0],
                       frame_probs.shape[0], frame_labels.shape[0],
                       sequence_frame_predictions.shape[0]])
  pianoroll_img = np.concatenate([
      probs_and_labels_image(onset_probs, onset_labels, max_length),
      probs_and_labels_image(frame_probs, frame_labels, max_length),
      probs_and_labels_image(sequence_frame_predictions, frame_labels,
                             max_length),
  ], axis=1)

  return np.flipud(np.transpose(pianoroll_img, [1, 0, 2]))
