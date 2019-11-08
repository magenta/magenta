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

import collections

from magenta.models.onsets_frames_transcription import data
from magenta.music import sequences_lib
import numpy as np


def predict_sequence(frame_predictions, onset_predictions, offset_predictions,
                     velocity_values, min_pitch, hparams,
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
        velocity_values=velocity_values)
  else:
    sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
        frames=frame_predictions,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_duration_ms=0,
        min_midi_pitch=min_pitch,
        onset_predictions=onset_predictions,
        offset_predictions=offset_predictions,
        velocity_values=velocity_values)

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


def posterior_pianoroll_image(frame_probs, frame_labels):
  """Create a pianoroll image showing frame posteriors, predictions & labels."""
  # TODO(fjord): Add back support for visualizing final predicted sequence.
  pianoroll_img = np.zeros([len(frame_probs), 2 * len(frame_probs[0]), 3])

  # Show overlap in yellow
  pianoroll_img[:, :, 0] = np.concatenate(
      [np.array(frame_labels),
       np.array(frame_probs)],
      axis=1)
  pianoroll_img[:, :, 1] = np.concatenate(
      [np.array(frame_labels),
       np.array(frame_labels)],
      axis=1)
  pianoroll_img[:, :, 2] = np.concatenate(
      [np.array(frame_labels),
       np.zeros_like(np.array(frame_probs))],
      axis=1)

  return np.flipud(np.transpose(pianoroll_img, [1, 0, 2]))
