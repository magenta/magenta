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

from magenta.models.polyamp import dataset_reader, instrument_family_mappings
from magenta.music import sequences_lib, constants
import numpy as np
import tensorflow.keras.backend as K


def predict_multi_sequence(frame_predictions, onset_predictions=None,
                           offset_predictions=None, active_onsets=None, qpm=None, hparams=None,
                           min_pitch=0):

    permuted_frame_predictions = K.permute_dimensions(frame_predictions, (2, 0, 1))

    if onset_predictions is not None:
        permuted_onset_predictions = K.permute_dimensions(onset_predictions, (2, 0, 1))
    else:
        permuted_onset_predictions = [None for _ in range(K.int_shape(permuted_frame_predictions)[0])]

    if offset_predictions is not None:
        permuted_offset_predictions = K.permute_dimensions(offset_predictions, (2, 0, 1))
    else:
        permuted_offset_predictions = [None for _ in range(K.int_shape(permuted_frame_predictions)[0])]

    if active_onsets is not None:
        permuted_active_onsets = K.permute_dimensions(active_onsets, (2, 0, 1))
    else:
        permuted_active_onsets = permuted_onset_predictions

    multi_sequence = None
    for instrument_idx in range(hparams.timbre_num_classes):
        frame_predictions = permuted_frame_predictions[instrument_idx]
        onset_predictions = permuted_onset_predictions[instrument_idx]
        offset_predictions = permuted_offset_predictions[instrument_idx]
        active_onsets = permuted_active_onsets[instrument_idx]
        sequence = predict_sequence(
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            active_onsets=active_onsets,
            velocity_values=None,
            hparams=hparams,
            min_pitch=min_pitch,
            program=instrument_family_mappings.family_to_midi_instrument[instrument_idx] - 1,
            instrument=instrument_idx,
            qpm=qpm)
        if multi_sequence is None:
            multi_sequence = sequence
        else:
            multi_sequence.notes.extend(sequence.notes)
    return multi_sequence


def predict_sequence(frame_predictions,
                     onset_predictions,
                     offset_predictions,
                     velocity_values,
                     min_pitch,
                     hparams,
                     onsets_only=False,
                     instrument=0,
                     program=0,
                     active_onsets=None,
                     qpm=None):
  """Predict sequence given model output."""
  if active_onsets is None:
    # this allows us to set a higher threshold for onsets that we force-add to the frames
    # vs onsets that determine the start of a note
    active_onsets = onset_predictions

  if qpm is None:
      qpm = constants.DEFAULT_QUARTERS_PER_MINUTE

  if not hparams.predict_onset_threshold:
    onset_predictions = None
  if not hparams.predict_offset_threshold:
    offset_predictions = None
  if not hparams.active_onset_threshold:
    active_onsets = None

  if onsets_only:
    if onset_predictions is None:
      raise ValueError(
          'Cannot do onset only prediction if onsets are not defined.')
    sequence_prediction = sequences_lib.pianoroll_onsets_to_note_sequence(
        onsets=onset_predictions,
        frames_per_second=dataset_reader.hparams_frames_per_second(hparams),
        note_duration_seconds=0.05,
        min_midi_pitch=min_pitch,
        velocity_values=velocity_values,
        instrument=instrument,
        program=program,
        qpm=qpm)
  else:
    sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
        frames=frame_predictions,
        frames_per_second=dataset_reader.hparams_frames_per_second(hparams),
        min_duration_ms=0,
        min_midi_pitch=min_pitch,
        onset_predictions=onset_predictions,
        offset_predictions=offset_predictions,
        velocity_values=velocity_values,
        instrument=instrument,
        program=program,
        qpm=qpm,
        active_onsets=active_onsets)

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
