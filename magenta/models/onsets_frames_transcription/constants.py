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

"""Defines shared constants used in transcription models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import librosa
from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription import audio_transform
import tensorflow as tf

DEFAULT_SAMPLE_RATE = 16000

DEFAULT_SPEC_TYPE = 'cqt'
DEFAULT_SPEC_LOG_AMPLITUDE = False
DEFAULT_SPEC_MEL_HTK = False

DEFAULT_SPEC_HOP_LENGTH = 512
DEFAULT_SPEC_N_BINS = 264  # (88/12)*36=264
DEFAULT_SPEC_FMIN = librosa.note_to_hz(['A0'])[0]

DEFAULT_CQT_BINS_PER_OCTAVE = 36

DEFAULT_FRAMES_PER_SECOND = DEFAULT_SAMPLE_RATE / DEFAULT_SPEC_HOP_LENGTH

MIN_MIDI_PITCH = librosa.note_to_midi('A0')
MAX_MIDI_PITCH = librosa.note_to_midi('C8')
MIDI_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1

MAX_MIDI_VELOCITY = 127

DEFAULT_CROP_TRAINING_SEQUENCE_TO_NOTES = False
DEFAULT_ONSET_MODE = 'length_ms'
DEFAULT_ONSET_LENGTH = 100
DEFAULT_ONSET_DELAY = 0
DEFAULT_MIN_FRAME_OCCUPANCY_FOR_LABEL = 0.0
DEFAULT_JITTER_AMOUNT_MS = 0
DEFAULT_NORMALIZE_AUDIO = False
DEFAULT_MIN_DURATION_MS = 0
DEFAULT_BACKWARD_SHIFT_AMOUNT_MS = 0
DEFAULT_BIDIRECTIONAL = True
DEFAULT_ONSET_OVERLAP = True
DEFAULT_OFFSET_LENGTH = 100

DEFAULT_AUDIO_HPARAMS = tf.contrib.training.HParams(
    sample_rate=DEFAULT_SAMPLE_RATE,
    spec_type=DEFAULT_SPEC_TYPE,
    spec_mel_htk=DEFAULT_SPEC_MEL_HTK,
    spec_log_amplitude=DEFAULT_SPEC_LOG_AMPLITUDE,
    spec_hop_length=DEFAULT_SPEC_HOP_LENGTH,
    spec_n_bins=DEFAULT_SPEC_N_BINS,
    spec_fmin=DEFAULT_SPEC_FMIN,
    cqt_bins_per_octave=DEFAULT_CQT_BINS_PER_OCTAVE,
    normalize_audio=DEFAULT_NORMALIZE_AUDIO,
    crop_training_sequence_to_notes=DEFAULT_CROP_TRAINING_SEQUENCE_TO_NOTES,
    onset_length=DEFAULT_ONSET_LENGTH,
    offset_length=DEFAULT_OFFSET_LENGTH,
    onset_mode=DEFAULT_ONSET_MODE,
    onset_delay=DEFAULT_ONSET_DELAY,
    min_frame_occupancy_for_label=DEFAULT_MIN_FRAME_OCCUPANCY_FOR_LABEL,
    jitter_amount_ms=DEFAULT_JITTER_AMOUNT_MS,
    min_duration_ms=DEFAULT_MIN_DURATION_MS,
    backward_shift_amount_ms=DEFAULT_BACKWARD_SHIFT_AMOUNT_MS,
    bidirectional=DEFAULT_BIDIRECTIONAL,
    onset_overlap=DEFAULT_ONSET_OVERLAP)

DEFAULT_HPARAMS = tf_utils.merge_hparams(
    DEFAULT_AUDIO_HPARAMS, audio_transform.DEFAULT_AUDIO_TRANSFORM_HPARAMS)
