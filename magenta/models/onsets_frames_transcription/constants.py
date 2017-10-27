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

"""Defines shared constants used in transcription models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports

import librosa

import tensorflow as tf

MIN_MIDI_PITCH = librosa.note_to_midi('A0')
MAX_MIDI_PITCH = librosa.note_to_midi('C8')
MIDI_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1

DEFAULT_CQT_BINS_PER_OCTAVE = 36
DEFAULT_JITTER_AMOUNT_MS = 0
DEFAULT_JITTER_WAV_AND_LABEL_SEPARATELY = False
DEFAULT_MIN_FRAME_OCCUPANCY_FOR_LABEL = 0.0
DEFAULT_NORMALIZE_AUDIO = False
DEFAULT_ONSET_DELAY = 0
DEFAULT_ONSET_LENGTH = 100
DEFAULT_ONSET_MODE = 'window'
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SPEC_FMIN = 30.0
DEFAULT_SPEC_HOP_LENGTH = 512
DEFAULT_SPEC_LOG_AMPLITUDE = True
DEFAULT_SPEC_N_BINS = 229
DEFAULT_SPEC_TYPE = 'mel'


DEFAULT_HPARAMS = tf.contrib.training.HParams(
    cqt_bins_per_octave=DEFAULT_CQT_BINS_PER_OCTAVE,
    jitter_amount_ms=DEFAULT_JITTER_AMOUNT_MS,
    jitter_wav_and_label_separately=DEFAULT_JITTER_WAV_AND_LABEL_SEPARATELY,
    min_frame_occupancy_for_label=DEFAULT_MIN_FRAME_OCCUPANCY_FOR_LABEL,
    normalize_audio=DEFAULT_NORMALIZE_AUDIO,
    onset_delay=DEFAULT_ONSET_DELAY,
    onset_length=DEFAULT_ONSET_LENGTH,
    onset_mode=DEFAULT_ONSET_MODE,
    sample_rate=DEFAULT_SAMPLE_RATE,
    spec_fmin=DEFAULT_SPEC_FMIN,
    spec_hop_length=DEFAULT_SPEC_HOP_LENGTH,
    spec_log_amplitude=DEFAULT_SPEC_LOG_AMPLITUDE,
    spec_n_bins=DEFAULT_SPEC_N_BINS,
    spec_type=DEFAULT_SPEC_TYPE,
)
