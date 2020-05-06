# Copyright 2020 The Magenta Authors.
# Modifications Copyright 2020 Jack Spencer Smith.
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

"""Configurations for transcription models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import librosa

from magenta.models.polyamp import audio_transform, melodic_model, timbre_model, \
    constants, full_model

Config = collections.namedtuple('Config', ('hparams',))

DEFAULT_HPARAMS = {
    **audio_transform.DEFAULT_AUDIO_TRANSFORM_HPARAMS,
    **{
        'model_id': None,
        'batch_size': 2,
        'epochs_per_save': 50,
        'using_plaidml': False,
        'eval_batch_size': 1,
        'predict_batch_size': 1,
        'shuffle_buffer_size': 64,
        'nsynth_shuffle_buffer_size': 256,
        'nsynth_batch_size': 2,
        'slakh_batch_size': 2,
        'use_drums': False,
        'timbre_training_max_instruments': 64,
        'timbre_max_start_offset': 160000,  # 10 seconds
        'timbre_min_len': 8000,
        'timbre_max_len': 0,
        'sample_rate': 16000,
        'spec_type': 'cqt',
        'spec_mel_htk': True,
        'spec_log_amplitude': True,
        'timbre_spec_type': 'cqt',
        'timbre_spec_log_amplitude': True,
        'spec_hop_length': 512,
        'timbre_hop_length': 256,
        'spec_n_bins': constants.SPEC_BANDS,
        'spec_fmin': librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),  # 30.0  A0
        'cqt_bins_per_octave': constants.BINS_PER_OCTAVE,  # 28,
        'truncated_length_secs': 0.0,
        'max_expected_train_example_len': 0,
        'onset_length': 32,
        'offset_length': 32,
        'onset_mode': 'length_ms',
        'onset_delay': 0,
        'min_frame_occupancy_for_label': 0.0,
        'jitter_amount_ms': 0,
        'min_duration_ms': 0,
        'backward_shift_amount_ms': 0,
    }}

CONFIG_MAP = {}
CONFIG_MAP['melodic'] = Config(
    hparams={**DEFAULT_HPARAMS,
             **melodic_model.get_default_hparams()},
)

CONFIG_MAP['timbre'] = Config(
    hparams={**DEFAULT_HPARAMS,
             **timbre_model.get_default_hparams()},
)

CONFIG_MAP['full'] = Config(
    hparams={**DEFAULT_HPARAMS,
             **melodic_model.get_default_hparams(),
             **timbre_model.get_default_hparams(),
             **full_model.get_default_hparams()},
)

DatasetConfig = collections.namedtuple(
    'DatasetConfig', ('name', 'path', 'num_mixes', 'process_for_training'))

DATASET_CONFIG_MAP = {}

DATASET_CONFIG_MAP['maestro'] = [
    DatasetConfig(
        'train',
        'gs://magentadata/datasets/maestro/v1.0.0/'
        'maestro-v1.0.0_ns_wav_train.tfrecord@10',
        num_mixes=None,
        process_for_training=True),
    DatasetConfig(
        'eval_train',
        'gs://magentadata/datasets/maestro/v1.0.0/'
        'maestro-v1.0.0_ns_wav_train.tfrecord@10',
        num_mixes=None,
        process_for_training=False),
    DatasetConfig(
        'test',
        'gs://magentadata/datasets/maestro/v1.0.0/'
        'maestro-v1.0.0_ns_wav_test.tfrecord@10',
        num_mixes=None,
        process_for_training=False),
    DatasetConfig(
        'validation',
        'gs://magentadata/datasets/maestro/v1.0.0/'
        'maestro-v1.0.0_ns_wav_validation.tfrecord@10',
        num_mixes=None,
        process_for_training=False),
]

DATASET_CONFIG_MAP['slakh'] = [
    DatasetConfig(
        'train',
        'E:/Datasets/slakh'
        'train.tfrecord@20',
        num_mixes=None,
        process_for_training=True),
]
