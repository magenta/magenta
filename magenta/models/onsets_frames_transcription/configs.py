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

"""Configurations for transcription models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.common import tf_utils
from magenta.contrib import training as contrib_training
from magenta.models.onsets_frames_transcription import audio_transform
from magenta.models.onsets_frames_transcription import model
from magenta.models.onsets_frames_transcription import model_tpu

Config = collections.namedtuple('Config', ('model_fn', 'hparams'))

DEFAULT_HPARAMS = tf_utils.merge_hparams(
    audio_transform.DEFAULT_AUDIO_TRANSFORM_HPARAMS,
    contrib_training.HParams(
        eval_batch_size=1,
        predict_batch_size=1,
        shuffle_buffer_size=64,
        sample_rate=16000,
        spec_type='mel',
        spec_mel_htk=True,
        spec_log_amplitude=True,
        spec_hop_length=512,
        spec_n_bins=229,
        spec_fmin=30.0,  # A0
        cqt_bins_per_octave=36,
        truncated_length_secs=0.0,
        max_expected_train_example_len=0,
        onset_length=32,
        offset_length=32,
        onset_mode='length_ms',
        onset_delay=0,
        min_frame_occupancy_for_label=0.0,
        jitter_amount_ms=0,
        min_duration_ms=0,
        backward_shift_amount_ms=0,
        velocity_scale=80.0,
        velocity_bias=10.0,
        drum_data_map='',
        drum_prediction_map='',
        velocity_loss_weight=1.0,
        splice_n_examples=0,
        viterbi_decoding=False,
        viterbi_alpha=0.5))

CONFIG_MAP = {}

CONFIG_MAP['onsets_frames'] = Config(
    model_fn=model.model_fn,
    hparams=tf_utils.merge_hparams(DEFAULT_HPARAMS,
                                   model.get_default_hparams()),
)

CONFIG_MAP['drums'] = Config(
    model_fn=model_tpu.model_fn,
    hparams=tf_utils.merge_hparams(
        tf_utils.merge_hparams(DEFAULT_HPARAMS,
                               model_tpu.get_default_hparams()),
        contrib_training.HParams(
            drums_only=True,
            # Ensure onsets take up only a single frame during pianoroll
            # generation.
            onset_length=0,
            velocity_scale=127.0,
            velocity_bias=0.0,
            batch_size=128,
            sample_rate=44100,
            spec_n_bins=250,
            hop_length=441,  # 10ms
            velocity_loss_weight=0.5,
            num_filters=[16, 16, 32],
            fc_size=256,
            onset_lstm_units=64,
            acoustic_rnn_dropout_keep_prob=0.50,
            drum_data_map='8-hit',
            learning_rate=0.0001,
            splice_n_examples=12,
            max_expected_train_example_len=100,
        )),
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
