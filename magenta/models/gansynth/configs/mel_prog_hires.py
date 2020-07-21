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

"""Config for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Hyperparameters
hifreqres = True
data_type = 'mel'  # 'linear', 'phase'
train_progressive = True
lr = 8e-4

# Define Config
hparams = {}

# Training
hparams['data_type'] = data_type
hparams['total_num_images'] = 11*1000*1000 if train_progressive else 4*1000*1000
hparams['discriminator_learning_rate'] = lr
hparams['generator_learning_rate'] = lr
hparams['train_progressive'] = train_progressive
hparams['stable_stage_num_images'] = 800*1000
hparams['transition_stage_num_images'] = 800*1000
hparams['save_summaries_num_images'] = 10*1000
hparams['batch_size_schedule'] = [8]

# Network
hparams['fmap_base'] = 4096
hparams['fmap_decay'] = 1.0
hparams['fmap_max'] = 256
hparams['fake_batch_size'] = 61
hparams['latent_vector_size'] = 256
hparams['kernel_size'] = 3

# Loss Functions
hparams['gradient_penalty_target'] = 1.0
hparams['gradient_penalty_weight'] = 10.0
hparams['real_score_penalty_weight'] = 0.001
hparams['generator_ac_loss_weight'] = 10.0
hparams['discriminator_ac_loss_weight'] = 10.0
hparams['gen_gl_consistency_loss_weight'] = 0.0

# STFT specific
hparams['dataset_name'] = 'nsynth_tfds'
hparams['g_fn'] = 'specgram'
hparams['d_fn'] = 'specgram'
hparams['scale_mode'] = 'ALL'
hparams['scale_base'] = 2
hparams['num_resolutions'] = 7

if hifreqres:
  hparams['start_height'] = 2
  hparams['start_width'] = 16
else:
  hparams['start_height'] = 4
  hparams['start_width'] = 8
