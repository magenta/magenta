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

#!/bin/bash

set -x
set -e

# Change this to directory where you want to save experiment logs:
logdir=$HOME/logs
# Change this to directory where data is loaded from:
data_dir=$HOME/data/
# Change this to your dataset class, which can be defined in lib_data.py.
dataset=Jsb16thSeparated

# Data preprocessing.
crop_piece_len=32
separate_instruments=True
quantization_level=0.125  # 16th notes

# Hyperparameters.
maskout_method=orderless
num_layers=32
num_filters=64
batch_size=10
use_sep_conv=True
architecture='dilated'
num_dilation_blocks=1
dilate_time_only=False
repeat_last_dilation_level=False
num_pointwise_splits=2
interleave_split_every_n_layers=2


# Run command.
python coconet_train.py \
  --logdir=$logdir \
  --log_process=True \
  --data_dir=$data_dir \
  --dataset=$dataset \
  --crop_piece_len=$crop_piece_len \
  --separate_instruments=$separate_instruments \
  --quantization_level=$quantization_level \
  --maskout_method=$maskout_method \
  --num_layers=$num_layers \
  --num_filters=$num_filters \
  --use_residual \
  --batch_size=$batch_size \
  --use_sep_conv=$use_sep_conv \
  --architecture=$architecture \
  --num_dilation_blocks=$num_dilation_blocks \
  --dilate_time_only=$dilate_time_only \
  --repeat_last_dilation_level=$repeat_last_dilation_level \
  --num_pointwise_splits=$num_pointwise_splits \
  --interleave_split_every_n_layers=$interleave_split_every_n_layers \
  --logtostderr
