#!/bin/bash

set -x
set -e

# Change this to dir for saving experiment logs.
logdir=$HOME/logs
# Change this to where data is loaded from.
data_dir="testdata"
data_dir=$HOME/data/
# Change this to your dataset class, which can be defined in lib_data.py.
dataset=TestData

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
