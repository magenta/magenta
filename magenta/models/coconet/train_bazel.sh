#!/bin/bash

set -x
set -e

# Change this to dir for saving experiment logs.
logdir="logs"
# Change this to where data is loaded from.
data_dir="testdata"
# Change this to your dataset class, which can be defined in lib_data.py.
dataset=TestData

# Data preprocessing.
crop_piece_len=64
separate_instruments=True
quantization_level=0.125  # 16th notes

# Hyperparameters.
maskout_method=orderless
num_layers=32
num_filters=64
batch_size=10

# Run command.
bazel run :coconet_train \
  -- \
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
  --logtostderr
