#!/bin/bash

# Copyright 2016 Google Inc. All Rights Reserved.
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


# Usage:
# ./run_basic_rnn_train.sh experiment_dir hyperparameter_string num_steps training_tfrecord [eval_tfrecord]
#
# Example:
# ./run_basic_rnn_train.sh /tmp/melody_lstm_2500_250 '{"rnn_layer_sizes":[50,20],"batch_size":100}' 20000 /tmp/train_melodies.tfrecord /tmp/eval_melodies.tfrecord

EXPERIMENT_DIR=$1
HPARAMS=$2
NUM_TRAINING_STEPS=$3
TRAIN_SET=$4
EVAL_SET=$5

# Get next run directory.
# http://stackoverflow.com/a/23961677
DATE=$(date +"%d%m%Y")
N=1

# Increment $N as long as a directory with that name exists
while [[ -d "$EXPERIMENT_DIR/$DATE-$N" ]] ; do
    N=$(($N+1))
done

RUN_DIR="$EXPERIMENT_DIR/$DATE-$N"

# Build train job.
bazel build //magenta/models/basic_rnn:basic_rnn_train
BINARY=../../../bazel-bin/magenta/models/basic_rnn/basic_rnn_train

# Run training job.
$BINARY --experiment_run_dir=$RUN_DIR --eval=false --sequence_example_file=$TRAIN_SET --hparams=$HPARAMS --num_training_steps=$NUM_TRAINING_STEPS &

# Run eval job if eval dataset is given.
if [ ! -z "$EVAL_SET" ]; then
  $BINARY --experiment_run_dir=$RUN_DIR --eval=true --sequence_example_file=$EVAL_SET --hparams=$HPARAMS --num_training_steps=$NUM_TRAINING_STEPS &
fi

# Run TensorBoard to see training and eval progress.
tensorboard --logdir=$EXPERIMENT_DIR
