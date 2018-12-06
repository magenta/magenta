#!/bin/bash

set -x
set -e

# Pass path to checkpoint directory as first argument to this script.
# You can also download a model pretrained on the J.S. Bach chorales dataset from here:
# http://download.magenta.tensorflow.org/models/coconet/checkpoint.zip
# and pass the path up to the inner most directory as first argument when running this
# script.
checkpoint=$1

# Change this to where data is loaded from.
data_dir="testdata"

# Change this to where evaluation results are stored.
eval_logdir="eval_logdir"

# Evaluation settings.
fold=valid
fold_index=1  # Optionally can specify index of specific piece to be evaluated.
unit=frame
chronological=false
ensemble_size=5  # Number of different orderings to average.

# Run command.
python coconet_evaluate.py \
--data_dir=$data_dir \
--eval_logdir=$eval_logdir \
--checkpoint=$checkpoint \
--fold=$fold \
--unit=$unit \
--chronological=$chronological \
--ensemble_size=5 \
#--fold_index=$fold_index
