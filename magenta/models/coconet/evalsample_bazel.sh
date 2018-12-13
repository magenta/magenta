#!/bin/bash

set -x
set -e

# Pass path to checkpoint directory as first argument to this script.
# You can also download a model pretrained on the J.S. Bach chorales dataset from here:
# http://download.magenta.tensorflow.org/models/coconet/checkpoint.zip
# and pass the path up to the inner most directory as first argument when running this
# script.
checkpoint=$1

# Change this to the path of samples to be evaluated.
sample_file=samples/generated_result.npy

# Change this to where evaluation results are stored.
eval_logdir="eval_logdir"

# Evaluation settings.
#fold_index=  # Optionally can specify index of specific piece to be evaluated.
unit=frame
chronological=false
ensemble_size=5  # Number of different orderings to average.


python coconet_evaluate.py \
--checkpoint=$checkpoint \
--eval_logdir=$eval_logdir \
--unit=$unit \
--chronological=$chronological \
--ensemble_size=5 \
--sample_npy_path=$sample_file
#--fold_index $fold_index
