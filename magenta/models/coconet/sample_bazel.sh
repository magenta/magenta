#!/bin/bash

set -x
set -e

# Pass path to checkpoint directory as first argument to this script.
# You can also download a model pretrained on the J.S. Bach chorales dataset from here:
# http://download.magenta.tensorflow.org/models/coconet/checkpoint.zip
# and pass the path up to the inner most directory as first argument when running this
# script.
checkpoint=$1

# Change this to path for saving samples.
generation_output_dir=$HOME/samples

# Generation parameters.
# Number of samples to generate in a batch.
gen_batch_size=2
piece_length=16
strategy=igibbs
tfsample=true

# Run command.
python coconet_sample.py \
--checkpoint="$checkpoint" \
--gen_batch_size=$gen_batch_size \
--piece_length=$piece_length \
--temperature=0.99 \
--strategy=$strategy \
--tfsample=$tfsample \
--generation_output_dir=$generation_output_dir \
--logtostderr


