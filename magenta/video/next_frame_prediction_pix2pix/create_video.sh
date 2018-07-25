#!/bin/bash

# Copyright 2017 Google Inc. All Rights Reserved.
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

echo "Train a model and make videos, including steps that generate pre recursive pairs"
echo "This script need to be launched from the main tf_toolbox directory, not from the directory witch the script is"
echo "!!! This script contains some rm -rf !!! use with care, check arg1 carfully"

if [ "$#" -ne 4 ]
then
    echo "arg 1 is for dataset absolute path"
    echo "arg 2 is for number of cycle"
    echo "arg 3 is for number of frame per video"
    echo "arg 4 is for backup folder absolute path"
else
    echo "Train pix2pix to predict the next frame"

    read -r -p "Do you want to generate the frames from video.mp4? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "creating the 'frames' dir"
            mkdir $1/frames
            rm $1/frames/*.jpg
            python ./magenta/video/tools/extract_frames.py \
                   --video_in $1/video.mp4 \
                   --path_out $1/frames
            ;;
        *)
            echo "keeping 'frames' folder"
            ;;
    esac

    read -r -p "Do you want to reset the first-frame using a frame from the video? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "copying the test frame"
            mkdir $1/img
            cp $1/frames/f0000001.jpg $1/img/first.jpg
            ;;
        *)
            echo "keeping test.jpg"
            ;;
    esac

    read -r -p "Do you want to generate the 'good' directory? just by copying the frame folder [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "creating the 'good' dir copying the frame folder"
            rm -rf $1/good
            mkdir $1/good
            cp $1/frames/* $1/good
            ;;
        *)
            echo "keeping 'good' folder"
            ;;
    esac

    read -r -p "Do you want to (re)create 'train' [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "recreate 'train'"
            rm -rf $1/train
            mkdir $1/train
            ;;
        *)
            echo "keeping 'train'"
            ;;
    esac

    read -r -p "Do you want to remove or recreate the previous logs (to clean tensorboard)? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "removing logs"
            rm -rf $1/logs
            mkdir $1/logs
            ;;
        *)
            echo "keeping logs"
            ;;
    esac

    read -r -p "Do you want to remove the previous generated video? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "removing video"
            rm $4/video*.mp4
            ;;
        *)
            echo "keeping video"
            ;;
    esac

    read -r -p "Do you want to remove the CURENT model? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "removing model checkpoint"
            rm $1/pix2pix.model*
            rm $1/checkpoint
            ;;
        *)
            echo "keeping model"
            ;;
    esac

    read -r -p "Do you want to (re)create 'test' and 'val'? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "recreate 'test'"
            mkdir $1/test
            rm $1/test/*.jpg
            python ./magenta/video/next_frame_prediction_pix2pix/join_pairs.py \
                   --path_left $1/frames \
                   --path_right $1/good \
                   --path_out $1/val \
                   --limit 10

            echo "recreate 'val'"
            mkdir $1/val
            rm $1/val/*.jpg
            python ./magenta/video/next_frame_prediction_pix2pix/join_pairs.py \
                   --path_left $1/frames \
                   --path_right $1/good \
                   --path_out $1/val \
                   --limit 10
            ;;
        *)
            echo "keeping 'test' and 'val'"
            ;;
    esac

    echo "#######################################"
    echo "starting sequence from 1 to $2"
    for i in $(seq 1 $2)
    do
        n=$(printf %03d $i)

        echo "making pairs $i/$2"
        python ./magenta/video/next_frame_prediction_pix2pix/join_pairs.py \
                   --path_left $1/frames \
                   --path_right $1/good \
                   --path_out $1/train \
                   --limit 1000
# 1000 is the default value, you can play with it and will get diferents results
        echo "trainning $i/$2"
        # main.py belong to the pix2ix_tensorflow package
        python ./external/pix2pix_tensorflow/main.py \
               --dataset_path $1 \
               --checkpoint_dir $1 \
               --epoch 5 \
               --max_steps 10000 \
               --phase train \
               --continue_train 1
# 10000 is the default value, you can play with it and will get diferents results
        echo "cleaning logs"
        rm $1/logs/*

        echo "backup model $i"
        mkdir $4/model_$n
        cp $1/checkpoint $4/model_$n
        cp $1/pix2pix.model* $4/model_$n

        echo "generate video test $i"
        ./magenta/video/next_frame_prediction_pix2pix/recursion_640.sh $4/model_$n $1/img/first.jpg $3 $4/video_$n

        echo "select some pairs for recursion"
        rm -rf $1/recur
        mkdir $1/recur
        python ./magenta/video/tools/random_pick.py --path_in $1/good --path_out $1/recur \
               --limit 100
# 100 is the default value, you can play with it and will get diferents results
        echo "use pre-recursion"
        python ./external/pix2pix_tensorflow/main.py \
               --checkpoint_dir $1 \
               --recursion 15 \
               --phase pre_recursion \
               --dataset_path $1/recur \
               --frames_path $1/frames
# 15 is the default value, you can play with it and will get diferents results
        echo "generate pairs from recursion"
        python ./magenta/video/next_frame_prediction_pix2pix/join_pairs.py \
               --path_left $1/recur \
               --path_right $1/good \
               --path_out $1/train \
               --prefix pr \
               --size 256

        echo "select some pairs for recursion (long)"
        rm -rf $1/recur
        mkdir $1/recur
        python ./magenta/video/tools/random_pick.py --path_in $1/good --path_out $1/recur \
               --limit 2
# 2 is the default value, you can play with it and will get diferents results
        echo "use pre-recursion (long)"
        python ./external/pix2pix_tensorflow/main.py \
               --checkpoint_dir $1 \
               --recursion 100 \
               --phase pre_recursion \
               --dataset_path $1/recur \
               --frames_path $1/frames
# 100 is the default value, you can play with it and will get diferents results
        echo "generate pairs from recursion (long)"
        python ./magenta/video/next_frame_prediction_pix2pix/join_pairs.py \
               --path_left $1/recur \
               --path_right $1/good \
               --path_out $1/train \
               --prefix pr \
               --size 256

    done
    echo "done $2 iterations"
fi
