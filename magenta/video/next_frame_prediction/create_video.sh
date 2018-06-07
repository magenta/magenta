#!/bin/bash

echo "Train a model and make videos, including steps that generate pre recursive pairs"
echo "This script need to be launched from the main tf_toolbox directory, not from the directory witch the script is"
echo "!!! This script contains some rm -rf !!! use with care, check arg1 carfully"

if [ "$#" -ne 4 ]
then
    echo "arg 1 is for dataset path"
    echo "arg 2 is for number of cycle"
    echo "arg 3 is for number of frame per video"
    echo "arg 4 is for backup folder"
else
    echo "Train pix2pix to predict the next frame"

    read -r -p "Do you want to generate the frames from video.mp4? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "creating the 'frames' dir"
            mkdir $1/frames
            rm $1/frames/*.jpg
            python img_tools/extract_frames.py \
                   --video_in $1/video.mp4 \
                   --path_out $1/frames
            ;;
        *)
            echo "keeping 'frames' folder"
            ;;
    esac

    read -r -p "Do you want to reset the first frame as black? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "reset first.jpg"
            cp img/black.jpg img/first.jpg
            ;;
        *)
            echo "keeping first.jpg"
            ;;
    esac

    read -r -p "Do you want to reset the test-frame using a frame from the video? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "copying the test frame"
            cp $1/frames/f0000001.jpg img/test.jpg
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

    read -r -p "Do you want to remove the previous generated pre recursive pairs? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "removing pre recursive pairs"
            rm $1/train/pr*.jpg
            ;;
        *)
            echo "keeping pairs"
            ;;
    esac

    read -r -p "Do you want to remove or recreate the previous logs (to clean tensorboard)? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "removing logs"
            mkdir $1/logs
            rm $1/logs/*
            ;;
        *)
            echo "keeping logs"
            ;;
    esac

    read -r -p "Do you want to remove the previous generated video? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "removing video"
            rm $1/v_*.mp4
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

    read -r -p "Do you want to remove ALL previous model? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "removing model_* folders"
            rm -rf $1/model_*
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
            python pix2pix/join_pairs.py \
                   --path_left $1/frames \
                   --path_right $1/good \
                   --path_out $1/val \
                   --limit 20

            echo "recreate 'val'"
            mkdir $1/val
            rm $1/val/*.jpg
            python pix2pix/join_pairs.py \
                   --path_left $1/frames \
                   --path_right $1/good \
                   --path_out $1/val \
                   --limit 20
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
        python pix2pix/join_pairs.py \
                   --path_left $1/frames \
                   --path_right $1/good \
                   --path_out $1/train \
                   --limit 1000
#was 1000
        echo "trainning $i/$2"
        python pix2pix/pix2pix-tf/main.py \
               --dataset_path $1 \
               --checkpoint_dir $1 \
               --epoch 5 \
               --max_steps 10000 \
               --phase train \
               --continue_train 1
#was 10000
        echo "cleaning logs"
        rm $1/logs/*

        echo "backup model $i"
        mkdir $4/model_$n
        cp $1/checkpoint $4/model_$n
        cp $1/pix2pix.model* $4/model_$n

        # this is the mecanism used for making the train video
        #echo "generate video $i"
        #./pix2pix/recursion.sh $1 img/first.jpg $3 $2/vc_$n.mp4
        #cp -f img/first.jpg img/backup.jpg

        echo "generate video test $i"
#        cp -f img/test.jpg img/start.jpg
        ./pix2pix/recursion_640.sh $4/model_$n img/first.jpg $3 $4/video_$n
#        cp -f img/backup.jpg img/first.jpg

        echo "select some pairs for recursion"
        rm -rf $1/recur
        mkdir $1/recur
        python img_tools/random_pick.py --path_in $1/good --path_out $1/recur \
               --limit 100 #100 for beyon #500 for train; 200 for green
        echo "use pre-recursion"
        python pix2pix/pix2pix-tf/main.py \
               --checkpoint_dir $1 \
               --recursion 15 \
               --phase pre_recursion \
               --dataset_path $1/recur \
               --frames_path $1/frames
#15 was too good fro green, was great for tokyo
        echo "generate pairs from recursion"
        python pix2pix/join_pairs.py \
               --path_left $1/recur \
               --path_right $1/good \
               --path_out $1/train \
               --prefix pr \
               --size 256

        echo "select some pairs for recursion (long)"
        rm -rf $1/recur
        mkdir $1/recur
        python img_tools/random_pick.py --path_in $1/good --path_out $1/recur \
               --limit 2 #5 for the first iteration of 19
        echo "use pre-recursion (long)"
        python pix2pix/pix2pix-tf/main.py \
               --checkpoint_dir $1 \
               --recursion 100 \
               --phase pre_recursion \
               --dataset_path $1/recur \
               --frames_path $1/frames
#100 was good for DJI
        echo "generate pairs from recursion (long)"
        python pix2pix/join_pairs.py \
               --path_left $1/recur \
               --path_right $1/good \
               --path_out $1/train \
               --prefix pr \
               --size 256

    done
    echo "done $2 iterations"
fi
