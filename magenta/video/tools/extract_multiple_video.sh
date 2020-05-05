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

echo "Create a list of frames from multiple videos"
echo "This script need to be launched from the main tf_toolbox directory, not from the directory witch the script is"

if [ "$#" -ne 3 ]
then
    echo "arg 1 is for ~/path/videos\"*.mp4\""
    echo "arg 2 is for offset between video"
    echo "arg 3 is for destination path for frames"
else
    echo "#######################################"
    echo "will expand $1"
    echo "will use $2 as offset betwwen video"
    echo "will use $3 as destination path"
    echo "#######################################"
    START=0
    names=( $1 )
echo names $names
    for file in "${names[@]}";
    do
        echo "$file"
    done
    echo "#######################################"


    for f in $1;
    do
        echo "Processing $f, starting at $START"
        python img_tools/extract_frames.py \
               --video_in $f \
               --path_out $3 \
               --start $START
        START=$(expr $START + $2)
    done
fi
