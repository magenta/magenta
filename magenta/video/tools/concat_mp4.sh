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

# concatenate multiple mp4 video to create a single one
echo "Be carfull! This script will remove all your .ts files!"

if [ "$#" -ne 2 ]
then
    echo "arg 1 is for video path"
    echo "arg 2 is for name of the video"
else
    echo "Concat video from path $1"
    # make mylist
    for f in $1/*.mp4; do
        echo "file '$f.ts'" >> mylist.txt;
        ffmpeg -i $f -c copy -bsf:v h264_mp4toannexb -f mpegts $f.ts
    done
    ffmpeg -safe 0 -f concat -i mylist.txt -codec copy $2
    rm mylist.txt
    rm $1/*.ts
fi

echo "to add music: ffmpeg -i $2 -i audio.mp3 -codec copy -shortest output.mp4"
