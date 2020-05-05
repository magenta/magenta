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

# this script need to be called from the root of tf-toolbox
echo "!!! this script need to be called from the root of tf-toolbox"


if [ "$#" -ne 2 ]
then
    echo "arg 1 is a path/shema that contains images to be converted"
    echo "  for instance: myfolder/prefix_%04d_sufix.png"
    echo "  will convert all file from"
    echo "    myfolder/prefix_0001_sufix.png to"
    echo "    myfolder/prefix_9999_sufix.png"
    echo "arg 2 is the video name, for instance video.mp4"
else
    ffmpeg -f image2 -i $1 $2

    echo "to add music: ffmpeg -i video.mp4 -i audio.mp3 -codec copy -shortest output.mp4"
fi
