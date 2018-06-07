#!/bin/bash
# concatenate mp4 video

echo "!!! this script will remove all your .ts files!!!"

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
