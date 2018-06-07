#!/bin/bash
# convert video to mp4 video

echo "!!! this script will try to convert your video to .mp4"

if [ "$#" -ne 1 ]
then
    echo "arg 1 is for mask like *.mov"
else
    echo "will convert $1 to .mp4"

    for i in $1;
    do name=`echo $i | cut -d'.' -f1`;
       echo $name;
       ffmpeg -i "$i" -f mp4 -vcodec libx264 -preset fast -hide_banner "${name}.mp4" ;
    done
fi


