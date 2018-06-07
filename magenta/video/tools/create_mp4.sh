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
