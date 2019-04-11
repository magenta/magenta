# This script use a trained model to generate a video
# it streches the output to 640x360px

if [ "$#" -ne 4 ]
then
    echo "arg 1 is for the model path, should contain a checkpoint"
    echo "arg 2 is the name of initial image .jpg"
    echo "arg 3 is for the number of recursion"
    echo "arg 4 is for video path/name"
else
    python pix2pix-tensorflow-0.1/main.py --checkpoint_dir $1/ --phase recursion --recursion $3 --file_name_in $2

    mkdir $4/
    python ../tools/convert2jpg.py \
           --path_in $1 \
           --path_out $4/ \
           --strech \
           --xsize 640 \
           --ysize 360 \
           --delete
    ffmpeg -i $4/%04d.jpg -vcodec libx264 $4.mp4
fi
