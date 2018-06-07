# This script use a trained model to generate a video

if [ "$#" -ne 4 ]
then
    echo "arg 1 is for the model path, should contain a checkpoint"
    echo "arg 2 is the name of initial image .jpg"
    echo "arg 3 is for the number of recursion"
    echo "arg 4 is for video name"
else
    python pix2pix/pix2pix-tf/main.py --checkpoint_dir $1 --phase recursion --recursion $3 --file_name_in $2

    ffmpeg -f image2 -i $1/%04d.bmp $4

    rm $1/*.bmp
fi
