# this script need to be called from the root of tf-toolbox

echo "!!! this script need to be called from the root of tf-toolbox"


if [ "$#" -ne 2 ]
then
    echo "arg 1 is a path that contains images to be enhanced"
    echo "arg 2 is a path that will be created and fullfilled with enhanced pictures"
else
    python ./third_party/enhancenet_pretrained/enhancenet.py \
           --path_in $1/ \
           --path_out $2/
fi
