#!/bin/bash
# This tools rename a selected type of file by number

if [ "$#" -ne 3 ]
then
    echo "arg 1 is the path"
    echo "arg 2 is the file extention, for instance 'png'"
    echo "arg 3 is the starting number"
else
    i=$3
    for file in $1/*.$2
    do
        new=$(printf "$1/f%07d.$2" "$i")
        mv -- "$file" "$new"
        let i=i+1
    done
fi
