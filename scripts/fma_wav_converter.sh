#!/bin/bash

set -ex

if [ $# -ne 1 ]; then
        echo "Usage: $0 <path to data directory>"
        exit 1
fi

data_dir=$1

for dir in $(ls -d $data_dir/*); do
        for file in $(ls $dir); do
                ffmpeg -y -i "$dir$file" $dir$(basename "$dir$file" .mp3).wav;
        done
done
