#!/bin/bash

set -ex

if [ $# -ne 1 ]; then
        echo "Usage: $0 <path to data directory>"
        exit 1
fi

data_dir=$1

for dir in $(ls -d $data_dir*); do
       for file in $(ls $dir); do
                # Should convert to 22 kHz and mono
                ffmpeg -i "$dir/$file" -ac 1 -ar 22050 "$dir/temp_$file" || true;
                mv "$dir/temp_$file" "$dir/$file";
        done
done