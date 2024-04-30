#!/bin/bash

set -ex

if [ $# -ne 2 ]; then
        echo "Usage: $0 <path to data directory> <extension, like mp3>"
        exit 1
fi

data_dir=$1
extension=$2

for dir in $(ls -d $data_dir*); do
       for file in $(ls $dir); do
                ffmpeg -y -i "$dir/$file" -ac 1 -ar 22050 $dir/$(basename "$file" .$extension).wav || true;
                rm -f $dir/$file;
        done
done
