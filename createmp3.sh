#!/bin/bash

SOURCE_FOLDER="data/test_data/raw"
TARGET_FOLDER="data/test_data/sound"

mkdir -p "$TARGET_FOLDER"

for video in "$SOURCE_FOLDER"/*; do
    filename=$(basename "$video")
    output="${filename%.*}.mp3"
    ffmpeg -i "$video" -ac 1 -ar 16000 "$TARGET_FOLDER/$output"
done

