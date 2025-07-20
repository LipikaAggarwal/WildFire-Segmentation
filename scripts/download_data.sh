#!/usr/bin/env bash
set -e

mkdir -p ../data/images ../data/masks

for i in {1..10}; do
  curl -sL \
    "https://raw.githubusercontent.com/hayatkhan8660-maker/Fire_Seg_Dataset/main/images_prepped_train/img%28${i}%29.jpg" \
    -o ../data/images/img${i}.jpg

  curl -sL \
    "https://raw.githubusercontent.com/hayatkhan8660-maker/Fire_Seg_Dataset/main/annotations_prepped_train/img%28${i}%29.png" \
    -o ../data/masks/img${i}.png
done

echo "✅ Downloaded 10 fire image-mask pairs."
