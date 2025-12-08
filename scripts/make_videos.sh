#!/bin/bash

BASE_DIR="/home/middledoor/PycharmProjects/bird_sahi_temporal/data/visdrone_vid/raw/val/sequences"
OUT_DIR="/home/middledoor/PycharmProjects/bird_sahi_temporal/data/visdrone_vid/videos"
FPS=20

mkdir -p "$OUT_DIR"

for SEQ in "$BASE_DIR"/*; do
    if [ -d "$SEQ" ]; then
        NAME=$(basename "$SEQ")
        OUT_PATH="$OUT_DIR/${NAME}.mp4"

        echo "[INFO] Converting $NAME → $OUT_PATH"

        ffmpeg -y -framerate $FPS -i "$SEQ/%07d.jpg" \
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
            -c:v libx264 -pix_fmt yuv420p -preset fast -crf 18 \
            "$OUT_PATH"
    fi
done

echo "[INFO] All sequences converted (with auto padding)."
