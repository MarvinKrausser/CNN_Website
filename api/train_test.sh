#!/usr/bin/env bash


for ((c = 1; c <= 32; c *= 2)); do
    python -m src.main yolo.train --train-dir ./data/faces_scenery/train/ --val-dir ./data/faces_scenery/test/ \
        --no-augment-val --c-hidden "$c" --batch-size 64 --epochs 200 --run-name "bn-weighted-loss-c$c" \
        --no-save --train-workers 4 --image-size 64 --grid 6
done


sudo -n shutdown -h +2
