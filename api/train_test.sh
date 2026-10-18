#!/usr/bin/env bash


for ((c = 1; c <= 32; c *= 2)); do
    python -m src.main yolo.train --train-dir ./data/faces_combined/train/ --val-dir ./data/faces_combined/test/ \
        --no-augment-val --c-hidden "$c" --batch-size 64 --epochs 200 --run-name "bn-combined-faces-c$c" \
        --no-save --train-workers 4 --image-size 256 --grid 8 \
        --learning-rate 1e-3 --weight-decay 0.05 --warmup-epochs 3 --min-learning-rate 1e-5
done


sudo -n shutdown -h +2
