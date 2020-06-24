#!/bin/bash

python3 ../interface/main.py --run_name=BlitzNet300_x4_VOC0712_det --image_size=300 --ckpt=65 --eval_min_conf=0.5 --detect
