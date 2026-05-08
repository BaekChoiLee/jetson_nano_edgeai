#!/bin/bash
cd $HOME/jetson-benchmark
echo "Starting evaluation..." > eval.log
nohup python3 benchmark/accuracy_eval.py --model mobilenetv3_small --data-dir data/imagenet_val --max-images 1000 >> eval.log 2>&1 &
nohup python3 benchmark/accuracy_eval.py --model resnet50 --data-dir data/imagenet_val --max-images 1000 >> eval.log 2>&1 &
echo "Background processes started." >> eval.log
