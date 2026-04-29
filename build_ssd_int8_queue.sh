#!/bin/bash
cd ~/jetson-benchmark
rm -f models/ssd_mobilenet_v2_int8_calib.cache
python3 convert/build_trt_int8_coco.py \
    --onnx models/ssd_mobilenet_v2_raw_fpinput.onnx \
    --engine models/ssd_mobilenet_v2_int8.engine \
    --model-name ssd_mobilenet_v2 \
    --data-dir data/coco_val/images \
    --max-images 200 \
    --batch-size 4 \
    --shape-name input_tensor \
    --shape-dims 1,320,320,3 \
    > models/ssd_mobilenet_v2_int8_coco.build.log 2>&1
echo "RC=$?"
ls -lh models/ssd_mobilenet_v2_int8.engine
