#!/usr/bin/env python3
"""Generate SSD MobileNet V2 anchor file (1917 anchors, 320x320 input).

Replicates the TF Object Detection API MultipleGridAnchorGenerator config
for ssd_mobilenet_v2_coco_2018_03_29 / ssd_mobilenet_v2_320x320_coco17.

Output: models/ssd_mobilenet_v2_anchors.npy  shape (1917, 4) [y1, x1, y2, x2] normalized
"""

import os
import sys
import numpy as np


def generate_anchors():
    """
    SSD MobileNet V2 anchor configuration (from TF OD API ssd_mobilenet_v2.config):
      num_layers: 6
      min_scale: 0.2
      max_scale: 0.95
      aspect_ratios: [1.0, 2.0, 0.5, 3.0, 0.333]
      reduce_boxes_in_lowest_layer: true
      interpolated_scale_aspect_ratio: 1.0

    Feature map grid sizes for 320x320 input:
      Layer 0: 20x20  (reduce_boxes → 3 boxes/cell)
      Layer 1: 10x10  (5 aspect_ratios + 1 interpolated → 6 boxes/cell)
      Layer 2:  5x5   (6 boxes/cell)
      Layer 3:  3x3   (6 boxes/cell)
      Layer 4:  2x2   (6 boxes/cell)
      Layer 5:  1x1   (6 boxes/cell)

    Total: 20*20*3 + 10*10*6 + 5*5*6 + 3*3*6 + 2*2*6 + 1*1*6
         = 1200 + 600 + 150 + 54 + 24 + 6 = 2034  ← doesn't match 1917

    Actual feature maps for SSD MobileNetV2 320x320 (from model checkpoint):
      Layer 0: 20x20  (3 boxes)
      Layer 1: 10x10  (6 boxes)
      Layer 2:  5x5   (6 boxes)
      Layer 3:  3x3   (6 boxes)
      Layer 4:  2x2   (6 boxes)
      Layer 5:  1x1   (6 boxes)
    Doesn't add to 1917 either (2034).

    For 1917: likely feature maps [19, 10, 5, 3, 2, 1] (from ssd_mobilenet_v2_300x300)
      19*19*3 + 10*10*6 + 5*5*6 + 3*3*6 + 2*2*6 + 1*1*6
      = 1083 + 600 + 150 + 54 + 24 + 6 = 1917 ✓

    Input size: 300x300 equivalent, feature maps: [19,10,5,3,2,1]
    (The 320x320 torchscript model may internally use 300x300 SSD config)
    """
    num_layers = 6
    min_scale = 0.2
    max_scale = 0.95
    aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
    feature_map_shapes = [19, 10, 5, 3, 2, 1]
    reduce_boxes_in_lowest_layer = True
    interpolated_scale_aspect_ratio = 1.0

    # Compute scales
    scales = [
        min_scale + (max_scale - min_scale) * i / (num_layers - 1)
        for i in range(num_layers)
    ]
    scales.append(1.0)  # extra for interpolated last scale

    anchors = []

    for layer_idx, (fm_size, scale, next_scale) in enumerate(
        zip(feature_map_shapes, scales[:-1], scales[1:])
    ):
        for row in range(fm_size):
            for col in range(fm_size):
                y_center = (row + 0.5) / fm_size
                x_center = (col + 0.5) / fm_size

                if reduce_boxes_in_lowest_layer and layer_idx == 0:
                    # Only 3 anchors per cell at lowest layer
                    box_specs = [
                        (0.1, 1.0),   # small square
                        (scale, 2.0), # wide
                        (scale, 0.5), # tall
                    ]
                    for h_scale, ar in box_specs:
                        h = h_scale / np.sqrt(ar)
                        w = h_scale * np.sqrt(ar)
                        anchors.append([
                            y_center - h / 2.0,
                            x_center - w / 2.0,
                            y_center + h / 2.0,
                            x_center + w / 2.0,
                        ])
                else:
                    # Standard anchors
                    for ar in aspect_ratios:
                        h = scale / np.sqrt(ar)
                        w = scale * np.sqrt(ar)
                        anchors.append([
                            y_center - h / 2.0,
                            x_center - w / 2.0,
                            y_center + h / 2.0,
                            x_center + w / 2.0,
                        ])

                    # Interpolated scale anchor (aspect ratio 1.0)
                    if interpolated_scale_aspect_ratio > 0.0:
                        interp_scale = np.sqrt(scale * next_scale)
                        h = interp_scale
                        w = interp_scale
                        anchors.append([
                            y_center - h / 2.0,
                            x_center - w / 2.0,
                            y_center + h / 2.0,
                            x_center + w / 2.0,
                        ])

    anchors = np.array(anchors, dtype=np.float32)
    print(f"Generated {len(anchors)} anchors, shape={anchors.shape}")
    assert len(anchors) == 1917, f"Expected 1917 anchors, got {len(anchors)}"
    return anchors


if __name__ == "__main__":
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "ssd_mobilenet_v2_anchors.npy"
    )
    out_path = os.path.normpath(out_path)

    anchors = generate_anchors()
    np.save(out_path, anchors)
    print(f"Saved to: {out_path}")
    print(f"Min: {anchors.min():.4f}, Max: {anchors.max():.4f}")
    print(f"Sample anchor[0]: {anchors[0]}")
    print(f"Sample anchor[1083]: {anchors[1083]}")
