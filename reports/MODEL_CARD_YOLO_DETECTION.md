---
language:
- en
license: agpl-3.0
tags:
- object-detection
- yolov8
- bounding-box
- computer-vision
- industrial-inspection
metrics:
- map
library_name: ultralytics
pipeline_tag: object-detection
---

# Industrial Defect Localization: YOLOv8n

## Model Overview
- **Architecture:** YOLOv8 Nano (Ultralytics, 2023)
- **Parameters:** ~3.2 Million
- **Input Resolution:** $640 \times 640 \times 3$
- **Target Application:** High-speed real-time localized bounding-box detection of surface flaws.
- **Repository:** `TAC13/industrial-damage-yolov8n`

## Detection Capabilities
- **Classes:** Industrial damage categories including cracks, pits, scratches, and patches.
- **Post-Processing:** Non-Maximum Suppression (NMS) with IoU threshold 0.45 and class confidence threshold 0.25.
- **Inference Latency:** Optimized for edge deployment (< 15 ms on GPU, < 45 ms on CPU).

## Calibration and Coordinate Mapping
- Bounding boxes $[bx, by, bw, bh]$ are mapped to physical metric dimensions using the calibrated camera scale ($s = 0.15\text{ mm/pixel}$).
- **Planar Limitation:** Physical dimensions derived from bounding boxes assume a flat target surface perpendicular to the optical axis.
