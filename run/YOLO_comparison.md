# YOLOv8 Model Comparison (coco128 quick eval)

Note: coco128 is a lightweight subset intended for quick comparisons, not final COCO accuracy.

| Model   | Params (M) | Inference (ms/img) | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|---------|------------|---------------------|---------------|------------|-------|----------|
| yolov8n | ~3.2       | ~118                | 0.64          | 0.537      | 0.605 | 0.446    |
| yolov8s | ~11.2      | ~244                | 0.797         | 0.664      | 0.760 | 0.589    |
| yolov8m | ~25.9      | ~482                | 0.712         | 0.730      | 0.784 | 0.614    |

Source: Ultralytics `yolo detect val ...` runs saved under `run/yolov8*/` and raw logs. Times measured on CPU (Apple M2) during validation; GPU will be faster.

## Why we used yolov8n in the MVP
- Lowest latency and smallest footprint for real-time assistive guidance on commodity devices.
- Adequate detection quality when paired with BLIP captioning and our narrative reasoner.
- Larger models (s/m) yield higher mAP but increase latency and power draw; they remain configurable via `YOLO_VARIANT`.

## Files
- Plots: see `run/yolov8n/`, `run/yolov8s/`, `run/yolov8m/`.
- This table is exported as PDF at `run/YOLO_comparison.pdf`.
