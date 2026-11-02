"""
Unit tests for Object Detection evaluation metrics and YOLO inference engine.
"""

from pathlib import Path
import numpy as np
import pytest

from src.evaluation.detection_metrics import (
    compute_iou_voc,
    non_maximum_suppression,
    calculate_average_precision,
)
from src.inference.detector_inference import (
    YOLODetector,
    DetectionBox,
    DetectionResponse,
    DEFECT_COLOR_PALETTE,
)


# =====================================================================
# 1. IoU Geometric Tests
# =====================================================================

def test_iou_identical_boxes():
    box = (10.0, 10.0, 50.0, 50.0)
    iou = compute_iou_voc(box, box)
    assert pytest.approx(iou, rel=1e-5) == 1.0


def test_iou_disjoint_boxes():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    iou = compute_iou_voc(box1, box2)
    assert iou == 0.0


def test_iou_partial_overlap():
    box1 = (0.0, 0.0, 10.0, 10.0)    # Area = 100
    box2 = (5.0, 0.0, 15.0, 10.0)    # Area = 100
    # Overlap: [5, 0] to [10, 10] -> w=5, h=10 -> area = 50
    # Union = 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    iou = compute_iou_voc(box1, box2)
    assert pytest.approx(iou, rel=1e-4) == 1.0 / 3.0


def test_iou_enclosed_box():
    outer = (0.0, 0.0, 100.0, 100.0)   # Area = 10,000
    inner = (20.0, 20.0, 80.0, 80.0)   # Area = 3,600
    # Intersection = 3,600, Union = 10,000
    iou = compute_iou_voc(outer, inner)
    assert pytest.approx(iou, rel=1e-5) == 0.36


def test_iou_touching_edge_zero_overlap():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (10.0, 0.0, 20.0, 10.0)  # Touches at x=10
    iou = compute_iou_voc(box1, box2)
    assert iou == 0.0


# =====================================================================
# 2. Non-Maximum Suppression (NMS) Tests
# =====================================================================

def test_nms_suppresses_redundant_boxes():
    # Two heavily overlapping boxes for class 0
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (12.0, 11.0, 51.0, 49.0)  # Heavily overlapping duplicate
    ]
    scores = [0.95, 0.82]
    classes = [0, 0]

    surviving = non_maximum_suppression(boxes, scores, classes, iou_threshold=0.5, conf_threshold=0.25)
    # Only the highest-confidence box (index 0) should survive
    assert surviving == [0]


def test_nms_preserves_different_classes_at_same_location():
    # Two identical boxes but representing different defect classes
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (10.0, 10.0, 50.0, 50.0)
    ]
    scores = [0.90, 0.85]
    classes = [0, 1]  # Class 0 vs Class 1

    surviving = non_maximum_suppression(boxes, scores, classes, iou_threshold=0.5, conf_threshold=0.25)
    # Both must survive because suppression must be class-specific
    assert sorted(surviving) == [0, 1]


def test_nms_filters_low_confidence_boxes():
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (100.0, 100.0, 150.0, 150.0)
    ]
    scores = [0.85, 0.15]  # Second box below 0.25
    classes = [0, 0]

    surviving = non_maximum_suppression(boxes, scores, classes, iou_threshold=0.5, conf_threshold=0.25)
    assert surviving == [0]


def test_nms_preserves_separated_boxes_same_class():
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (100.0, 100.0, 150.0, 150.0)  # Far apart
    ]
    scores = [0.90, 0.88]
    classes = [0, 0]

    surviving = non_maximum_suppression(boxes, scores, classes, iou_threshold=0.5, conf_threshold=0.25)
    assert sorted(surviving) == [0, 1]


# =====================================================================
# 3. Average Precision (mAP) Tests
# =====================================================================

def test_ap_perfect_detection():
    recalls = np.array([0.2, 0.5, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0])
    ap = calculate_average_precision(recalls, precisions)
    assert pytest.approx(ap, rel=1e-4) == 1.0


def test_ap_zero_detection():
    recalls = np.array([0.0, 0.0])
    precisions = np.array([0.0, 0.0])
    ap = calculate_average_precision(recalls, precisions)
    assert ap == 0.0


# =====================================================================
# 4. YOLO Detector Inference Engine Tests
# =====================================================================

def test_yolo_detector_initialization():
    detector = YOLODetector(model_path="yolov8n.pt", device="cpu")
    assert detector is not None
    assert detector.conf_threshold == 0.25
    assert detector.iou_threshold == 0.45
    assert detector.device == "cpu"


def test_yolo_detector_predict_dummy_image():
    detector = YOLODetector(model_path="yolov8n.pt", device="cpu")
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

    response = detector.predict(dummy_img)

    assert isinstance(response, DetectionResponse)
    assert response.image_width == 200
    assert response.image_height == 200
    assert isinstance(response.is_defective, bool)
    assert response.inference_time_ms > 0
    assert isinstance(response.detections, list)


def test_yolo_detector_draw_detections():
    detector = YOLODetector(model_path="yolov8n.pt", device="cpu")
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

    mock_detection = DetectionBox(
        class_id=0,
        class_name="crazing",
        confidence=0.92,
        box_xyxy=(20.0, 20.0, 80.0, 80.0),
        box_normalized=(0.25, 0.25, 0.3, 0.3),
        area_pixels=3600.0,
        area_ratio=0.09
    )

    annotated = detector.draw_detections(dummy_img, [mock_detection])
    assert annotated.shape == dummy_img.shape
    assert annotated.dtype == dummy_img.dtype
    # The annotated image should no longer be completely black
    assert np.any(annotated > 0)
