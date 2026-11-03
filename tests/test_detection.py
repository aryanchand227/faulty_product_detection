"""
Unit tests for Object Detection evaluation metrics and YOLO inference engine.
"""

from pathlib import Path
import numpy as np
import pytest

from src.evaluation.detection_metrics import (
    compute_iou_voc,
    non_maximum_suppression,
    calculate_all_points_interpolated_ap,
    calculate_average_precision,
    match_detections_for_class,
    evaluate_detections_for_class,
    evaluate_dataset_detections,
)
from src.inference.detector_inference import (
    YOLODetector,
    DetectionBox,
    DetectionResponse,
    DEFECT_COLOR_PALETTE,
    compute_bbox_union_area,
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
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (12.0, 11.0, 51.0, 49.0)  # Heavily overlapping duplicate
    ]
    scores = [0.95, 0.82]
    classes = [0, 0]

    surviving = non_maximum_suppression(boxes, scores, classes, iou_threshold=0.5, conf_threshold=0.25)
    assert surviving == [0]


def test_nms_preserves_different_classes_at_same_location():
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (10.0, 10.0, 50.0, 50.0)
    ]
    scores = [0.90, 0.85]
    classes = [0, 1]

    surviving = non_maximum_suppression(boxes, scores, classes, iou_threshold=0.5, conf_threshold=0.25)
    assert sorted(surviving) == [0, 1]


def test_nms_filters_low_confidence_boxes():
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (100.0, 100.0, 150.0, 150.0)
    ]
    scores = [0.85, 0.15]
    classes = [0, 0]

    surviving = non_maximum_suppression(boxes, scores, classes, iou_threshold=0.5, conf_threshold=0.25)
    assert surviving == [0]


def test_nms_preserves_separated_boxes_same_class():
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (100.0, 100.0, 150.0, 150.0)
    ]
    scores = [0.90, 0.88]
    classes = [0, 0]

    surviving = non_maximum_suppression(boxes, scores, classes, iou_threshold=0.5, conf_threshold=0.25)
    assert sorted(surviving) == [0, 1]


# =====================================================================
# 3. All-Points Interpolated AP & Greedy Matching Tests
# =====================================================================

def test_ap_perfect_detection():
    recalls = np.array([0.2, 0.5, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0])
    ap = calculate_all_points_interpolated_ap(recalls, precisions)
    assert pytest.approx(ap, rel=1e-4) == 1.0


def test_ap_zero_detection():
    recalls = np.array([0.0, 0.0])
    precisions = np.array([0.0, 0.0])
    ap = calculate_all_points_interpolated_ap(recalls, precisions)
    assert ap == 0.0


def test_greedy_matching_duplicate_prediction_penalized_as_fp():
    """Verify that multiple predictions overlapping the same GT box yield exactly one TP and one FP."""
    pred_boxes = [
        (10.0, 10.0, 50.0, 50.0),  # Conf 0.90 - matches GT
        (12.0, 12.0, 52.0, 52.0)   # Conf 0.85 - duplicate prediction on same GT
    ]
    pred_scores = [0.90, 0.85]
    pred_images = ["img_1", "img_1"]

    gt_boxes = [
        (10.0, 10.0, 50.0, 50.0)
    ]
    gt_images = ["img_1"]

    tp, fp, num_gt = match_detections_for_class(
        pred_boxes, pred_scores, pred_images, gt_boxes, gt_images, iou_threshold=0.5
    )

    assert num_gt == 1
    # Sorted order: rank 0 (conf 0.90) is TP, rank 1 (conf 0.85) is FP
    assert np.array_equal(tp, [1.0, 0.0])
    assert np.array_equal(fp, [0.0, 1.0])


def test_greedy_matching_multiple_gt_boxes():
    """Verify greedy matching across multiple GT boxes in the same image."""
    gt_boxes = [
        (0.0, 0.0, 50.0, 50.0),      # GT 1
        (100.0, 100.0, 150.0, 150.0) # GT 2
    ]
    gt_images = ["img_1", "img_1"]

    pred_boxes = [
        (0.0, 0.0, 50.0, 50.0),      # Conf 0.95 -> matches GT 1 (TP)
        (100.0, 100.0, 150.0, 150.0),# Conf 0.90 -> matches GT 2 (TP)
        (20.0, 20.0, 30.0, 30.0)     # Conf 0.70 -> IoU with GT 1 is (100/2500) = 0.04 < 0.5 (FP)
    ]
    pred_scores = [0.95, 0.90, 0.70]
    pred_images = ["img_1", "img_1", "img_1"]

    tp, fp, num_gt = match_detections_for_class(
        pred_boxes, pred_scores, pred_images, gt_boxes, gt_images, iou_threshold=0.5
    )

    assert num_gt == 2
    assert np.array_equal(tp, [1.0, 1.0, 0.0])
    assert np.array_equal(fp, [0.0, 0.0, 1.0])


def test_greedy_matching_confidence_ties_deterministic():
    """Verify that predictions with identical confidence scores break ties deterministically."""
    gt_boxes = [(10.0, 10.0, 50.0, 50.0)]
    gt_images = ["img_1"]

    pred_boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (11.0, 11.0, 51.0, 51.0)
    ]
    pred_scores = [0.80, 0.80]  # Exact tie
    pred_images = ["img_1", "img_1"]

    tp, fp, num_gt = match_detections_for_class(
        pred_boxes, pred_scores, pred_images, gt_boxes, gt_images, iou_threshold=0.5
    )

    assert tp.sum() == 1.0
    assert fp.sum() == 1.0


def test_evaluate_class_zero_ground_truth():
    """Verify handling when a class has predictions but zero ground truth instances."""
    pred_boxes = [(10.0, 10.0, 50.0, 50.0)]
    pred_scores = [0.90]
    pred_images = ["img_1"]

    res = evaluate_detections_for_class(
        pred_boxes, pred_scores, pred_images, gt_boxes=[], gt_image_ids=[]
    )
    assert np.isnan(res["ap"])
    assert res["num_gt"] == 0
    assert res["num_fp"] == 1


def test_evaluate_dataset_detections_summary():
    """Verify end-to-end mAP@50 and mAP@50:95 calculation across classes."""
    all_gt = {
        0: [(10.0, 10.0, 50.0, 50.0)],
        1: [(20.0, 20.0, 80.0, 80.0)]
    }
    all_gt_imgs = {0: ["img_1"], 1: ["img_2"]}

    all_pred = {
        0: [(10.0, 10.0, 50.0, 50.0)],
        1: [(20.0, 20.0, 80.0, 80.0)]
    }
    all_pred_scores = {0: [0.95], 1: [0.92]}
    all_pred_imgs = {0: ["img_1"], 1: ["img_2"]}

    res = evaluate_dataset_detections(
        all_pred, all_pred_scores, all_pred_imgs,
        all_gt, all_gt_imgs,
        class_names={0: "crazing", 1: "scratches"}
    )

    assert pytest.approx(res["mAP50"], rel=1e-4) == 1.0
    assert pytest.approx(res["mAP50_95"], rel=1e-4) == 1.0


# =====================================================================
# 4. Bounding Box Area Union & Overlap Tests
# =====================================================================

def test_compute_bbox_union_area_disjoint():
    # Two disjoint boxes of size 10x10 -> total area should be exactly 200 pixels
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (20.0, 20.0, 30.0, 30.0)
    ]
    union_px, union_ratio = compute_bbox_union_area(boxes, img_w=100, img_h=100)
    assert union_px == 200.0
    assert pytest.approx(union_ratio, rel=1e-5) == 200.0 / 10000.0


def test_compute_bbox_union_area_overlapping():
    # Box 1: [0, 0, 10, 10] -> area 100
    # Box 2: [5, 0, 15, 10] -> area 100
    # Overlap: [5, 0] to [10, 10] -> area 50
    # Union area should be 100 + 100 - 50 = 150 (NOT 200!)
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (5.0, 0.0, 15.0, 10.0)
    ]
    union_px, union_ratio = compute_bbox_union_area(boxes, img_w=100, img_h=100)
    assert union_px == 150.0
    assert pytest.approx(union_ratio, rel=1e-5) == 150.0 / 10000.0


# =====================================================================
# 5. YOLO Detector Inference Engine Tests
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
    assert isinstance(response.total_bbox_union_area_pixels, float)
    assert isinstance(response.total_bbox_union_area_ratio, float)
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
        bbox_area_pixels=3600.0,
        bbox_area_ratio=0.09
    )

    annotated = detector.draw_detections(dummy_img, [mock_detection])
    assert annotated.shape == dummy_img.shape
    assert annotated.dtype == dummy_img.dtype
    assert np.any(annotated > 0)
