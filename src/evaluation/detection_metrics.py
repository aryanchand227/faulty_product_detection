"""
Object Detection Evaluation Metrics Module.

Implements from first principles:
1. Bounding Box Intersection over Union (IoU)
2. Class-Aware Non-Maximum Suppression (NMS)
3. Greedy Bipartite Prediction-to-Ground-Truth Matching at IoU threshold:
   - Confidence-descending priority
   - Strict one-to-one ground truth matching (duplicates penalized as False Positives)
   - Proper handling of classes with zero ground-truth instances
4. All-Points Interpolated Average Precision (AP):
   - Right-endpoint step/rectangle integration under the monotonic precision envelope
     AP = sum_k (R_k - R_{k-1}) * P_interp(R_k)
   - (Note: This is step/rectangle integration, NOT trapezoidal integration)
5. Mean Average Precision (mAP@50 and COCO mAP@50:95)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


def compute_iou_voc(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes in VOC format:
    box = (xmin, ymin, xmax, ymax)

    IoU = Area(Overlap) / Area(Union)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area

    if union_area <= 0.0:
        return 0.0

    return float(intersection_area / union_area)


def non_maximum_suppression(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    classes: List[int],
    iou_threshold: float = 0.45,
    conf_threshold: float = 0.25
) -> List[int]:
    """
    Class-Aware Non-Maximum Suppression (NMS) Algorithm from First Principles.

    Steps:
    1. Filter out candidate boxes with confidence score below conf_threshold.
    2. Group remaining candidates by class ID (class-aware).
    3. Sort candidates within each class descending by confidence score.
    4. Greedily select the highest-scoring box and suppress overlapping boxes (IoU > iou_threshold).
    5. Repeat until no candidate boxes remain.

    Returns:
        List[int]: Indices of surviving bounding boxes in original list.
    """
    valid_indices = [i for i, s in enumerate(scores) if s >= conf_threshold]
    if not valid_indices:
        return []

    class_groups: Dict[int, List[int]] = {}
    for idx in valid_indices:
        cls_id = classes[idx]
        class_groups.setdefault(cls_id, []).append(idx)

    keep_indices: List[int] = []

    for cls_id, group_indices in class_groups.items():
        # Sort by confidence score descending, with index as secondary key for deterministic tie-breaking
        group_indices = sorted(group_indices, key=lambda i: (scores[i], -i), reverse=True)

        while group_indices:
            best_idx = group_indices.pop(0)
            keep_indices.append(best_idx)

            remaining = []
            for other_idx in group_indices:
                iou = compute_iou_voc(boxes[best_idx], boxes[other_idx])
                if iou <= iou_threshold:
                    remaining.append(other_idx)

            group_indices = remaining

    return sorted(keep_indices)


def calculate_all_points_interpolated_ap(
    recalls: np.ndarray,
    precisions: np.ndarray
) -> float:
    """
    Computes all-points interpolated Average Precision (AP) from a Precision-Recall curve.

    Mathematical Formulation:
    -------------------------
    1. Monotonic precision envelope:
       P_interp(r) = max_{r' >= r} P(r')
    2. Right-endpoint rectangle / step integration (COCO benchmark standard):
       AP = sum_{k} (R_{k} - R_{k-1}) * P_interp(R_{k})

    Note on Terminology:
    --------------------
    This is step/rectangle integration under the interpolated precision envelope,
    NOT trapezoidal integration. Trapezoidal integration ((P_k + P_{k-1})/2 * delta_R)
    would erroneously underestimate the continuous step envelope.

    Args:
        recalls: 1D array of recall points in [0, 1].
        precisions: 1D array of corresponding precision values in [0, 1].

    Returns:
        float: Calculated AP score in [0.0, 1.0].
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Prepend 0 to recall and precision, append 1 to recall and 0 to precision
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute monotonic envelope: P(r) = max_{r' >= r} P(r')
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum of rectangular step areas under the envelope
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return max(0.0, min(1.0, ap))


# Backwards-compatibility alias
calculate_average_precision = calculate_all_points_interpolated_ap


def match_detections_for_class(
    pred_boxes: List[Tuple[float, float, float, float]],
    pred_scores: List[float],
    pred_image_ids: List[str],
    gt_boxes: List[Tuple[float, float, float, float]],
    gt_image_ids: List[str],
    iou_threshold: float = 0.50
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Performs greedy bipartite matching between predictions and ground-truth boxes for a single class.

    Rules:
    1. Sort all predictions globally descending by confidence score.
    2. For each prediction, find the ground-truth box in the same image with the highest IoU.
    3. If highest IoU >= iou_threshold:
       - If that GT box has NOT been matched yet -> True Positive (TP = 1, FP = 0).
       - If that GT box was ALREADY matched by an earlier higher-confidence prediction -> False Positive (TP = 0, FP = 1).
    4. If highest IoU < iou_threshold -> False Positive (TP = 0, FP = 1).
    5. Predictions in images with no GT boxes are marked as False Positives.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: (tp_array, fp_array, total_num_gt)
    """
    num_gt = len(gt_boxes)
    num_preds = len(pred_boxes)

    if num_preds == 0:
        return np.array([]), np.array([]), num_gt

    # Group ground truth boxes by image_id
    gt_by_image: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for idx, (img_id, box) in enumerate(zip(gt_image_ids, gt_boxes)):
        gt_by_image.setdefault(img_id, []).append((idx, box))

    # Track which GT boxes have already been claimed (set of global GT indices)
    matched_gt_indices: Set[int] = set()

    # Sort prediction indices descending by confidence, breaking ties deterministically
    sorted_pred_indices = sorted(range(num_preds), key=lambda i: (pred_scores[i], -i), reverse=True)

    tp = np.zeros(num_preds, dtype=np.float32)
    fp = np.zeros(num_preds, dtype=np.float32)

    for rank, p_idx in enumerate(sorted_pred_indices):
        p_box = pred_boxes[p_idx]
        p_img = pred_image_ids[p_idx]

        # Candidate ground truths in this image
        candidates = gt_by_image.get(p_img, [])
        if not candidates:
            # No ground truth in this image -> False Positive
            fp[rank] = 1.0
            continue

        # Compute IoU between prediction and all GT boxes in this image
        # Sort candidates descending by IoU
        candidates_with_iou = [
            (gt_idx, compute_iou_voc(p_box, g_box))
            for gt_idx, g_box in candidates
        ]
        candidates_with_iou.sort(key=lambda x: x[1], reverse=True)

        # Greedily match with the highest-IoU UNMATCHED ground-truth box
        matched = False
        for gt_idx, iou in candidates_with_iou:
            if iou < iou_threshold:
                # Since candidates are sorted descending by IoU, no subsequent candidate will satisfy threshold
                break
            if gt_idx not in matched_gt_indices:
                # Valid match found: assign True Positive and claim ground truth
                matched_gt_indices.add(gt_idx)
                tp[rank] = 1.0
                matched = True
                break

        if not matched:
            # Either highest-IoU boxes were already claimed, or all had IoU < threshold -> False Positive
            fp[rank] = 1.0

    return tp, fp, num_gt


def evaluate_detections_for_class(
    pred_boxes: List[Tuple[float, float, float, float]],
    pred_scores: List[float],
    pred_image_ids: List[str],
    gt_boxes: List[Tuple[float, float, float, float]],
    gt_image_ids: List[str],
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates detections for a single defect class at a specified IoU threshold.

    Handles edge cases:
    - Zero ground truth instances: returns AP = float('nan'), documented as unrepresented class.
    - Zero predictions: returns AP = 0.0.

    Returns:
        Dict with keys: 'ap', 'precision', 'recall', 'num_gt', 'num_preds', 'num_tp', 'num_fp'
    """
    tp, fp, num_gt = match_detections_for_class(
        pred_boxes, pred_scores, pred_image_ids,
        gt_boxes, gt_image_ids, iou_threshold=iou_threshold
    )

    if num_gt == 0:
        return {
            "ap": float("nan"),
            "precision": np.array([]),
            "recall": np.array([]),
            "num_gt": 0,
            "num_preds": len(pred_boxes),
            "num_tp": 0,
            "num_fp": len(pred_boxes)
        }

    if len(tp) == 0:
        return {
            "ap": 0.0,
            "precision": np.array([]),
            "recall": np.array([]),
            "num_gt": num_gt,
            "num_preds": 0,
            "num_tp": 0,
            "num_fp": 0
        }

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / float(num_gt)
    precisions = cum_tp / (cum_tp + cum_fp)

    ap = calculate_all_points_interpolated_ap(recalls, precisions)

    return {
        "ap": round(ap, 4),
        "precision": precisions,
        "recall": recalls,
        "num_gt": num_gt,
        "num_preds": len(tp),
        "num_tp": int(cum_tp[-1]) if len(cum_tp) > 0 else 0,
        "num_fp": int(cum_fp[-1]) if len(cum_fp) > 0 else 0
    }


def evaluate_dataset_detections(
    all_pred_boxes: Dict[int, List[Tuple[float, float, float, float]]],
    all_pred_scores: Dict[int, List[float]],
    all_pred_image_ids: Dict[int, List[str]],
    all_gt_boxes: Dict[int, List[Tuple[float, float, float, float]]],
    all_gt_image_ids: Dict[int, List[str]],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive dataset evaluation:
    - Per-class AP@50 and mAP@50
    - Multi-threshold mAP@50:95 (averaged across IoU thresholds [0.50, 0.55, ..., 0.95])

    Returns:
        Dict[str, Any]: Telemetry report with per-class and mean metrics.
    """
    all_classes = sorted(list(set(all_gt_boxes.keys()) | set(all_pred_boxes.keys())))
    iou_thresholds = np.linspace(0.50, 0.95, 10)

    per_class_ap50: Dict[str, float] = {}
    per_class_map: Dict[str, float] = {}

    for cls_id in all_classes:
        c_name = class_names.get(cls_id, f"class_{cls_id}") if class_names else f"class_{cls_id}"

        p_boxes = all_pred_boxes.get(cls_id, [])
        p_scores = all_pred_scores.get(cls_id, [])
        p_imgs = all_pred_image_ids.get(cls_id, [])

        g_boxes = all_gt_boxes.get(cls_id, [])
        g_imgs = all_gt_image_ids.get(cls_id, [])

        # Evaluate at IoU 0.50
        res50 = evaluate_detections_for_class(p_boxes, p_scores, p_imgs, g_boxes, g_imgs, iou_threshold=0.50)
        per_class_ap50[c_name] = res50["ap"]

        # Evaluate across 10 IoU thresholds for COCO mAP@50:95
        ap_list = []
        for iou_t in iou_thresholds:
            res_t = evaluate_detections_for_class(p_boxes, p_scores, p_imgs, g_boxes, g_imgs, iou_threshold=float(iou_t))
            if not np.isnan(res_t["ap"]):
                ap_list.append(res_t["ap"])

        per_class_map[c_name] = round(float(np.mean(ap_list)), 4) if ap_list else float("nan")

    # Mean metrics across valid classes
    valid_ap50 = [v for v in per_class_ap50.values() if not np.isnan(v)]
    valid_map = [v for v in per_class_map.values() if not np.isnan(v)]

    map50 = round(float(np.mean(valid_ap50)), 4) if valid_ap50 else 0.0
    map50_95 = round(float(np.mean(valid_map)), 4) if valid_map else 0.0

    return {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "per_class_ap50": per_class_ap50,
        "per_class_mAP50_95": per_class_map
    }
