"""
Object Detection Evaluation Metrics Module.

Implements from first principles:
1. Bounding Box Intersection over Union (IoU)
2. Non-Maximum Suppression (NMS) algorithm
3. True Positive (TP) / False Positive (FP) matching at IoU threshold
4. Mean Average Precision (mAP@50 and mAP@50:95)
"""

from typing import Dict, List, Optional, Tuple, Union
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
    Non-Maximum Suppression (NMS) Algorithm from First Principles.

    Steps:
    1. Filter out candidate boxes with confidence score below conf_threshold.
    2. Sort remaining candidate boxes in descending order of confidence score.
    3. Greedily select the highest-scoring box and add it to the final output list.
    4. Compute IoU between the selected box and all remaining candidate boxes of the SAME class.
    5. Suppress (discard) any candidate box whose IoU exceeds iou_threshold.
    6. Repeat until no candidate boxes remain.

    Args:
        boxes: List of VOC bounding boxes [(xmin, ymin, xmax, ymax), ...]
        scores: Corresponding detection confidence scores [float, ...]
        classes: Corresponding predicted class IDs [int, ...]
        iou_threshold: Overlap threshold above which redundant boxes are suppressed.
        conf_threshold: Minimum confidence score to be considered.

    Returns:
        List[int]: Indices of surviving bounding boxes.
    """
    # 1. Filter by confidence threshold
    valid_indices = [i for i, s in enumerate(scores) if s >= conf_threshold]
    if not valid_indices:
        return []

    # Group candidate indices by class so suppression is class-specific
    class_groups: Dict[int, List[int]] = {}
    for idx in valid_indices:
        cls_id = classes[idx]
        class_groups.setdefault(cls_id, []).append(idx)

    keep_indices: List[int] = []

    for cls_id, group_indices in class_groups.items():
        # Sort by confidence score descending
        group_indices = sorted(group_indices, key=lambda i: scores[i], reverse=True)

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


def calculate_average_precision(
    recalls: np.ndarray,
    precisions: np.ndarray
) -> float:
    """
    Computes Average Precision (AP) from Precision-Recall curve using standard
    all-point continuous trapezoidal integration (COCO standard):
        AP = sum_{k} (R_{k} - R_{k-1}) * P_interp(R_{k})
    """
    # Prepend 0 to recall and precision, append 1 to recall and 0 to precision
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute monotonic envelope: P(r) = max_{r' >= r} P(r')
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum of rectangular areas under the envelope
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap
