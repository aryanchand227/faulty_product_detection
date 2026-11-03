"""
Industrial Anomaly Detection Evaluation Metrics.

Implements standard benchmark metrics for unsupervised quality inspection:
1. Image-level AUROC: Measures discrimination between normal and defective parts.
2. Pixel-level AUROC: Measures spatial localization accuracy of defect heatmaps.
3. Area Under Per-Region Overlap (AUPRO): Evaluates per-defect component recall
   bounded at a maximum False Positive Rate (typically FPR <= 0.30) to prevent
   large defects from dominating the score over tiny cracks.
4. Decision Threshold Optimization: Calculates F1-optimal and target-FPR triage thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from sklearn import metrics


@dataclass
class AnomalyEvaluationResult:
    """Summary of comprehensive anomaly inspection performance."""
    image_auroc: float
    pixel_auroc: float
    aupro: Optional[float]
    optimal_threshold: float
    f1_score: float
    precision: float
    recall: float
    confusion_matrix: Dict[str, int]  # {"tp": int, "fp": int, "tn": int, "fn": int}


def compute_image_auroc(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray]
) -> float:
    """
    Computes image-level Area Under the Receiver Operating Characteristic curve.

    Args:
        y_true: Binary ground-truth labels (0: normal, 1: defective).
        y_scores: Continuous anomaly scores (e.g. max patch distance).

    Returns:
        float: Image AUROC score in [0.0, 1.0].
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_scores, dtype=np.float32)

    # Edge cases: single class present in evaluation set
    if len(np.unique(y_t)) < 2:
        return 1.0 if np.all(y_t == 0) or np.all(y_t == 1) else 0.5

    return float(metrics.roc_auc_score(y_t, y_s))


def compute_pixel_auroc(
    ground_truth_masks: Union[List[np.ndarray], np.ndarray],
    anomaly_maps: Union[List[np.ndarray], np.ndarray]
) -> float:
    """
    Computes pixel-level Area Under the Receiver Operating Characteristic curve.

    Args:
        ground_truth_masks: Array or list of binary masks (H, W) in {0, 1}.
        anomaly_maps: Array or list of continuous anomaly score heatmaps (H, W).

    Returns:
        float: Pixel AUROC score in [0.0, 1.0].
    """
    y_true_flat = np.concatenate([np.asarray(m).ravel() > 0 for m in ground_truth_masks]).astype(np.int32)
    y_score_flat = np.concatenate([np.asarray(m).ravel() for m in anomaly_maps]).astype(np.float32)

    if len(np.unique(y_true_flat)) < 2:
        return 1.0

    # Downsample points for evaluation if > 10,000,000 pixels to maintain test speed
    if len(y_true_flat) > 5_000_000:
        step = len(y_true_flat) // 2_000_000
        y_true_flat = y_true_flat[::step]
        y_score_flat = y_score_flat[::step]

    return float(metrics.roc_auc_score(y_true_flat, y_score_flat))


def compute_aupro(
    ground_truth_masks: Union[List[np.ndarray], np.ndarray],
    anomaly_maps: Union[List[np.ndarray], np.ndarray],
    max_fpr: float = 0.30,
    num_thresholds: int = 200
) -> float:
    """
    Computes Area Under Per-Region Overlap (AUPRO) curve up to max_fpr.

    Per-Region Overlap weights each connected defect component equally, regardless
    of whether it is a 5-pixel scratch or a 10,000-pixel broken fragment.

    Args:
        ground_truth_masks: List/Array of binary ground-truth masks.
        anomaly_maps: List/Array of continuous anomaly score heatmaps.
        max_fpr: Upper integration limit for False Positive Rate (default 0.30).
        num_thresholds: Number of discrete evaluation thresholds across the score range.

    Returns:
        float: AUPRO score in [0.0, 1.0].
    """
    gt_list = [np.asarray(m, dtype=np.uint8) > 0 for m in ground_truth_masks]
    score_list = [np.asarray(m, dtype=np.float32) for m in anomaly_maps]

    # 1. Identify all connected defect regions across all images
    defect_regions: List[Tuple[int, np.ndarray, int]] = []  # (image_idx, component_mask, area)
    total_normal_pixels = 0

    for img_idx, mask in enumerate(gt_list):
        total_normal_pixels += int(np.sum(~mask))
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for label_id in range(1, num_labels):
            comp_mask = (labels == label_id)
            comp_area = int(np.sum(comp_mask))
            if comp_area > 0:
                defect_regions.append((img_idx, comp_mask, comp_area))

    if not defect_regions or total_normal_pixels == 0:
        return 1.0

    # 2. Select evaluation thresholds across the anomaly score range
    all_scores = np.concatenate([s.ravel() for s in score_list])
    min_score, max_score = float(np.min(all_scores)), float(np.max(all_scores))
    if min_score >= max_score:
        return 1.0

    thresholds = np.linspace(max_score, min_score, num_thresholds)

    fpr_list: List[float] = [0.0]
    pro_list: List[float] = [0.0]

    for thresh in thresholds:
        # Binarize anomaly maps at this threshold
        binary_preds = [s >= thresh for s in score_list]

        # False Positive Rate = FP / (FP + TN)
        fp_pixels = sum(int(np.sum(pred & ~gt)) for pred, gt in zip(binary_preds, gt_list))
        fpr = fp_pixels / total_normal_pixels

        # Per-Region Overlap = mean(TP_component / component_area)
        region_overlaps: List[float] = []
        for img_idx, comp_mask, comp_area in defect_regions:
            tp_comp = int(np.sum(binary_preds[img_idx] & comp_mask))
            region_overlaps.append(tp_comp / comp_area)

        pro = float(np.mean(region_overlaps))

        fpr_list.append(fpr)
        pro_list.append(pro)

        if fpr >= max_fpr:
            break

    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)

    # Sort monotonically by FPR
    sort_idx = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sort_idx]
    pro_arr = pro_arr[sort_idx]

    # Clip to max_fpr
    valid_mask = fpr_arr <= max_fpr
    fpr_valid = fpr_arr[valid_mask]
    pro_valid = pro_arr[valid_mask]

    # If maximum FPR reached is below max_fpr, extend with last known PRO
    if len(fpr_valid) > 0 and fpr_valid[-1] < max_fpr:
        fpr_valid = np.append(fpr_valid, max_fpr)
        pro_valid = np.append(pro_valid, pro_valid[-1])

    if len(fpr_valid) < 2:
        return 0.0

    # Trapezoidal integration divided by max_fpr for normalization to [0, 1]
    raw_integral = float(np.trapezoid(pro_valid, fpr_valid)) if hasattr(np, "trapezoid") else float(np.trapz(pro_valid, fpr_valid))
    aupro = raw_integral / max_fpr
    return float(np.clip(aupro, 0.0, 1.0))


def find_optimal_threshold(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    criterion: str = "f1"
) -> Tuple[float, float, float, float, Dict[str, int]]:
    """
    Finds the optimal decision threshold and computes corresponding confusion matrix.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_scores: Continuous anomaly scores.
        criterion: 'f1' (maximize F1 score) or 'balanced_acc'.

    Returns:
        Tuple: (best_threshold, best_f1, precision, recall, confusion_matrix)
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_scores, dtype=np.float32)

    unique_scores = np.unique(y_s)
    if len(unique_scores) == 1:
        return float(unique_scores[0]), 1.0, 1.0, 1.0, {"tp": int(np.sum(y_t == 1)), "fp": 0, "tn": int(np.sum(y_t == 0)), "fn": 0}

    # Evaluate percentiles to test candidate thresholds
    threshold_candidates = np.percentile(unique_scores, np.linspace(5, 95, 100))

    best_thresh = float(threshold_candidates[0])
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for thresh in threshold_candidates:
        preds = (y_s >= thresh).astype(np.int32)

        tp = int(np.sum((preds == 1) & (y_t == 1)))
        fp = int(np.sum((preds == 1) & (y_t == 0)))
        tn = int(np.sum((preds == 0) & (y_t == 0)))
        fn = int(np.sum((preds == 0) & (y_t == 1)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_prec = prec
            best_rec = rec
            best_cm = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    return best_thresh, best_f1, best_prec, best_rec, best_cm


def evaluate_anomaly_detection(
    y_true_labels: Union[List[int], np.ndarray],
    y_image_scores: Union[List[float], np.ndarray],
    ground_truth_masks: Optional[Union[List[np.ndarray], np.ndarray]] = None,
    anomaly_maps: Optional[Union[List[np.ndarray], np.ndarray]] = None,
    max_fpr: float = 0.30
) -> AnomalyEvaluationResult:
    """
    Runs complete evaluation suite across image-level and pixel-level metrics.
    """
    img_auroc = compute_image_auroc(y_true_labels, y_image_scores)
    thresh, f1, prec, rec, cm = find_optimal_threshold(y_true_labels, y_image_scores)

    pix_auroc = 0.0
    aupro_val = None

    if ground_truth_masks is not None and anomaly_maps is not None:
        pix_auroc = compute_pixel_auroc(ground_truth_masks, anomaly_maps)
        aupro_val = compute_aupro(ground_truth_masks, anomaly_maps, max_fpr=max_fpr)

    return AnomalyEvaluationResult(
        image_auroc=round(img_auroc, 4),
        pixel_auroc=round(pix_auroc, 4),
        aupro=round(aupro_val, 4) if aupro_val is not None else None,
        optimal_threshold=round(thresh, 4),
        f1_score=round(f1, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        confusion_matrix=cm
    )

