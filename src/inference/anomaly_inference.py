"""
PatchCore Anomaly Detection Inference Engine.

Performs real-time industrial surface inspection:
1. Computes image-level anomaly score and compares against calibrated decision thresholds.
2. Generates continuous pixel anomaly heatmaps.
3. Classifies product status into three industrial triage tiers:
   - PASS: Normal / nominal surface.
   - REVIEW: Marginal / borderline anomaly requiring secondary inspection.
   - FAIL: High-confidence defect, rejected from production line.
4. Calculates physical defect dimensions via camera calibration factor (mm_per_pixel).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import cv2
import numpy as np
from PIL import Image
import torch

from src.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    normalize_tensor,
    to_tensor,
)
from src.models.anomaly_detection.patchcore import PatchCore


@dataclass
class AnomalyRegion:
    """Connected component anomaly region details."""
    region_id: int
    area_pixels: int
    area_ratio: float
    centroid: Tuple[float, float]
    bounding_box_xywh: Tuple[int, int, int, int]
    physical_area_mm2: Optional[float]


@dataclass
class AnomalyInspectionResponse:
    """Complete industrial inspection triage report."""
    decision: str  # "PASS", "REVIEW", or "FAIL"
    image_anomaly_score: float
    threshold: float
    is_anomalous: bool
    defect_pixel_count: int
    defect_area_ratio: float
    physical_defect_area_mm2: Optional[float]
    inference_time_ms: float
    num_anomaly_regions: int
    regions: List[AnomalyRegion]


class PatchCoreInferencer:
    """
    Industrial inference engine for PatchCore anomaly inspection.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        patchcore_instance: Optional[PatchCore] = None,
        device: str = "cpu",
        target_size: Tuple[int, int] = (224, 224),
        default_threshold: float = 0.75,
        review_margin: float = 0.15
    ) -> None:
        self.device = torch.device(device)
        self.target_size = target_size
        self.default_threshold = default_threshold
        self.review_margin = review_margin

        if patchcore_instance is not None:
            self.model = patchcore_instance
        elif model_path is not None:
            self.model = PatchCore(device=self.device)
            self.model.load(model_path)
        else:
            # Fallback initialization (e.g. for testing)
            self.model = PatchCore(device=self.device)

        self.threshold = getattr(self.model, "threshold", 0.0)
        if self.threshold <= 0.0:
            self.threshold = default_threshold

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        threshold: Optional[float] = None,
        review_margin: Optional[float] = None,
        mm_per_pixel: Optional[float] = None,
        sigma: float = 4.0,
        score_method: str = "max",
        top_k: int = 5
    ) -> Tuple[AnomalyInspectionResponse, np.ndarray, np.ndarray]:
        """
        Executes anomaly inspection on an input product image.

        Args:
            image_input: File path, PIL Image, or NumPy array (RGB/BGR).
            threshold: Decision threshold override.
            review_margin: Fraction around threshold classified as "REVIEW".
            mm_per_pixel: Camera calibration factor in mm per pixel.
            sigma: Standard deviation for Gaussian smoothing.
            score_method: 'max' or 'top_k'.
            top_k: Number of highest-scoring patches to average if score_method='top_k'.

        Returns:
            Tuple containing:
            - AnomalyInspectionResponse dataclass
            - Binary mask (uint8 {0, 255}) at original image resolution
            - Anomaly score heatmap (float32 [0.0, max_score]) at original image resolution
        """
        thresh = threshold if threshold is not None else self.threshold
        margin = review_margin if review_margin is not None else self.review_margin

        # 1. Parse and validate image
        if isinstance(image_input, (str, Path)):
            img_p = Path(image_input)
            if not img_p.exists():
                raise FileNotFoundError(f"Input image not found: {img_p}")
            orig_bgr = cv2.imread(str(img_p))
            if orig_bgr is None:
                raise ValueError(f"Failed to decode image: {img_p}")
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = orig_rgb.shape[:2]
        elif isinstance(image_input, Image.Image):
            orig_w, orig_h = image_input.size
            orig_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            orig_h, orig_w = image_input.shape[:2]
            orig_rgb = image_input if len(image_input.shape) == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 2. Resize and standardize
        resized = cv2.resize(orig_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = to_tensor(resized)
        img_tensor = normalize_tensor(img_tensor, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        # 3. Model forward pass (if memory bank is initialized)
        if self.model.memory_bank is not None:
            patch_dists, img_scores, spatial_hw = self.model.predict_patches(
                img_tensor, score_method=score_method, top_k=top_k
            )
            img_score = float(img_scores[0].item())
            # Generate smoothed anomaly map at original resolution
            anomaly_map = self.model.compute_anomaly_maps(
                patch_dists, spatial_hw, output_hw=(orig_h, orig_w), sigma=sigma
            )[0]
        else:
            # Fallback for un-fitted model (testing)
            img_score = 0.0
            anomaly_map = np.zeros((orig_h, orig_w), dtype=np.float32)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Binary anomaly mask thresholding
        binary_mask = (anomaly_map >= thresh).astype(np.uint8) * 255
        defect_pixels = int(np.count_nonzero(binary_mask))
        total_pixels = orig_w * orig_h
        defect_ratio = float(defect_pixels / total_pixels) if total_pixels > 0 else 0.0

        # 5. Connected components analysis for localized defect regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        regions: List[AnomalyRegion] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            comp_phys_mm2 = round(area * (float(mm_per_pixel) ** 2), 3) if mm_per_pixel is not None else None
            regions.append(AnomalyRegion(
                region_id=i,
                area_pixels=area,
                area_ratio=round(area / total_pixels, 6) if total_pixels > 0 else 0.0,
                centroid=(round(cx, 2), round(cy, 2)),
                bounding_box_xywh=(x, y, w, h),
                physical_area_mm2=comp_phys_mm2
            ))

        total_phys_mm2 = round(defect_pixels * (float(mm_per_pixel) ** 2), 2) if mm_per_pixel is not None else None

        # 6. Industrial Triage Decision Logic:
        # - PASS: img_score < lower_bound and defect_pixels == 0
        # - REVIEW: within uncertainty band around threshold
        # - FAIL: img_score >= upper_bound or significant defect area
        lower_bound = thresh * (1.0 - margin)
        upper_bound = thresh * (1.0 + margin)

        if img_score >= upper_bound or defect_ratio > 0.02:
            decision = "FAIL"
            is_anomalous = True
        elif img_score >= lower_bound:
            decision = "REVIEW"
            is_anomalous = True
        else:
            decision = "PASS"
            is_anomalous = False

        response = AnomalyInspectionResponse(
            decision=decision,
            image_anomaly_score=round(img_score, 4),
            threshold=round(thresh, 4),
            is_anomalous=is_anomalous,
            defect_pixel_count=defect_pixels,
            defect_area_ratio=round(defect_ratio, 6),
            physical_defect_area_mm2=total_phys_mm2,
            inference_time_ms=round(elapsed_ms, 2),
            num_anomaly_regions=len(regions),
            regions=regions
        )

        return response, binary_mask, anomaly_map

    @staticmethod
    def draw_anomaly_overlay(
        image_rgb: np.ndarray,
        anomaly_map: np.ndarray,
        threshold: float,
        alpha: float = 0.50
    ) -> np.ndarray:
        """
        Draws visual inspection overlay combining jet colormap with original image.

        Args:
            image_rgb: Original RGB image array (H, W, 3).
            anomaly_map: Anomaly heatmap (H, W) with continuous scores.
            threshold: Anomaly threshold for defect contouring.
            alpha: Heatmap blend transparency.

        Returns:
            np.ndarray: Annotated RGB image (H, W, 3).
        """
        vis_img = image_rgb.copy()
        h, w = vis_img.shape[:2]

        # Normalize heatmap to [0, 255]
        max_val = max(float(np.max(anomaly_map)), threshold * 1.5, 1e-6)
        norm_map = np.clip((anomaly_map / max_val) * 255.0, 0, 255).astype(np.uint8)

        # Apply Jet colormap
        heatmap_bgr = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend heatmap onto image where score > threshold * 0.5
        mask_interest = (anomaly_map >= (threshold * 0.5))[:, :, np.newaxis]
        vis_img = np.where(mask_interest, (vis_img * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8), vis_img)

        # Draw red contours around defect regions
        binary = (anomaly_map >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)

        return vis_img

