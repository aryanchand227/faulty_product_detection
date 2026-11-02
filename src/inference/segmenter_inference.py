"""
U-Net Semantic Segmentation Inference Engine for Industrial Inspection.

Provides a production-grade inference interface for:
- Loading trained U-Net checkpoint weights
- Running forward inference to predict pixel-level defect masks
- Computing defect area metrics, percentage surface damage, and connected defect components
- Generating semi-transparent visual color overlays and contour boundaries
"""

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch

from src.models.segmentation.unet import UNet
from src.data.preprocessing import load_image, to_tensor, normalize_tensor

logger = logging.getLogger(__name__)


@dataclass
class DefectComponent:
    """Represents an individual connected defect blob on the product surface."""
    component_id: int
    area_pixels: int
    area_ratio: float
    centroid: Tuple[float, float]
    bounding_box_xywh: Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class SegmentationResponse:
    """Structured inspection payload for pixel-level semantic defect segmentation."""
    image_width: int
    image_height: int
    is_defective: bool
    total_defect_pixels: int
    defect_area_ratio: float                     # Fraction of total product surface damaged
    num_defect_components: int
    inference_time_ms: float
    components: List[DefectComponent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UNetSegmenter:
    """
    Industrial Defect Semantic Segmentation Engine.

    Encapsulates PyTorch U-Net inference, pixel-level probability thresholding,
    connected-component defect blob extraction, and visual overlay blending.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        target_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
        base_channels: int = 32
    ) -> None:
        self.device = torch.device(device)
        self.target_size = target_size
        self.threshold = float(threshold)
        self.base_channels = base_channels

        self.model = UNet(
            in_channels=3,
            num_classes=1,
            base_channels=base_channels,
            bilinear=True
        ).to(self.device)

        if model_path is not None:
            model_p = Path(model_path).resolve()
            if model_p.exists():
                logger.info(f"Loading U-Net weights from: {model_p} on device: {self.device}")
                checkpoint = torch.load(model_p, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint)
                else:
                    self.model = checkpoint
            else:
                logger.warning(f"Checkpoint not found at {model_p}; initialized with random weights.")

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        threshold: Optional[float] = None
    ) -> Tuple[SegmentationResponse, np.ndarray, np.ndarray]:
        """
        Executes pixel segmentation on an image.

        Args:
            image_input: File path, PIL Image, or NumPy array (RGB/BGR).
            threshold: Optional threshold override.

        Returns:
            Tuple containing:
            - SegmentationResponse dataclass
            - Binary mask (uint8, {0, 255}) resized to original image dimensions
            - Probability map (float32, [0, 1]) resized to original image dimensions
        """
        thresh = threshold if threshold is not None else self.threshold

        # Load or read original image to preserve dimensions
        if isinstance(image_input, (str, Path)):
            img_p = Path(image_input)
            if not img_p.exists():
                raise FileNotFoundError(f"Inspection image not found at: {img_p}")
            orig_bgr = cv2.imread(str(img_p))
            if orig_bgr is None:
                raise ValueError(f"Failed to load image from: {img_p}")
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

        # Preprocess to target size and normalize
        resized = cv2.resize(orig_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = to_tensor(resized)
        img_tensor = normalize_tensor(img_tensor).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        # Forward pass
        logits = self.model(img_tensor)
        probs_tensor = torch.sigmoid(logits).squeeze().cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Resize probability map and binary mask back to original image dimensions
        prob_map_orig = cv2.resize(probs_tensor, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask_orig = (prob_map_orig >= thresh).astype(np.uint8) * 255

        # Extract connected defect components
        total_defect_px = int(np.count_nonzero(binary_mask_orig))
        total_pixels = orig_w * orig_h
        defect_area_ratio = float(total_defect_px / total_pixels) if total_pixels > 0 else 0.0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask_orig, connectivity=8
        )

        components: List[DefectComponent] = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = int(stats[i, cv2.CC_STAT_AREA])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            components.append(DefectComponent(
                component_id=i,
                area_pixels=area,
                area_ratio=round(area / total_pixels, 6) if total_pixels > 0 else 0.0,
                centroid=(round(cx, 2), round(cy, 2)),
                bounding_box_xywh=(x, y, w, h)
            ))

        response = SegmentationResponse(
            image_width=orig_w,
            image_height=orig_h,
            is_defective=total_defect_px > 0,
            total_defect_pixels=total_defect_px,
            defect_area_ratio=round(defect_area_ratio, 6),
            num_defect_components=len(components),
            inference_time_ms=round(elapsed_ms, 2),
            components=components
        )

        return response, binary_mask_orig, prob_map_orig

    def draw_segmentation_overlay(
        self,
        image: np.ndarray,
        binary_mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 220),  # Crimson Red in BGR
        alpha: float = 0.45
    ) -> np.ndarray:
        """
        Blends semi-transparent color overlay over defect pixels and traces crisp contours.

        Args:
            image: Original image (BGR).
            binary_mask: Single-channel uint8 mask {0, 255}.
            color: Color tuple (B, G, R) for defect overlay.
            alpha: Transparency factor in [0, 1].

        Returns:
            np.ndarray: Blended inspection image with visible defect boundaries.
        """
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        defect_pixels = binary_mask > 0

        if np.any(defect_pixels):
            # 1. Semi-transparent color blend
            overlay = annotated.copy()
            overlay[defect_pixels] = color
            cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

            # 2. Draw crisp contour outlines around defect boundaries
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, thickness=2, lineType=cv2.LINE_AA)

        return annotated
