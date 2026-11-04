"""
Unified Multi-Tier Industrial Quality Inspection Pipeline.

Orchestrates the complete industrial vision pipeline across three specialized tiers:
- Tier 1: Unsupervised Anomaly Gatekeeper (PatchCore)
- Tier 2: Multi-Class Defect Classifier (ResNet-18 / EfficientNet-B0)
- Tier 3: Spatial Defect Localizer (Ultralytics YOLOv8 Bounding Boxes + PyTorch U-Net Segmentation)
- Camera Calibration: Metric physical dimension transformation (mm and mm²)
- Decision Engine: Multi-factor operational triage (PASS / REVIEW / FAIL) and pneumatic diverter routing
"""

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.inference.anomaly_inference import AnomalyInspectionResponse, PatchCoreInferencer
from src.inference.classifier_inference import ClassifierInferencer, DefectClassificationResponse
from src.inference.detector_inference import DetectionResponse, YOLODetector
from src.inference.segmenter_inference import SegmentationResponse, UNetSegmenter
from src.inspection.camera_calibration import CameraCalibrationConfig, CameraCalibrator
from src.inspection.decision_engine import (
    DecisionEngine,
    InspectionDecision,
    InspectionVerdict,
    QualityThresholds,
    RoutingAction,
    SeverityLevel,
)
from src.inspection.visualizer import InspectionVisualizer

logger = logging.getLogger(__name__)


@dataclass
class UnifiedInspectionResult:
    """
    Comprehensive multi-tier industrial quality inspection report.

    Attributes:
        verdict: High-level operational triage verdict (PASS / REVIEW / FAIL)
        tier1_anomaly: Telemetry from PatchCore anomaly gatekeeper
        tier2_classification: Optional defect categorization results from classifier
        tier3_detection: Optional bounding boxes from YOLO detector
        tier3_segmentation: Optional pixel segmentation masks from U-Net
        physical_metrics: Calibrated real-world damage dimensions in mm and mm²
        annotated_image: Optional RGB image with multi-tier annotations and industrial HUD banner
        total_latency_ms: Total end-to-end wall-clock inspection time in milliseconds
        early_exit_applied: True if Tier 1 gatekeeper bypassed Tiers 2 & 3 for nominal speedup
    """
    verdict: InspectionVerdict
    tier1_anomaly: AnomalyInspectionResponse
    tier2_classification: Optional[DefectClassificationResponse] = None
    tier3_detection: Optional[DetectionResponse] = None
    tier3_segmentation: Optional[SegmentationResponse] = None
    physical_metrics: Dict[str, Any] = field(default_factory=dict)
    annotated_image: Optional[np.ndarray] = None
    total_latency_ms: float = 0.0
    early_exit_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts inspection result to structured dictionary payload."""
        data: Dict[str, Any] = {
            "verdict": self.verdict.to_dict(),
            "tier1_anomaly": {
                "decision": self.tier1_anomaly.decision,
                "anomaly_score": self.tier1_anomaly.image_anomaly_score,
                "threshold": self.tier1_anomaly.threshold,
                "is_anomalous": self.tier1_anomaly.is_anomalous,
                "latency_ms": self.tier1_anomaly.inference_time_ms
            },
            "physical_metrics": self.physical_metrics,
            "total_latency_ms": self.total_latency_ms,
            "early_exit_applied": self.early_exit_applied
        }

        if self.tier2_classification is not None:
            data["tier2_classification"] = self.tier2_classification.to_dict()
        if self.tier3_detection is not None:
            data["tier3_detection"] = self.tier3_detection.to_dict()
        if self.tier3_segmentation is not None:
            data["tier3_segmentation"] = self.tier3_segmentation.to_dict()

        return data


class UnifiedQualityInspector:
    """
    Industrial Quality Inspector orchestrating multi-tier defect analysis.
    """

    def __init__(
        self,
        tier1_inferencer: Optional[PatchCoreInferencer] = None,
        tier2_inferencer: Optional[ClassifierInferencer] = None,
        tier3_detector: Optional[YOLODetector] = None,
        tier3_segmenter: Optional[UNetSegmenter] = None,
        calibrator: Optional[CameraCalibrator] = None,
        decision_engine: Optional[DecisionEngine] = None,
        thresholds: Optional[QualityThresholds] = None,
        mm_per_pixel: float = 0.10,
        early_exit: bool = True,
        device: str = "cpu"
    ) -> None:
        """
        Initializes the unified quality inspection pipeline.

        Args:
            tier1_inferencer: Initialized PatchCoreInferencer (or creates default).
            tier2_inferencer: Initialized ClassifierInferencer (or creates default).
            tier3_detector: Initialized YOLODetector (or creates mock/default).
            tier3_segmenter: Initialized UNetSegmenter (or creates default).
            calibrator: CameraCalibrator instance (or creates default with mm_per_pixel).
            decision_engine: DecisionEngine rule evaluator (or creates default).
            thresholds: QualityThresholds configuration for decision rules.
            mm_per_pixel: Physical calibration scale (default 0.10 mm/pixel).
            early_exit: If True, nominal items passing Tier 1 skip downstream models for speed.
            device: Hardware device target ('cpu', 'cuda').
        """
        self.device = device
        self.early_exit = early_exit

        # 1. Camera Calibrator
        self.calibrator = calibrator or CameraCalibrator(mm_per_pixel=mm_per_pixel)

        # 2. Decision Engine
        self.decision_engine = decision_engine or DecisionEngine(thresholds=thresholds)

        # 3. Vision Models
        self.tier1_inferencer = tier1_inferencer or PatchCoreInferencer(device=self.device)
        self.tier2_inferencer = tier2_inferencer or ClassifierInferencer(device=self.device)
        self.tier3_detector = tier3_detector  # Can be None if YOLO weights not installed
        self.tier3_segmenter = tier3_segmenter or UNetSegmenter(device=self.device)

    def inspect(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        render_visuals: bool = True,
        force_full_pipeline: bool = False
    ) -> UnifiedInspectionResult:
        """
        Executes end-to-end industrial inspection on a product image.

        Args:
            image_input: File path, PIL Image, or NumPy array.
            render_visuals: If True, renders annotated visual overlay with HUD banner.
            force_full_pipeline: If True, runs Tiers 2 & 3 even if Tier 1 reports nominal PASS.

        Returns:
            UnifiedInspectionResult: Complete inspection telemetry and verdict.
        """
        t_start = time.perf_counter()

        # Load RGB image array for visualization and geometry
        image_rgb = self._load_image_rgb(image_input)
        img_h, img_w = image_rgb.shape[:2]

        # =================================================================
        # TIER 1: PatchCore Anomaly Gatekeeper
        # =================================================================
        t1_response, anomaly_heatmap, _ = self.tier1_inferencer.predict(
            image_rgb,
            mm_per_pixel=self.calibrator.mm_per_pixel
        )

        anomaly_score = t1_response.image_anomaly_score
        tau = self.decision_engine.thresholds.anomaly_threshold
        delta = self.decision_engine.thresholds.anomaly_margin
        is_clear_nominal = anomaly_score < (tau * (1.0 - delta))

        # Check Early-Exit Condition (Conveyor High-Throughput Optimization)
        if self.early_exit and is_clear_nominal and not force_full_pipeline:
            verdict = self.decision_engine.evaluate(
                anomaly_score=anomaly_score,
                is_anomalous=False,
                defect_class="nominal",
                class_confidence=1.0 - (anomaly_score / tau if tau > 0 else 0.0),
                pixel_defect_area_mm2=0.0,
                bbox_union_area_mm2=0.0,
                num_defect_regions=0
            )

            total_elapsed = (time.perf_counter() - t_start) * 1000.0

            annotated_img = None
            if render_visuals:
                annotated_img = InspectionVisualizer.render_hud_overlay(
                    image_rgb=image_rgb,
                    verdict=verdict,
                    anomaly_heatmap=None,
                    boxes=None,
                    segmentation_mask=None,
                    latency_ms=total_elapsed
                )

            return UnifiedInspectionResult(
                verdict=verdict,
                tier1_anomaly=t1_response,
                tier2_classification=None,
                tier3_detection=None,
                tier3_segmentation=None,
                physical_metrics={
                    "total_defect_area_mm2": 0.0,
                    "mm_per_pixel": self.calibrator.mm_per_pixel,
                    "num_defects": 0
                },
                annotated_image=annotated_img,
                total_latency_ms=round(total_elapsed, 2),
                early_exit_applied=True
            )

        # =================================================================
        # TIER 2: Defect Classification (Morphology Categorization)
        # =================================================================
        t2_response = self.tier2_inferencer.predict(image_rgb)
        defect_class = t2_response.predicted_class_name
        class_conf = t2_response.confidence
        class_entropy = t2_response.entropy

        # =================================================================
        # TIER 3A: Defect Localization (YOLO Bounding Boxes)
        # =================================================================
        t3_boxes = []
        bbox_union_area_mm2 = 0.0
        t3_det_response = None

        if self.tier3_detector is not None:
            try:
                t3_det_response = self.tier3_detector.predict(
                    image_rgb,
                    mm_per_pixel=self.calibrator.mm_per_pixel
                )
                t3_boxes = t3_det_response.detections
                if t3_det_response.physical_union_area_mm2 is not None:
                    bbox_union_area_mm2 = t3_det_response.physical_union_area_mm2
                else:
                    bbox_union_area_mm2 = self.calibrator.pixel_area_to_mm2(
                        t3_det_response.total_bbox_union_area_pixels
                    )
            except Exception as e:
                logger.warning(f"Tier 3 YOLO detection encountered exception: {e}")

        # =================================================================
        # TIER 3B: Defect Segmentation (PyTorch U-Net Pixel Masks)
        # =================================================================
        t3_seg_response, seg_mask, _ = self.tier3_segmenter.predict(
            image_rgb,
            mm_per_pixel=self.calibrator.mm_per_pixel
        )

        pixel_defect_area_mm2 = (
            t3_seg_response.physical_defect_area_mm2
            if t3_seg_response.physical_defect_area_mm2 is not None
            else self.calibrator.pixel_area_to_mm2(t3_seg_response.total_defect_pixels)
        )

        num_regions = max(
            t3_seg_response.num_defect_components,
            len(t3_boxes),
            t1_response.num_anomaly_regions
        )

        # =================================================================
        # DECISION ENGINE: Multi-Tier Synthesis & Diverter Routing
        # =================================================================
        verdict = self.decision_engine.evaluate(
            anomaly_score=anomaly_score,
            is_anomalous=t1_response.is_anomalous,
            defect_class=defect_class,
            class_confidence=class_conf,
            class_entropy=class_entropy,
            pixel_defect_area_mm2=pixel_defect_area_mm2,
            bbox_union_area_mm2=bbox_union_area_mm2,
            num_defect_regions=num_regions
        )

        total_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Physical Metrics Payload
        physical_metrics = {
            "total_defect_area_mm2": round(verdict.physical_defect_area_mm2, 3),
            "pixel_defect_area_mm2": round(pixel_defect_area_mm2, 3),
            "bbox_union_area_mm2": round(bbox_union_area_mm2, 3),
            "num_defect_regions": num_regions,
            "mm_per_pixel": round(self.calibrator.mm_per_pixel, 5),
            "image_width_px": img_w,
            "image_height_px": img_h
        }

        # Render Industrial Visual Overlay
        annotated_img = None
        if render_visuals:
            annotated_img = InspectionVisualizer.render_hud_overlay(
                image_rgb=image_rgb,
                verdict=verdict,
                anomaly_heatmap=anomaly_heatmap,
                boxes=t3_boxes,
                segmentation_mask=seg_mask,
                latency_ms=total_elapsed
            )

        return UnifiedInspectionResult(
            verdict=verdict,
            tier1_anomaly=t1_response,
            tier2_classification=t2_response,
            tier3_detection=t3_det_response,
            tier3_segmentation=t3_seg_response,
            physical_metrics=physical_metrics,
            annotated_image=annotated_img,
            total_latency_ms=round(total_elapsed, 2),
            early_exit_applied=False
        )

    def _load_image_rgb(self, image_input: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Utility ensuring standard uint8 RGB array format."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Inspection input image not found: {p}")
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise ValueError(f"Failed to read image at: {p}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return image_input[:, :, :3]
            return image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
