"""
Unified Multi-Tier Industrial Quality Inspection Pipeline.

Orchestrates the industrial vision pipeline across three specialized tiers:
- Tier 1: Unsupervised Anomaly Gatekeeper (PatchCore)
- Tier 2: Multi-Class Defect Classifier (ResNet-18 / EfficientNet-B0)
- Tier 3: Spatial Defect Localizer (Ultralytics YOLOv8 Bounding Boxes + PyTorch U-Net Segmentation)
- Camera Calibration: Metric dimension transformation (mm and mm²)
- Decision Engine: Multi-factor operational triage (PASS / REVIEW / FAIL) and pneumatic diverter routing

Product Domain Separation:
--------------------------
1. GLASS_BOTTLE (MVTec AD Bottle Domain):
   - 3D curved geometry: Optical surface area is estimated projected sensor-plane area.
   - Tier 1 PatchCore is the primary gatekeeper and localization engine (0.9848 Pixel AUROC).
   - Prevents domain leakage by withholding steel-trained supervised classifiers from glass containers.
2. STEEL_SHEET (NEU-DET Steel Surface Domain):
   - Flat planar geometry: Planar metric scaling (Area = px * s^2) is mathematically rigorous.
   - Tier 2 Classifier + Tier 3 YOLO / U-Net localize and categorize steel defects.
3. GENERIC_AOI:
   - Configurable multi-tier inspection for general manufacturing lines.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.inference.anomaly_inference import AnomalyInspectionResponse, PatchCoreInferencer
from src.inference.classifier_inference import ClassifierInferencer, DefectClassificationResponse
from src.inference.detector_inference import DetectionBox, DetectionResponse, YOLODetector
from src.inference.segmenter_inference import DefectComponent, SegmentationResponse, UNetSegmenter
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


class ProductDomain(str, Enum):
    """Product domain configuration defining valid model combinations."""
    GLASS_BOTTLE = "GLASS_BOTTLE"        # 3D curved glassware (MVTec AD Bottle)
    STEEL_SHEET = "STEEL_SHEET"          # Flat planar steel strip (NEU-DET)
    GENERIC_AOI = "GENERIC_AOI"          # General optical inspection


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
        physical_metrics: Calibrated damage dimensions in mm and mm²
        annotated_image: Optional RGB image with multi-tier annotations and industrial HUD banner
        total_latency_ms: Total end-to-end wall-clock inspection time in milliseconds
        early_exit_applied: True if Tier 1 gatekeeper bypassed downstream heads for nominal speedup
        product_domain: Domain under which inspection was performed
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
    product_domain: str = "GLASS_BOTTLE"
    morphological_features: List[Dict[str, Any]] = field(default_factory=list)
    certificate: Optional[Dict[str, Any]] = None

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
            "early_exit_applied": self.early_exit_applied,
            "product_domain": self.product_domain,
            "morphological_features": self.morphological_features
        }

        if self.certificate is not None:
            data["certificate"] = self.certificate
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
        domain: ProductDomain = ProductDomain.GLASS_BOTTLE,
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
        Initializes the unified quality inspection pipeline with domain constraints.

        Args:
            domain: ProductDomain (GLASS_BOTTLE, STEEL_SHEET, GENERIC_AOI).
            tier1_inferencer: Initialized PatchCoreInferencer (or creates default).
            tier2_inferencer: Initialized ClassifierInferencer (or creates default for domain).
            tier3_detector: Initialized YOLODetector (or None if domain does not support).
            tier3_segmenter: Initialized UNetSegmenter (or None if domain does not support).
            calibrator: CameraCalibrator instance.
            decision_engine: DecisionEngine rule evaluator.
            thresholds: QualityThresholds configuration for decision rules.
            mm_per_pixel: Physical calibration scale (default 0.10 mm/pixel).
            early_exit: If True, nominal items passing Tier 1 skip downstream models for speed.
            device: Hardware device target ('cpu', 'cuda').
        """
        self.domain = domain
        self.device = device
        self.early_exit = early_exit

        # 1. Camera Calibrator
        self.calibrator = calibrator or CameraCalibrator(mm_per_pixel=mm_per_pixel)

        # 2. Decision Engine
        self.decision_engine = decision_engine or DecisionEngine(thresholds=thresholds)

        # 3. Vision Models
        self.tier1_inferencer = tier1_inferencer or PatchCoreInferencer(device=self.device)
        self.tier2_inferencer = tier2_inferencer or ClassifierInferencer(device=self.device)
        self.tier3_detector = tier3_detector
        self.tier3_segmenter = tier3_segmenter or UNetSegmenter(device=self.device)

    def inspect(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        render_visuals: bool = True,
        force_full_pipeline: bool = False,
        extract_morphology: bool = True,
        generate_certificate: bool = True
    ) -> UnifiedInspectionResult:
        """
        Executes end-to-end industrial inspection on a product image.

        Args:
            image_input: File path, PIL Image, or NumPy array.
            render_visuals: If True, renders annotated visual overlay with HUD banner.
            force_full_pipeline: If True, runs full diagnostic pipeline even if Tier 1 is nominal.
            extract_morphology: If True, extracts granular shape metrics (circularity, solidity, aspect ratio, zoning).
            generate_certificate: If True, generates a cryptographically hashed inspection certificate.

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
        tau = self.decision_engine.thresholds.anomaly_threshold
        delta = self.decision_engine.thresholds.anomaly_margin

        t1_response, anomaly_heatmap, _ = self.tier1_inferencer.predict(
            image_rgb,
            threshold=tau,
            review_margin=delta,
            mm_per_pixel=self.calibrator.mm_per_pixel
        )

        anomaly_score = t1_response.image_anomaly_score
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

            res = UnifiedInspectionResult(
                verdict=verdict,
                tier1_anomaly=t1_response,
                tier2_classification=None,
                tier3_detection=None,
                tier3_segmentation=None,
                physical_metrics={
                    "total_defect_area_mm2": 0.0,
                    "measurement_mode": "planar_surface" if self.domain == ProductDomain.STEEL_SHEET else "projected_sensor_plane",
                    "mm_per_pixel": self.calibrator.mm_per_pixel,
                    "num_defects": 0
                },
                annotated_image=annotated_img,
                total_latency_ms=round(total_elapsed, 2),
                early_exit_applied=True,
                product_domain=self.domain.value,
                morphological_features=[],
                certificate=None
            )
            if generate_certificate:
                from src.inspection.defect_analysis import InspectionReportGenerator
                cert = InspectionReportGenerator.generate_certificate(res, [])
                res.certificate = cert.to_dict()
            return res

        # =================================================================
        # DOMAIN-SPECIFIC DIAGNOSTIC ESCALATION
        # =================================================================
        t2_response = None
        t3_det_response = None
        t3_seg_response = None
        t3_boxes: List[DetectionBox] = []
        seg_mask: Optional[np.ndarray] = None
        defect_class: str = "unmodeled_anomaly"
        class_conf: float = 1.0
        class_entropy: float = 0.0
        pixel_defect_area_mm2: float = 0.0
        bbox_union_area_mm2: float = 0.0
        num_regions: int = 0

        if self.domain == ProductDomain.GLASS_BOTTLE:
            # -------------------------------------------------------------
            # Glassware Domain: PatchCore Heatmap Localization
            # -------------------------------------------------------------
            # Do not use steel-trained supervised models for glass bottles.
            # Localize directly using PatchCore's verified pixel anomaly map (0.9848 Pixel AUROC)
            if anomaly_heatmap is not None:
                # Binarize anomaly map at threshold tau
                h_resized = cv2.resize(anomaly_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                binary_mask = (h_resized >= tau).astype(np.uint8)
                seg_mask = binary_mask

                defect_px_count = int(np.count_nonzero(binary_mask))
                # For curved bottles, area is estimated projected sensor-plane area
                pixel_defect_area_mm2 = self.calibrator.estimate_projected_area_mm2(defect_px_count)

                # Connected components for defect counting
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
                num_regions = max(0, num_labels - 1)

                defect_class = "glass_anomaly" if defect_px_count > 0 else "nominal"
                class_conf = min(1.0, float(anomaly_score / (tau * 1.5))) if tau > 0 else 1.0

                components_list = []
                for label_id in range(1, num_labels):
                    area_px = int(stats[label_id, cv2.CC_STAT_AREA])
                    bx = int(stats[label_id, cv2.CC_STAT_LEFT])
                    by = int(stats[label_id, cv2.CC_STAT_TOP])
                    bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
                    bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                    cx, cy = float(centroids[label_id][0]), float(centroids[label_id][1])
                    comp_area_mm2 = self.calibrator.estimate_projected_area_mm2(area_px)

                    components_list.append(DefectComponent(
                        component_id=label_id,
                        area_pixels=area_px,
                        area_ratio=float(area_px / (img_w * img_h)),
                        centroid=(round(cx, 1), round(cy, 1)),
                        bounding_box_xywh=(bx, by, bw, bh),
                        physical_area_mm2=round(comp_area_mm2, 3)
                    ))

                t3_seg_response = SegmentationResponse(
                    image_width=img_w,
                    image_height=img_h,
                    is_defective=defect_px_count > 0,
                    total_defect_pixels=defect_px_count,
                    defect_area_ratio=round(float(defect_px_count / (img_w * img_h)), 6),
                    num_defect_components=num_regions,
                    inference_time_ms=t1_response.inference_time_ms,
                    physical_defect_area_mm2=round(pixel_defect_area_mm2, 3),
                    components=components_list
                )
            else:
                pixel_defect_area_mm2 = 0.0
                num_regions = 0

        else:
            # -------------------------------------------------------------
            # Steel Surface Domain (or Generic AOI):
            # -------------------------------------------------------------
            # Execute Tier 2 Classifier
            t2_response = self.tier2_inferencer.predict(image_rgb)
            defect_class = t2_response.predicted_class_name
            class_conf = t2_response.confidence
            class_entropy = t2_response.entropy

            # Execute Tier 3A YOLO Detector
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
                            t3_det_response.total_bbox_union_area_pixels,
                            is_planar_surface=True
                        )
                except Exception as e:
                    logger.warning(f"Tier 3 YOLO detection encountered exception: {e}")

            # Execute Tier 3B U-Net Segmenter
            t3_seg_response, seg_mask, _ = self.tier3_segmenter.predict(
                image_rgb,
                mm_per_pixel=self.calibrator.mm_per_pixel
            )
            pixel_defect_area_mm2 = (
                t3_seg_response.physical_defect_area_mm2
                if t3_seg_response.physical_defect_area_mm2 is not None
                else self.calibrator.pixel_area_to_mm2(t3_seg_response.total_defect_pixels, is_planar_surface=True)
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
            is_anomalous=t1_response.is_anomalous or (anomaly_score >= tau),
            defect_class=defect_class,
            class_confidence=class_conf,
            class_entropy=class_entropy,
            pixel_defect_area_mm2=pixel_defect_area_mm2,
            bbox_union_area_mm2=bbox_union_area_mm2,
            num_defect_regions=num_regions
        )

        total_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Physical Metrics Payload
        measurement_mode = (
            "planar_surface" if self.domain == ProductDomain.STEEL_SHEET else "projected_sensor_plane"
        )
        physical_metrics = {
            "total_defect_area_mm2": round(verdict.physical_defect_area_mm2, 3),
            "pixel_defect_area_mm2": round(pixel_defect_area_mm2, 3),
            "bbox_union_area_mm2": round(bbox_union_area_mm2, 3),
            "measurement_mode": measurement_mode,
            "is_planar_surface": self.domain == ProductDomain.STEEL_SHEET,
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

        # Extract Morphological Shape Descriptors
        morph_dict_list: List[Dict[str, Any]] = []
        morph_obj_list = []
        if extract_morphology and seg_mask is not None and np.count_nonzero(seg_mask) > 0:
            from src.inspection.defect_analysis import DefectMorphologyAnalyzer
            analyzer = DefectMorphologyAnalyzer()
            morph_obj_list = analyzer.analyze_mask(
                binary_mask=seg_mask,
                mm_per_pixel=self.calibrator.mm_per_pixel,
                domain=self.domain
            )
            morph_dict_list = [m.to_dict() for m in morph_obj_list]

        res = UnifiedInspectionResult(
            verdict=verdict,
            tier1_anomaly=t1_response,
            tier2_classification=t2_response,
            tier3_detection=t3_det_response,
            tier3_segmentation=t3_seg_response,
            physical_metrics=physical_metrics,
            annotated_image=annotated_img,
            total_latency_ms=round(total_elapsed, 2),
            early_exit_applied=False,
            product_domain=self.domain.value,
            morphological_features=morph_dict_list,
            certificate=None
        )

        if generate_certificate:
            from src.inspection.defect_analysis import InspectionReportGenerator
            cert = InspectionReportGenerator.generate_certificate(res, morph_obj_list)
            res.certificate = cert.to_dict()

        return res

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
