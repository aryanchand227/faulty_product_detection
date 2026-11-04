"""
Unit and Integration Tests for Phase 8: Unified Multi-Tier Decision Pipeline
and Camera Calibration Integration.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.classifier_inference import ClassifierInferencer, DefectClassificationResponse
from src.inference.detector_inference import DetectionBox
from src.inspection.camera_calibration import CameraCalibrationConfig, CameraCalibrator
from src.inspection.decision_engine import (
    DecisionEngine,
    InspectionDecision,
    InspectionVerdict,
    QualityThresholds,
    RoutingAction,
    SeverityLevel,
)
from src.inspection.unified_pipeline import UnifiedInspectionResult, UnifiedQualityInspector
from src.inspection.visualizer import InspectionVisualizer


# -------------------------------------------------------------------------
# 1. Camera Calibration Unit Tests
# -------------------------------------------------------------------------
def test_camera_calibrator_linear_and_area_scaling():
    """Validates 1D metric length and 2D area conversions."""
    calibrator = CameraCalibrator(mm_per_pixel=0.05)  # 1 px = 0.05 mm, 20 px = 1 mm

    assert calibrator.mm_per_pixel == 0.05
    assert pytest.approx(calibrator.mm2_per_pixel2, abs=1e-6) == 0.0025

    # 1D linear conversion
    px_len = 100.0
    mm_len = calibrator.pixels_to_mm(px_len)
    assert pytest.approx(mm_len, abs=1e-4) == 5.0
    assert pytest.approx(calibrator.mm_to_pixels(mm_len), abs=1e-4) == px_len

    # 2D area conversion
    px_area = 400.0  # 20 x 20 pixels
    mm2_area = calibrator.pixel_area_to_mm2(px_area)
    assert pytest.approx(mm2_area, abs=1e-4) == 1.0  # (20 * 0.05) * (20 * 0.05) = 1.0 mm^2
    assert pytest.approx(calibrator.mm2_to_pixel_area(mm2_area), abs=1e-4) == px_area

    # Box dimensioning
    box_xywh = (10.0, 20.0, 50.0, 100.0)
    dim_dict = calibrator.compute_box_dimensions_mm(box_xywh)
    assert dim_dict["width_mm"] == 2.5
    assert dim_dict["height_mm"] == 5.0
    assert dim_dict["area_mm2"] == 12.5
    assert dim_dict["aspect_ratio"] == 0.5


def test_camera_calibrator_fov_scaling():
    """Validates scale computation from physical field of view."""
    calibrator = CameraCalibrator()
    scale = calibrator.calibrate_from_fov(fov_width_mm=192.0, image_width_px=1920)
    assert pytest.approx(scale, abs=1e-5) == 0.10
    assert calibrator.mm_per_pixel == 0.10


def test_camera_calibrator_synthetic_checkerboard():
    """Validates optical sub-pixel corner detection and scale calculation on synthetic grid."""
    pattern_size = (7, 5)  # 7 columns, 5 rows internal corners
    square_size_px = 30
    square_size_mm = 6.0  # 6 mm per 30 px => 0.20 mm/pixel

    cols, rows = pattern_size
    img_w = (cols + 2) * square_size_px
    img_h = (rows + 2) * square_size_px
    board = np.ones((img_h, img_w), dtype=np.uint8) * 255

    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y1 = (r + 1) * square_size_px
                y2 = y1 + square_size_px
                x1 = (c + 1) * square_size_px
                x2 = x1 + square_size_px
                board[y1:y2, x1:x2] = 0

    calibrator = CameraCalibrator()
    success, scale = calibrator.calibrate_from_checkerboard(
        board,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm
    )

    assert success is True
    assert pytest.approx(scale, rel=0.05) == 0.20


# -------------------------------------------------------------------------
# 2. Defect Classifier Inferencer Tests
# -------------------------------------------------------------------------
def test_classifier_inferencer_forward():
    """Validates Tier 2 classifier forward pass and uncertainty output."""
    inferencer = ClassifierInferencer(
        class_names=["good", "crack", "dent"],
        device="cpu",
        target_size=(64, 64)
    )

    dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    response = inferencer.predict(dummy_img)

    assert isinstance(response, DefectClassificationResponse)
    assert response.predicted_class_name in ["good", "crack", "dent"]
    assert 0.0 <= response.confidence <= 1.0
    assert response.entropy >= 0.0
    assert sum(response.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert response.inference_time_ms >= 0.0


# -------------------------------------------------------------------------
# 3. Decision Engine Routing Rules Tests
# -------------------------------------------------------------------------
def test_decision_engine_nominal_pass():
    """Tests early PASS verdict for nominal item."""
    engine = DecisionEngine(QualityThresholds(anomaly_threshold=15.0, anomaly_margin=0.15))

    verdict = engine.evaluate(
        anomaly_score=8.5,  # 8.5 < 15.0 * 0.85 = 12.75
        is_anomalous=False,
        defect_class="good",
        class_confidence=0.95,
        pixel_defect_area_mm2=0.0,
        num_defect_regions=0
    )

    assert verdict.decision == InspectionDecision.PASS
    assert verdict.severity == SeverityLevel.NONE
    assert verdict.routing == RoutingAction.LINE_CONTINUE
    assert len(verdict.rejection_reasons) == 0
    assert "PASSED" in verdict.engineering_rationale


def test_decision_engine_critical_defect_fail():
    """Tests FAIL verdict and pneumatic diverter routing for severe defects."""
    engine = DecisionEngine(QualityThresholds(
        anomaly_threshold=15.0,
        critical_area_mm2=4.0,
        critical_classes=["crazing", "broken_large"]
    ))

    # Condition 1: Critical defect category
    verdict1 = engine.evaluate(
        anomaly_score=14.0,
        is_anomalous=False,
        defect_class="crazing",
        class_confidence=0.88,
        pixel_defect_area_mm2=1.2,
        num_defect_regions=1
    )
    assert verdict1.decision == InspectionDecision.FAIL
    assert verdict1.severity == SeverityLevel.CRITICAL
    assert verdict1.routing == RoutingAction.AUTOMATIC_REJECT_BIN

    # Condition 2: Excessive damage area
    verdict2 = engine.evaluate(
        anomaly_score=12.0,
        is_anomalous=False,
        defect_class="scratch",
        class_confidence=0.80,
        pixel_defect_area_mm2=6.8,  # > 4.0 mm^2
        num_defect_regions=2
    )
    assert verdict2.decision == InspectionDecision.FAIL
    assert verdict2.severity == SeverityLevel.CRITICAL


def test_decision_engine_borderline_review():
    """Tests REVIEW verdict for borderline scores or moderate defects."""
    engine = DecisionEngine(QualityThresholds(
        anomaly_threshold=15.0,
        anomaly_margin=0.10,  # [13.5, 16.5]
        max_acceptable_area_mm2=0.50
    ))

    verdict = engine.evaluate(
        anomaly_score=14.2,  # Inside borderline band [13.5, 16.5]
        is_anomalous=False,
        defect_class="scratch",
        class_confidence=0.72,
        pixel_defect_area_mm2=0.80,  # Moderate area
        num_defect_regions=1
    )

    assert verdict.decision == InspectionDecision.REVIEW
    assert verdict.routing == RoutingAction.MANUAL_INSPECTION_BAY
    assert "OPERATOR REVIEW" in verdict.engineering_rationale


# -------------------------------------------------------------------------
# 4. Multi-Tier Visualizer & HUD Overlay Tests
# -------------------------------------------------------------------------
def test_visualizer_hud_overlay_dimensions():
    """Validates annotated overlay and HUD header generation."""
    h, w = 120, 160
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    verdict = InspectionVerdict(
        decision=InspectionDecision.FAIL,
        severity=SeverityLevel.CRITICAL,
        routing=RoutingAction.AUTOMATIC_REJECT_BIN,
        primary_defect_type="broken_rim",
        confidence=0.91,
        physical_defect_area_mm2=8.45,
        anomaly_score=21.4,
        is_anomalous=True
    )

    boxes = [
        DetectionBox(
            class_id=1,
            class_name="broken_rim",
            confidence=0.91,
            box_xyxy=(20.0, 30.0, 70.0, 80.0),
            box_normalized=(0.28, 0.45, 0.31, 0.41),
            bbox_area_pixels=2500.0,
            bbox_area_ratio=0.13
        )
    ]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:80, 20:70] = 1

    hud_result = InspectionVisualizer.render_hud_overlay(
        image_rgb=img,
        verdict=verdict,
        anomaly_heatmap=None,
        boxes=boxes,
        segmentation_mask=mask,
        latency_ms=34.2
    )

    # Header adds 56 pixels to height
    assert hud_result.shape == (h + 56, w, 3)
    assert hud_result.dtype == np.uint8


def test_visualizer_quad_panel_dashboard():
    """Validates 4-pane comparative dashboard creation."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    verdict = InspectionVerdict(
        decision=InspectionDecision.PASS,
        severity=SeverityLevel.NONE,
        routing=RoutingAction.LINE_CONTINUE,
        primary_defect_type="nominal",
        confidence=0.99,
        physical_defect_area_mm2=0.0,
        anomaly_score=6.2,
        is_anomalous=False
    )

    quad = InspectionVisualizer.create_quad_panel_dashboard(
        image_rgb=img,
        verdict=verdict,
        target_size=(150, 150)
    )

    assert quad.shape == (300, 300, 3)


# -------------------------------------------------------------------------
# 5. Unified Quality Inspector Pipeline End-to-End Tests
# -------------------------------------------------------------------------
class MockPatchCoreInferencer:
    """Mock Tier 1 inferencer for deterministic integration testing."""
    def __init__(self, score: float = 6.0):
        self.score = score

    def predict(self, img, mm_per_pixel=0.10):
        from src.inference.anomaly_inference import AnomalyInspectionResponse
        resp = AnomalyInspectionResponse(
            decision="PASS" if self.score < 15.0 else "FAIL",
            image_anomaly_score=self.score,
            threshold=15.0,
            is_anomalous=self.score >= 15.0,
            defect_pixel_count=0 if self.score < 15.0 else 50,
            defect_area_ratio=0.0 if self.score < 15.0 else 0.05,
            physical_defect_area_mm2=0.0 if self.score < 15.0 else 0.5,
            inference_time_ms=10.0,
            num_anomaly_regions=0 if self.score < 15.0 else 1,
            regions=[]
        )
        return resp, np.zeros((64, 64), dtype=np.float32), np.zeros((64, 64, 3), dtype=np.uint8)


def test_unified_quality_inspector_early_exit():
    """Confirms that nominal items trigger early exit without executing Tiers 2 & 3."""
    inspector = UnifiedQualityInspector(
        tier1_inferencer=MockPatchCoreInferencer(score=5.0),
        early_exit=True,
        thresholds=QualityThresholds(anomaly_threshold=15.0, anomaly_margin=0.15),
        device="cpu"
    )

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = inspector.inspect(dummy_img, render_visuals=True)

    assert isinstance(result, UnifiedInspectionResult)
    assert result.early_exit_applied is True
    assert result.verdict.decision == InspectionDecision.PASS
    assert result.tier2_classification is None
    assert result.tier3_detection is None
    assert result.tier3_segmentation is None
    assert result.annotated_image is not None
    assert result.total_latency_ms >= 0.0


def test_unified_quality_inspector_full_pipeline_on_anomaly():
    """Confirms that anomalous items trigger full execution across all three tiers."""
    inspector = UnifiedQualityInspector(
        tier1_inferencer=MockPatchCoreInferencer(score=22.5),  # High anomaly score
        early_exit=True,
        thresholds=QualityThresholds(anomaly_threshold=15.0, anomaly_margin=0.15),
        device="cpu"
    )

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = inspector.inspect(dummy_img, render_visuals=True)

    assert result.early_exit_applied is False
    assert result.verdict.decision == InspectionDecision.FAIL
    assert result.tier2_classification is not None
    assert result.tier3_segmentation is not None
    assert "total_defect_area_mm2" in result.physical_metrics
    assert result.annotated_image is not None
    assert result.total_latency_ms > 0.0
