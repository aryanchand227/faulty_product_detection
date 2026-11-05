"""
Unit tests for Defect Morphology Analysis and Structured Inspection Reporting.
"""

import hashlib
import json
import math
import numpy as np
import pytest

from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import (
    InspectionDecision,
    InspectionVerdict,
    QualityThresholds,
    RoutingAction,
    SeverityLevel,
)
from src.inspection.defect_analysis import (
    DefectMorphologyAnalyzer,
    InspectionCertificate,
    InspectionReportGenerator,
    MorphologicalFeatures,
    MorphologyType,
    SpatialZone,
    ZONE_RISK_MULTIPLIERS,
    MORPHOLOGY_RISK_MULTIPLIERS,
)
from src.inspection.unified_pipeline import ProductDomain, UnifiedInspectionResult
from src.inference.anomaly_inference import AnomalyInspectionResponse


def create_synthetic_circle(radius: int = 25, img_size: int = 200) -> np.ndarray:
    """Creates a clean synthetic circular defect mask."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)
    import cv2
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def create_synthetic_crack(length: int = 80, thickness: int = 3, img_size: int = 200) -> np.ndarray:
    """Creates an elongated, high-aspect-ratio crack defect mask."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    import cv2
    pt1 = (img_size // 2 - length // 2, img_size // 2)
    pt2 = (img_size // 2 + length // 2, img_size // 2)
    cv2.line(mask, pt1, pt2, 255, thickness)
    return mask


def create_synthetic_c_shape(img_size: int = 200) -> np.ndarray:
    """Creates a non-convex C-shaped mask with low solidity."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    import cv2
    cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
    # Cut out the interior to make a C-shape
    cv2.rectangle(mask, (80, 70), (160, 130), 0, -1)
    return mask


def test_morphology_circular_pit():
    """Validates that a round circular contour is classified as COMPACT_PIT with high circularity and solidity."""
    mask = create_synthetic_circle(radius=25, img_size=200)
    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=10)
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.10, domain=ProductDomain.GLASS_BOTTLE)

    assert len(features) == 1
    f = features[0]

    # Circularity for circle should be high (>= 0.80 on discrete grid)
    assert f.circularity >= 0.75
    # Solidity should be ~1.0 (convex)
    assert f.solidity >= 0.90
    # Aspect ratio should be ~1.0
    assert 1.0 <= f.aspect_ratio <= 1.4
    assert f.morphology_type == MorphologyType.COMPACT_PIT_OR_INCLUSION
    assert f.equivalent_diameter_mm > 0.0


def test_morphology_filamentary_crack():
    """Validates that an elongated slit contour is classified as FILAMENTARY_CRACK with high aspect ratio."""
    mask = create_synthetic_crack(length=90, thickness=3, img_size=200)
    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=10)
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.10, domain=ProductDomain.GLASS_BOTTLE)

    assert len(features) == 1
    f = features[0]

    # High aspect ratio
    assert f.aspect_ratio >= 5.0
    # Low circularity
    assert f.circularity <= 0.35
    assert f.morphology_type == MorphologyType.FILAMENTARY_CRACK
    assert f.morphology_risk_weight == MORPHOLOGY_RISK_MULTIPLIERS[MorphologyType.FILAMENTARY_CRACK]


def test_morphology_diffuse_non_convex():
    """Validates that a non-convex concave defect exhibits low solidity and diffuse classification."""
    mask = create_synthetic_c_shape(img_size=200)
    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=10)
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.10, domain=ProductDomain.GLASS_BOTTLE)

    assert len(features) == 1
    f = features[0]
    # Low solidity because convex hull covers the hollow gap
    assert f.solidity <= 0.65
    assert f.morphology_type in (MorphologyType.DIFFUSE_CONTAMINATION, MorphologyType.IRREGULAR_BLEMISH)


def test_spatial_zoning_glass_bottle():
    """Verifies vertical functional zoning on glassware."""
    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=5)

    # 1. Defect at top rim (y=10 on 200x200 -> norm_y = 0.05)
    mask_rim = np.zeros((200, 200), dtype=np.uint8)
    mask_rim[5:15, 95:105] = 255
    f_rim = analyzer.analyze_mask(mask_rim, domain=ProductDomain.GLASS_BOTTLE)[0]
    assert f_rim.spatial_zone == SpatialZone.SEALING_RIM
    assert f_rim.zone_risk_weight == 2.5

    # 2. Defect at bottle body (y=100 on 200x200 -> norm_y = 0.50)
    mask_body = np.zeros((200, 200), dtype=np.uint8)
    mask_body[95:105, 95:105] = 255
    f_body = analyzer.analyze_mask(mask_body, domain=ProductDomain.GLASS_BOTTLE)[0]
    assert f_body.spatial_zone == SpatialZone.BODY_SIDEWALL
    assert f_body.zone_risk_weight == 1.0

    # 3. Defect at bottle base/heel (y=190 on 200x200 -> norm_y = 0.95)
    mask_base = np.zeros((200, 200), dtype=np.uint8)
    mask_base[185:195, 95:105] = 255
    f_base = analyzer.analyze_mask(mask_base, domain=ProductDomain.GLASS_BOTTLE)[0]
    assert f_base.spatial_zone == SpatialZone.BASE_HEEL
    assert f_base.zone_risk_weight == 2.0


def test_spatial_zoning_steel_sheet():
    """Verifies lateral functional zoning on rolling steel strips."""
    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=5)

    # Edge strip defect (x=8 on 200x200 -> norm_x = 0.04)
    mask_edge = np.zeros((200, 200), dtype=np.uint8)
    mask_edge[95:105, 5:15] = 255
    f_edge = analyzer.analyze_mask(mask_edge, domain=ProductDomain.STEEL_SHEET)[0]
    assert f_edge.spatial_zone == SpatialZone.EDGE_SHEAR_ZONE
    assert f_edge.zone_risk_weight == 2.0

    # Center strip defect (x=100 on 200x200 -> norm_x = 0.50)
    mask_center = np.zeros((200, 200), dtype=np.uint8)
    mask_center[95:105, 95:105] = 255
    f_center = analyzer.analyze_mask(mask_center, domain=ProductDomain.STEEL_SHEET)[0]
    assert f_center.spatial_zone == SpatialZone.MAIN_SURFACE
    assert f_center.zone_risk_weight == 1.0


def test_inspection_certificate_generation_and_cryptographic_hash():
    """Verifies that structured inspection certificates are generated with valid SHA-256 integrity signatures."""
    verdict = InspectionVerdict(
        decision=InspectionDecision.FAIL,
        severity=SeverityLevel.CRITICAL,
        routing=RoutingAction.AUTOMATIC_REJECT_BIN,
        primary_defect_type="broken_large",
        confidence=0.98,
        physical_defect_area_mm2=45.2,
        anomaly_score=26.50,
        is_anomalous=True,
        engineering_rationale="Severe structural fracture exceeding critical tolerance limit."
    )
    t1_response = AnomalyInspectionResponse(
        decision="FAIL",
        image_anomaly_score=26.50,
        threshold=15.50,
        is_anomalous=True,
        defect_pixel_count=150,
        defect_area_ratio=0.015,
        physical_defect_area_mm2=45.2,
        inference_time_ms=25.0,
        num_anomaly_regions=1,
        regions=[]
    )
    res = UnifiedInspectionResult(
        verdict=verdict,
        tier1_anomaly=t1_response,
        product_domain="GLASS_BOTTLE",
        physical_metrics={"measurement_mode": "projected_sensor_plane", "scale_mm_per_pixel": 0.15},
        total_latency_ms=28.4,
        early_exit_applied=False
    )

    mask = create_synthetic_crack(length=60, thickness=4, img_size=100)
    analyzer = DefectMorphologyAnalyzer()
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.15, domain=ProductDomain.GLASS_BOTTLE)

    cert = InspectionReportGenerator.generate_certificate(
        inspection_result=res,
        morphological_features=features,
        source_image_bytes=b"synthetic_raw_image_bytes"
    )

    assert cert.certificate_id.startswith("CERT-")
    assert cert.decision == "FAIL"
    assert cert.severity == "CRITICAL"
    assert cert.routing_action == "AUTOMATIC_REJECT_BIN"
    assert cert.dominant_morphology == MorphologyType.FILAMENTARY_CRACK.value
    assert len(cert.sha256_audit_signature) == 64  # Valid 256-bit hex string

    # Verify JSON serialization and markdown rendering
    cert_json = cert.to_json()
    assert "sha256_audit_signature" in cert_json
    cert_md = cert.to_markdown()
    assert "INDUSTRIAL QUALITY INSPECTION CERTIFICATE" in cert_md
    assert "AUTOMATIC_REJECT_BIN" in cert_md
