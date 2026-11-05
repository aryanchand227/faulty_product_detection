"""
Unit tests for Defect Morphology Analysis, Multi-Zone Overlap, Area Reconciliation,
and HMAC-SHA256 Cryptographic Authentication.
"""

import hmac
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
    DEFAULT_HMAC_SECRET,
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
    """Validates that a round circular contour is classified as GEOMETRIC_COMPACT_BLOB."""
    mask = create_synthetic_circle(radius=25, img_size=200)
    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=10)
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.10, domain=ProductDomain.GLASS_BOTTLE)

    assert len(features) == 1
    f = features[0]

    # Circularity for circle should be high (>= 0.75 on discrete grid)
    assert f.circularity >= 0.75
    assert f.solidity >= 0.90
    assert 1.0 <= f.aspect_ratio <= 1.4
    assert f.is_shape_reliable is True
    assert f.morphology_type == MorphologyType.GEOMETRIC_COMPACT_BLOB
    assert f.equivalent_diameter_mm > 0.0
    assert f.major_axis_length_mm > 0.0


def test_morphology_filamentary_crack_major_axis():
    """
    Validates that an elongated slit contour is classified as GEOMETRIC_ELONGATED_SLIT
    and that major_axis_length_mm reveals true crack length while equivalent_diameter conceals it.
    """
    mask = create_synthetic_crack(length=90, thickness=3, img_size=200)
    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=10)
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.10, domain=ProductDomain.GLASS_BOTTLE)

    assert len(features) == 1
    f = features[0]

    assert f.aspect_ratio >= 5.0
    assert f.circularity <= 0.35
    assert f.morphology_type == MorphologyType.GEOMETRIC_ELONGATED_SLIT
    assert f.morphology_risk_weight == MORPHOLOGY_RISK_MULTIPLIERS[MorphologyType.GEOMETRIC_ELONGATED_SLIT]

    # Major axis length (~9.0 mm at 0.10 mm/px) must accurately represent crack length
    assert 8.0 <= f.major_axis_length_mm <= 10.0
    # Equivalent diameter is much smaller (~1.8 mm), demonstrating why reporting major axis is critical
    assert f.equivalent_diameter_mm < 3.0
    assert f.major_axis_length_mm > f.equivalent_diameter_mm * 2.5


def test_morphology_small_defect_rasterization_bias_flag():
    """Validates that a tiny defect (<25 pixels) is flagged with is_shape_reliable=False."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # 5x5 square: polygon area = 16 pixels (between min_contour_area 10 and reliable 25)
    mask[40:45, 40:45] = 255

    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=10, min_reliable_area_px=25)
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.10, domain=ProductDomain.GLASS_BOTTLE)

    assert len(features) == 1
    f = features[0]
    assert 10 <= f.area_pixels < 25
    assert f.is_shape_reliable is False  # Discrete grid rasterization bias flag active


def test_morphology_diffuse_non_convex():
    """Validates that a non-convex concave defect exhibits low solidity and diffuse classification."""
    mask = create_synthetic_c_shape(img_size=200)
    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=10)
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.10, domain=ProductDomain.GLASS_BOTTLE)

    assert len(features) == 1
    f = features[0]
    assert f.solidity <= 0.65
    assert f.morphology_type in (MorphologyType.GEOMETRIC_DIFFUSE_CLUSTER, MorphologyType.GEOMETRIC_IRREGULAR)


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


def test_multi_zone_overlap_highest_risk_touched():
    """
    CRITICAL AUDIT TEST: Verifies that a defect spanning the boundary between
    SEALING_RIM (y < 32) and NECK_SHOULDER (y >= 32) whose centroid falls in NECK_SHOULDER
    is assigned to SEALING_RIM under the highest-risk touched rule.
    """
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Defect extends from y=20 (in sealing rim) to y=55 (in neck/shoulder)
    # Total height = 35 px. Sealing rim portion = 12 px (y=20 to 31). Neck portion = 23 px (y=32 to 55).
    # Centroid is at y ~ 37.5 (in NECK_SHOULDER!)
    mask[20:55, 95:105] = 255

    analyzer = DefectMorphologyAnalyzer(min_contour_area_px=10)
    features = analyzer.analyze_mask(mask, domain=ProductDomain.GLASS_BOTTLE)

    assert len(features) == 1
    f = features[0]
    # Centroid is in neck/shoulder
    assert f.centroid_xy[1] > 32.0

    # BUT primary zone MUST be SEALING_RIM because it touches the sealing rim!
    assert f.spatial_zone == SpatialZone.SEALING_RIM
    assert f.zone_risk_weight == 2.5
    # Overlap dictionary records both zones
    assert "SEALING_RIM" in f.zone_overlaps
    assert "NECK_SHOULDER" in f.zone_overlaps
    assert f.zone_overlaps["SEALING_RIM"] > 0.0
    assert f.zone_overlaps["NECK_SHOULDER"] > 0.0


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


def test_area_reconciliation_consistency_invariant():
    """
    AUDIT TEST: Verifies area reconciliation invariant:
    total_defect_area_mm2 == component_defect_area_mm2 + filtered_noise_area_mm2.
    """
    verdict = InspectionVerdict(
        decision=InspectionDecision.FAIL,
        severity=SeverityLevel.CRITICAL,
        routing=RoutingAction.AUTOMATIC_REJECT_BIN,
        primary_defect_type="broken_large",
        confidence=0.98,
        physical_defect_area_mm2=50.00,  # Total mask area
        anomaly_score=26.50,
        is_anomalous=True,
        engineering_rationale="Defect area exceeds critical limit."
    )
    t1_response = AnomalyInspectionResponse(
        decision="FAIL",
        image_anomaly_score=26.50,
        threshold=15.50,
        is_anomalous=True,
        defect_pixel_count=5000,
        defect_area_ratio=0.05,
        physical_defect_area_mm2=50.00,
        inference_time_ms=25.0,
        num_anomaly_regions=1,
        regions=[]
    )
    res = UnifiedInspectionResult(
        verdict=verdict,
        tier1_anomaly=t1_response,
        product_domain="GLASS_BOTTLE",
        physical_metrics={"measurement_mode": "projected_sensor_plane", "scale_mm_per_pixel": 0.10},
        total_latency_ms=28.4,
        early_exit_applied=False
    )

    # Synthetic component with 40.0 mm2 area (leaving 10.0 mm2 as filtered noise)
    mask = create_synthetic_circle(radius=20, img_size=100)
    analyzer = DefectMorphologyAnalyzer()
    features = analyzer.analyze_mask(mask, mm_per_pixel=0.10, domain=ProductDomain.GLASS_BOTTLE)

    cert = InspectionReportGenerator.generate_certificate(
        inspection_result=res,
        morphological_features=features,
        config_version="CFG-v1.0.5"
    )

    # Check that area reconciliation is strictly verified
    assert cert.is_area_reconciled is True
    reconciled_sum = cert.component_defect_area_mm2 + cert.filtered_noise_area_mm2
    assert abs(cert.total_defect_area_mm2 - reconciled_sum) < 0.002
    assert cert.config_version == "CFG-v1.0.5"


def test_hmac_sha256_authentication_and_tamper_detection():
    """
    CRITICAL AUDIT TEST: Verifies HMAC-SHA256 authentication.
    Confirms that any post-generation modification of certificate fields
    is immediately detected as unauthorized tampering.
    """
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
        source_image_bytes=b"authentic_source_image_payload"
    )

    cert_dict = cert.to_dict()

    # 1. Valid certificate verification
    assert InspectionReportGenerator.verify_certificate(cert_dict) is True

    # 2. Tampered certificate verification: modify verdict from FAIL to PASS
    tampered_dict_1 = dict(cert_dict)
    tampered_dict_1["decision"] = "PASS"
    assert InspectionReportGenerator.verify_certificate(tampered_dict_1) is False

    # 3. Tampered certificate verification: modify defect area from 45.2 to 0.0
    tampered_dict_2 = dict(cert_dict)
    tampered_dict_2["total_defect_area_mm2"] = 0.0
    assert InspectionReportGenerator.verify_certificate(tampered_dict_2) is False

    # 4. Tampered certificate verification: wrong secret key
    assert InspectionReportGenerator.verify_certificate(cert_dict, hmac_secret=b"fraudulent-key") is False
