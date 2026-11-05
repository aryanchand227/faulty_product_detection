"""
API Integration & Functional Unit Tests for FastAPI Inspection Service.
Includes tests for authorization guards, audit logging, input validation bounds,
and threadpool offloading.
"""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.inspection_service import InspectionService
from src.inference.anomaly_inference import AnomalyInspectionResponse
from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import DecisionEngine, QualityThresholds
from src.inspection.unified_pipeline import ProductDomain, UnifiedQualityInspector

client = TestClient(app)

ADMIN_HEADERS = {"X-API-Key": "aoi-admin-secret-key-2026"}


def make_test_image_bytes(is_defective: bool = False, width: int = 100, height: int = 100) -> bytes:
    """Helper creating in-memory PNG image bytes."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 180
    if is_defective:
        # Draw a synthetic crack
        cv2.line(img, (20, height // 2), (width - 20, height // 2), (20, 20, 20), 4)
    success, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


class MockPatchCoreInferencer:
    """Mock Tier 1 inferencer for fast, offline API testing."""
    def __init__(self, score: float = 5.0):
        self.score = score

    def predict(self, img, threshold=15.0, review_margin=0.15, mm_per_pixel=0.10, **kwargs):
        is_anom = self.score >= threshold
        decision = "FAIL" if is_anom else "PASS"
        h, w = img.shape[:2]

        hmap = np.ones((h, w), dtype=np.float32) * self.score
        if is_anom:
            # Add concentrated high-anomaly hotspot
            hmap[20:40, 20:40] = self.score + 5.0

        resp = AnomalyInspectionResponse(
            decision=decision,
            image_anomaly_score=self.score,
            threshold=threshold,
            is_anomalous=is_anom,
            defect_pixel_count=400 if is_anom else 0,
            defect_area_ratio=0.04 if is_anom else 0.0,
            physical_defect_area_mm2=4.0 if is_anom else 0.0,
            inference_time_ms=12.5,
            num_anomaly_regions=1 if is_anom else 0,
            regions=[]
        )
        return resp, hmap, (h, w)


def test_health_check_endpoint():
    """Verify that GET /api/v1/health returns 200 and healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "python_version" in data
    assert "Industrial Damage Detection API is running" in data["message"]


def test_get_and_update_thresholds_authorized():
    """Verify GET and PUT on /api/v1/inspect/thresholds with valid authorization."""
    # GET
    get_res = client.get("/api/v1/inspect/thresholds")
    assert get_res.status_code == 200
    t_data = get_res.json()
    assert "anomaly_threshold" in t_data
    assert "config_version" in t_data

    # PUT with valid admin headers
    update_payload = {
        "anomaly_threshold": 16.20,
        "critical_area_mm2": 30.0,
        "anomaly_margin": 0.12
    }
    put_res = client.put("/api/v1/inspect/thresholds", json=update_payload, headers=ADMIN_HEADERS)
    assert put_res.status_code == 200
    updated_data = put_res.json()
    assert updated_data["anomaly_threshold"] == 16.20
    assert updated_data["critical_area_mm2"] == 30.0
    assert updated_data["anomaly_margin"] == 0.12
    assert "config_version" in updated_data


def test_thresholds_authorization_rejection():
    """Verify that unauthorized PUT requests are rejected with 401 Unauthorized."""
    bad_headers = {"X-API-Key": "unauthorized-fraudulent-token"}
    put_res = client.put(
        "/api/v1/inspect/thresholds",
        json={"anomaly_threshold": 20.0},
        headers=bad_headers
    )
    assert put_res.status_code == 401
    assert "unauthorized" in put_res.json()["detail"].lower()


def test_threshold_input_bounds_validation():
    """Verify that invalid threshold values violating engineering limits are rejected with 422."""
    # Anomaly threshold exceeds 50.0 limit
    put_res = client.put(
        "/api/v1/inspect/thresholds",
        json={"anomaly_threshold": 150.0},
        headers=ADMIN_HEADERS
    )
    assert put_res.status_code == 422

    # Negative critical area
    put_res_neg = client.put(
        "/api/v1/inspect/thresholds",
        json={"critical_area_mm2": -10.0},
        headers=ADMIN_HEADERS
    )
    assert put_res_neg.status_code == 422


def test_get_and_update_calibration():
    """Verify GET and PUT on /api/v1/inspect/calibration with audit trail recording."""
    # GET
    get_res = client.get("/api/v1/inspect/calibration")
    assert get_res.status_code == 200
    c_data = get_res.json()
    assert "mm_per_pixel" in c_data

    # PUT
    update_payload = {"mm_per_pixel": 0.22, "units": "mm"}
    put_res = client.put("/api/v1/inspect/calibration", json=update_payload, headers=ADMIN_HEADERS)
    assert put_res.status_code == 200
    assert put_res.json()["mm_per_pixel"] == 0.22


def test_audit_log_endpoint():
    """Verify that GET /api/v1/inspect/audit-log tracks configuration updates."""
    # Make a configuration change
    client.put("/api/v1/inspect/thresholds", json={"anomaly_threshold": 17.00}, headers=ADMIN_HEADERS)

    response = client.get("/api/v1/inspect/audit-log")
    assert response.status_code == 200
    log_entries = response.json()
    assert len(log_entries) >= 1
    latest_entry = log_entries[-1]
    assert "timestamp_utc" in latest_entry
    assert "action" in latest_entry
    assert "config_version" in latest_entry


def test_system_info_endpoint():
    """Verify GET /api/v1/inspect/system-info returns hardware device, config version, and capabilities."""
    response = client.get("/api/v1/inspect/system-info")
    assert response.status_code == 200
    data = response.json()
    assert data["service_name"] == "Industrial Quality Inspection API"
    assert "config_version" in data
    assert "GLASS_BOTTLE" in data["supported_domains"]
    assert "STEEL_SHEET" in data["supported_domains"]
    assert data["active_audit_entries_count"] >= 0


def test_inspect_image_nominal_early_exit():
    """Validates that a low-anomaly image triggers early exit and PASS verdict over HTTP API."""
    mock_t1 = MockPatchCoreInferencer(score=6.0)  # Nominal score < threshold
    mock_inspector = UnifiedQualityInspector(
        domain=ProductDomain.GLASS_BOTTLE,
        tier1_inferencer=mock_t1,
        early_exit=True,
        thresholds=QualityThresholds(anomaly_threshold=15.0, anomaly_margin=0.15),
        device="cpu"
    )
    mock_service = InspectionService(inspector=mock_inspector)
    InspectionService.set_instance(mock_service)

    img_bytes = make_test_image_bytes(is_defective=False)
    response = client.post(
        "/api/v1/inspect/image",
        files={"file": ("nominal_bottle.png", img_bytes, "image/png")},
        data={"domain": "GLASS_BOTTLE", "render_visuals": "true", "force_full_pipeline": "false"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["early_exit_applied"] is True
    assert data["verdict"]["decision"] == "PASS"
    assert data["verdict"]["routing"] == "LINE_CONTINUE"
    assert data["verdict"]["severity"] == "NONE"
    assert data["certificate"] is not None
    assert "hmac_sha256_signature" in data["certificate"]
    assert len(data["certificate"]["hmac_sha256_signature"]) == 64
    assert data["annotated_image_base64"] is not None
    assert data["annotated_image_base64"].startswith("data:image/png;base64,")


def test_inspect_image_defective_with_morphology():
    """Validates that an anomalous image triggers full inspection, morphology analysis, and FAIL verdict."""
    mock_t1 = MockPatchCoreInferencer(score=28.0)  # Defect score >> threshold
    mock_inspector = UnifiedQualityInspector(
        domain=ProductDomain.GLASS_BOTTLE,
        tier1_inferencer=mock_t1,
        early_exit=True,
        thresholds=QualityThresholds(anomaly_threshold=15.0, anomaly_margin=0.15, critical_area_mm2=2.0),
        device="cpu"
    )
    mock_service = InspectionService(inspector=mock_inspector)
    InspectionService.set_instance(mock_service)

    img_bytes = make_test_image_bytes(is_defective=True)
    response = client.post(
        "/api/v1/inspect/image",
        files={"file": ("broken_bottle.png", img_bytes, "image/png")},
        data={
            "domain": "GLASS_BOTTLE",
            "render_visuals": "true",
            "force_full_pipeline": "true",
            "extract_morphology": "true",
            "generate_certificate": "true"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["early_exit_applied"] is False
    assert data["verdict"]["decision"] == "FAIL"
    assert data["verdict"]["routing"] == "AUTOMATIC_REJECT_BIN"
    assert data["verdict"]["severity"] == "CRITICAL"
    assert len(data["morphological_features"]) >= 1

    first_feat = data["morphological_features"][0]
    assert "circularity" in first_feat
    assert "solidity" in first_feat
    assert "aspect_ratio" in first_feat
    assert "major_axis_length_mm" in first_feat
    assert "is_shape_reliable" in first_feat
    assert "spatial_zone" in first_feat
    assert "heuristic_severity_score" in first_feat
    assert data["certificate"]["is_area_reconciled"] is True


def test_inspect_image_dimension_limits_rejected():
    """Verify that images with dimensions exceeding 4096px or below 16px are rejected."""
    # Under 16x16
    tiny_bytes = make_test_image_bytes(is_defective=False, width=8, height=8)
    res_tiny = client.post(
        "/api/v1/inspect/image",
        files={"file": ("tiny.png", tiny_bytes, "image/png")}
    )
    assert res_tiny.status_code == 422
    assert "minimum inspection resolution" in res_tiny.json()["detail"]


def test_inspect_image_invalid_content_type():
    """Verify that uploading a text or invalid file returns 400 Bad Request."""
    response = client.post(
        "/api/v1/inspect/image",
        files={"file": ("notes.txt", b"just a plain text file", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported image media type" in response.json()["detail"]


def test_inspect_image_empty_file():
    """Verify that uploading a 0-byte file returns 400 Bad Request."""
    response = client.post(
        "/api/v1/inspect/image",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]
