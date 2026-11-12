"""
Integration and End-to-End Tests for Historical Inspection Queries,
Cryptographic Certificate Verification, Analytics Aggregation, Soft Archival,
RBAC Authorization Roles, and Bounded Batch Concurrency.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.database.connection import Base, get_db
from backend.app.database import crud
from tests.test_database import _sample_inspection_payload


@pytest.fixture(scope="function")
def api_client_with_db():
    """Provides a TestClient wired to an isolated in-memory SQLite database."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client, TestingSessionLocal

    app.dependency_overrides.clear()


def _create_synthetic_image_bytes(color=(200, 200, 200), width=64, height=64) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_api_health_check_with_db(api_client_with_db):
    client, _ = api_client_with_db
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True
    assert "version" in data


def test_api_list_inspections_empty_and_populated(api_client_with_db):
    client, Session = api_client_with_db
    # 1. Empty check
    res_empty = client.get("/api/v1/inspections")
    assert res_empty.status_code == 200
    data = res_empty.json()
    assert data["total"] == 0
    assert len(data["items"]) == 0

    # 2. Populate records
    with Session() as s:
        crud.create_inspection_record(s, _sample_inspection_payload("CERT-API-01", "PASS", "NONE", 11.0))
        crud.create_inspection_record(s, _sample_inspection_payload("CERT-API-02", "FAIL", "CRITICAL", 27.5))

    res_pop = client.get("/api/v1/inspections?decision=FAIL")
    assert res_pop.status_code == 200
    data_pop = res_pop.json()
    assert data_pop["total"] == 1
    assert data_pop["items"][0]["certificate_id"] == "CERT-API-02"
    assert data_pop["items"][0]["decision"] == "FAIL"


def test_api_get_inspection_detail_and_404(api_client_with_db):
    client, Session = api_client_with_db
    rec_id = None
    with Session() as s:
        rec = crud.create_inspection_record(s, _sample_inspection_payload("CERT-DETAIL-01"))
        rec_id = rec.id

    # 1. Success by ID
    res = client.get(f"/api/v1/inspections/{rec_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["certificate_id"] == "CERT-DETAIL-01"
    assert "defect_components" in data
    assert len(data["defect_components"]) == 1

    # 2. Success by Certificate ID string
    res_cert = client.get("/api/v1/inspections/CERT-DETAIL-01")
    assert res_cert.status_code == 200
    assert res_cert.json()["id"] == rec_id

    # 3. 404 for unknown
    res_404 = client.get("/api/v1/inspections/999999")
    assert res_404.status_code == 404


def test_api_export_certificate_json_and_markdown(api_client_with_db):
    client, Session = api_client_with_db
    with Session() as s:
        crud.create_inspection_record(s, _sample_inspection_payload("CERT-EXPORT-01"))

    # JSON export
    res_json = client.get("/api/v1/inspections/CERT-EXPORT-01/certificate?format=json")
    assert res_json.status_code == 200
    assert res_json.json()["certificate_id"] == "CERT-EXPORT-01"
    assert res_json.json()["key_id"] == "KEY-2026-V1"

    # Markdown export
    res_md = client.get("/api/v1/inspections/CERT-EXPORT-01/certificate?format=markdown")
    assert res_md.status_code == 200
    assert "text/markdown" in res_md.headers["content-type"]
    assert "# INDUSTRIAL QUALITY INSPECTION CERTIFICATE" in res_md.text
    assert "HMAC Key Identifier:" in res_md.text


def test_api_analytics_summary_endpoint(api_client_with_db):
    client, Session = api_client_with_db
    with Session() as s:
        crud.create_inspection_record(s, _sample_inspection_payload("CERT-AN-01", "PASS", "NONE", 10.0))
        crud.create_inspection_record(s, _sample_inspection_payload("CERT-AN-02", "FAIL", "CRITICAL", 26.0))

    res = client.get("/api/v1/analytics/summary?days=7")
    assert res.status_code == 200
    data = res.json()
    assert data["total_inspections"] == 2
    assert data["pass_count"] == 1
    assert data["fail_count"] == 1
    assert data["pass_rate_pct"] == 50.0
    assert data["fail_rate_pct"] == 50.0
    assert "latency_metrics" in data


def test_api_zone_heatmap_endpoint(api_client_with_db):
    client, Session = api_client_with_db
    with Session() as s:
        crud.create_inspection_record(s, _sample_inspection_payload("CERT-ZONE-01"))

    res = client.get("/api/v1/analytics/zone-heatmap")
    assert res.status_code == 200
    data = res.json()
    assert "zones" in data
    assert len(data["zones"]) == 1
    assert data["zones"][0]["zone"] == "SEALING_RIM"


def test_api_model_registry_endpoint(api_client_with_db):
    client, _ = api_client_with_db
    res = client.get("/api/v1/models/registry")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert any("PatchCore" in m["model_name"] for m in data["models"])
    assert any("ResNet-18" in m["model_name"] for m in data["models"])
    assert any("YOLO" in m["model_name"] for m in data["models"])
    assert any("U-Net" in m["model_name"] for m in data["models"])


def test_api_batch_inspection_endpoint(api_client_with_db):
    client, _ = api_client_with_db
    img1 = _create_synthetic_image_bytes(color=(128, 128, 128))
    img2 = _create_synthetic_image_bytes(color=(220, 220, 220))

    files = [
        ("files", ("sample1.png", img1, "image/png")),
        ("files", ("sample2.png", img2, "image/png"))
    ]

    res = client.post("/api/v1/inspect/batch?domain=GLASS_BOTTLE&persist_to_db=true", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_submitted"] == 2
    assert data["successful_inspections"] == 2
    assert len(data["results"]) == 2


def test_api_batch_partial_failure_isolation(api_client_with_db):
    """Verifies that an unparseable or corrupted image in a batch does not abort other valid inspections."""
    client, _ = api_client_with_db
    valid_img = _create_synthetic_image_bytes(color=(100, 100, 100))
    corrupted_bytes = b"NOT_A_VALID_IMAGE_DATA_STREAM"

    files = [
        ("files", ("good_item.png", valid_img, "image/png")),
        ("files", ("corrupted_item.png", corrupted_bytes, "image/png"))
    ]

    res = client.post("/api/v1/inspect/batch?domain=GLASS_BOTTLE&persist_to_db=true", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_submitted"] == 2
    assert data["successful_inspections"] == 1
    assert data["failed_inspections"] == 1

    good_result = [r for r in data["results"] if r["filename"] == "good_item.png"][0]
    bad_result = [r for r in data["results"] if r["filename"] == "corrupted_item.png"][0]

    assert good_result["status"] == "success"
    assert bad_result["status"] == "error"
    assert "Failed to decode image" in bad_result["error_message"]


def test_api_archive_endpoint_and_rbac(api_client_with_db):
    """Verifies RBAC protection on soft archival: operator rejected (403), admin accepted (200)."""
    client, Session = api_client_with_db
    rec_id = None
    with Session() as s:
        rec = crud.create_inspection_record(s, _sample_inspection_payload("CERT-ARCH-01"))
        rec_id = rec.id

    # 1. Reject without key
    res_no_key = client.post(f"/api/v1/inspections/{rec_id}/archive", json={"reason": "Auditor archiving"})
    assert res_no_key.status_code == 401

    # 2. Reject with operator key (insufficient role)
    res_op = client.post(
        f"/api/v1/inspections/{rec_id}/archive",
        json={"reason": "Auditor archiving"},
        headers={"X-API-Key": "aoi-operator-secret-key-2026"}
    )
    assert res_op.status_code == 403

    # 3. Accept with admin key
    res_admin = client.post(
        f"/api/v1/inspections/{rec_id}/archive",
        json={"reason": "Annual compliance archive migration"},
        headers={"X-API-Key": "aoi-admin-secret-key-2026"}
    )
    assert res_admin.status_code == 200
    assert res_admin.json()["record_status"] == "ARCHIVED"

    # 4. Standard list excludes archived
    res_list = client.get("/api/v1/inspections")
    assert res_list.json()["total"] == 0

    # 5. List with include_archived=true includes archived
    res_list_all = client.get("/api/v1/inspections?include_archived=true")
    assert res_list_all.json()["total"] == 1


def test_api_rbac_threshold_protection(api_client_with_db):
    """Verifies that operational threshold updates strictly require admin credentials."""
    client, _ = api_client_with_db

    # 1. Reject missing credentials
    res_no_key = client.put("/api/v1/inspect/thresholds", json={"anomaly_threshold": 18.0})
    assert res_no_key.status_code == 401

    # 2. Reject operator key
    res_op = client.put(
        "/api/v1/inspect/thresholds",
        json={"anomaly_threshold": 18.0},
        headers={"X-API-Key": "aoi-operator-secret-key-2026"}
    )
    assert res_op.status_code == 403

    # 3. Accept admin key
    res_admin = client.put(
        "/api/v1/inspect/thresholds",
        json={"anomaly_threshold": 18.0},
        headers={"X-API-Key": "aoi-admin-secret-key-2026"}
    )
    assert res_admin.status_code == 200
    assert res_admin.json()["anomaly_threshold"] == 18.0


def test_api_delete_inspection_endpoint_behavior(api_client_with_db, monkeypatch):
    """Verifies that in compliance mode hard-delete returns 405, and in dev mode cascades successfully."""
    client, Session = api_client_with_db
    rec_id = None
    with Session() as s:
        rec = crud.create_inspection_record(s, _sample_inspection_payload("CERT-DEL-API"))
        rec_id = rec.id

    # 1. Reject wrong key
    res_unauth = client.delete(f"/api/v1/inspections/{rec_id}", headers={"X-API-Key": "wrong-key"})
    assert res_unauth.status_code == 401

    # 2. In compliance mode (default ALLOW_HARD_DELETE=False), hard delete returns 405 Method Not Allowed
    monkeypatch.setattr(settings, "ALLOW_HARD_DELETE", False)
    res_compliance = client.delete(f"/api/v1/inspections/{rec_id}", headers={"X-API-Key": "aoi-admin-secret-key-2026"})
    assert res_compliance.status_code == 405
    assert "Hard deletion is prohibited" in res_compliance.json()["detail"]

    # 3. In dev mode (ALLOW_HARD_DELETE=True), hard delete succeeds
    monkeypatch.setattr(settings, "ALLOW_HARD_DELETE", True)
    res_auth = client.delete(f"/api/v1/inspections/{rec_id}", headers={"X-API-Key": "aoi-admin-secret-key-2026"})
    assert res_auth.status_code == 200
    assert res_auth.json()["status"] == "success"

    # Verify deleted
    with Session() as s:
        assert crud.get_inspection(s, rec_id) is None
