"""
Integration and End-to-End Tests for Historical Inspection Queries,
Cryptographic Certificate Verification, Analytics Aggregation, and Batch Processing API.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

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

    # Markdown export
    res_md = client.get("/api/v1/inspections/CERT-EXPORT-01/certificate?format=markdown")
    assert res_md.status_code == 200
    assert "text/markdown" in res_md.headers["content-type"]
    assert "# INDUSTRIAL QUALITY INSPECTION CERTIFICATE" in res_md.text


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
    tiers = [m["tier"] for m in data["models"]]
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


def test_api_delete_inspection_endpoint_authorized(api_client_with_db):
    client, Session = api_client_with_db
    rec_id = None
    with Session() as s:
        rec = crud.create_inspection_record(s, _sample_inspection_payload("CERT-DEL-API"))
        rec_id = rec.id

    # 1. Reject without API key or with invalid key
    res_unauth = client.delete(f"/api/v1/inspections/{rec_id}", headers={"X-API-Key": "wrong-key"})
    assert res_unauth.status_code == 401

    # 2. Success with valid API key
    res_auth = client.delete(f"/api/v1/inspections/{rec_id}", headers={"X-API-Key": "aoi-admin-secret-key-2026"})
    assert res_auth.status_code == 200
    assert res_auth.json()["status"] == "success"

    # Verify deleted
    with Session() as s:
        assert crud.get_inspection(s, rec_id) is None
