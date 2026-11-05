"""
Unit and Integration Tests for Industrial AOI Database Persistence Layer.
Verifies SQLite/PostgreSQL ORM mapping, CRUD operations, relational cascades,
cryptographic tamper detection, and quality KPI analytics.
"""

from datetime import datetime, timezone
import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database.connection import Base
from backend.app.database import crud
from backend.app.database.models import (
    InspectionRecord,
    DefectComponentRecord,
    AuditLogRecord,
)


@pytest.fixture(scope="function")
def db_session():
    """Provides an isolated in-memory SQLite database session for each test function."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def _sample_inspection_payload(cert_id="CERT-TEST-001", decision="FAIL", severity="CRITICAL", anomaly_score=25.5):
    """Generates synthetic inspection output payload."""
    return {
        "product_domain": "GLASS_BOTTLE",
        "total_latency_ms": 42.5,
        "early_exit_applied": False,
        "verdict": {
            "decision": decision,
            "severity": severity,
            "routing": "AUTOMATIC_REJECT_BIN",
            "anomaly_score": anomaly_score,
            "primary_defect_type": "glass_anomaly",
            "physical_defect_area_mm2": 150.25,
            "engineering_rationale": "Test failure rationale"
        },
        "physical_metrics": {
            "measurement_mode": "projected_sensor_plane",
            "mm_per_pixel": 0.15
        },
        "certificate": {
            "certificate_id": cert_id,
            "dominant_morphology": "GEOMETRIC_ELONGATED_SLIT",
            "total_defect_area_mm2": 150.25,
            "component_defect_area_mm2": 145.0,
            "filtered_noise_area_mm2": 5.25,
            "is_area_reconciled": True,
            "max_single_defect_area_mm2": 145.0,
            "max_defect_length_mm": 28.5,
            "config_version": "CFG-v1.0.0",
            "source_image_hash": "a1b2c3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "sha256_integrity_hash": "dummy_integrity_hash",
            "hmac_sha256_signature": "dummy_hmac_signature",
            "zone_summary": {
                "SEALING_RIM": {"count": 1, "total_area_mm2": 145.0, "risk_weight": 2.5}
            }
        },
        "morphological_features": [
            {
                "contour_id": 1,
                "area_pixels": 6444,
                "physical_area_mm2": 145.0,
                "perimeter_pixels": 450.0,
                "physical_perimeter_mm": 67.5,
                "bounding_box_xywh": [100, 20, 50, 190],
                "centroid_xy": [125.0, 115.0],
                "aspect_ratio": 3.8,
                "circularity": 0.18,
                "is_shape_reliable": True,
                "solidity": 0.88,
                "compactness": 31.4,
                "extent": 0.65,
                "equivalent_diameter_mm": 13.59,
                "major_axis_length_mm": 28.5,
                "max_feret_diameter_mm": 29.1,
                "morphology_type": "GEOMETRIC_ELONGATED_SLIT",
                "spatial_zone": "SEALING_RIM",
                "zone_overlaps": {"SEALING_RIM": 1.0},
                "zone_risk_weight": 2.5,
                "morphology_risk_weight": 2.2,
                "heuristic_severity_score": 797.5
            }
        ]
    }


def test_crud_create_and_get_inspection(db_session):
    payload = _sample_inspection_payload(cert_id="CERT-20261106-A1B2")
    rec = crud.create_inspection_record(db_session, payload)

    assert rec.id is not None
    assert rec.certificate_id == "CERT-20261106-A1B2"
    assert rec.decision == "FAIL"
    assert rec.severity == "CRITICAL"
    assert rec.product_domain == "GLASS_BOTTLE"
    assert len(rec.components) == 1
    assert rec.components[0].morphology_type == "GEOMETRIC_ELONGATED_SLIT"
    assert rec.components[0].spatial_zone == "SEALING_RIM"

    # Query by ID and by certificate string
    by_id = crud.get_inspection(db_session, rec.id)
    assert by_id is not None
    assert by_id.certificate_id == "CERT-20261106-A1B2"

    by_cert = crud.get_inspection(db_session, "CERT-20261106-A1B2")
    assert by_cert is not None
    assert by_cert.id == rec.id


def test_crud_list_inspections_pagination_and_filters(db_session):
    # Insert 3 sample records
    crud.create_inspection_record(db_session, _sample_inspection_payload("CERT-01", "PASS", "NONE", 11.2))
    crud.create_inspection_record(db_session, _sample_inspection_payload("CERT-02", "REVIEW", "LOW", 16.0))
    crud.create_inspection_record(db_session, _sample_inspection_payload("CERT-03", "FAIL", "CRITICAL", 28.4))

    # All records
    recs, total = crud.list_inspections(db_session, skip=0, limit=10)
    assert total == 3
    assert len(recs) == 3

    # Filter by decision
    recs_pass, total_pass = crud.list_inspections(db_session, decision="PASS")
    assert total_pass == 1
    assert recs_pass[0].decision == "PASS"

    recs_fail, total_fail = crud.list_inspections(db_session, decision="FAIL")
    assert total_fail == 1
    assert recs_fail[0].decision == "FAIL"

    # Pagination
    recs_p1, _ = crud.list_inspections(db_session, skip=0, limit=2)
    assert len(recs_p1) == 2
    recs_p2, _ = crud.list_inspections(db_session, skip=2, limit=2)
    assert len(recs_p2) == 1


def test_crud_cascade_delete(db_session):
    payload = _sample_inspection_payload("CERT-DEL-01")
    rec = crud.create_inspection_record(db_session, payload)
    rec_id = rec.id

    assert db_session.query(DefectComponentRecord).filter(DefectComponentRecord.inspection_id == rec_id).count() == 1

    deleted = crud.delete_inspection(db_session, rec_id)
    assert deleted is True

    # Confirm parent and children are deleted
    assert db_session.query(InspectionRecord).filter(InspectionRecord.id == rec_id).first() is None
    assert db_session.query(DefectComponentRecord).filter(DefectComponentRecord.inspection_id == rec_id).count() == 0


def test_crud_analytics_summary_kpis(db_session):
    crud.create_inspection_record(db_session, _sample_inspection_payload("CERT-K1", "PASS", "NONE", 10.0))
    crud.create_inspection_record(db_session, _sample_inspection_payload("CERT-K2", "PASS", "NONE", 12.0))
    crud.create_inspection_record(db_session, _sample_inspection_payload("CERT-K3", "FAIL", "CRITICAL", 25.0))
    crud.create_inspection_record(db_session, _sample_inspection_payload("CERT-K4", "REVIEW", "MEDIUM", 16.5))

    stats = crud.get_analytics_summary(db_session, domain="GLASS_BOTTLE")

    assert stats["total_inspections"] == 4
    assert stats["pass_count"] == 2
    assert stats["review_count"] == 1
    assert stats["fail_count"] == 1
    assert stats["pass_rate_pct"] == 50.0
    assert stats["mean_anomaly_score"] == (10.0 + 12.0 + 25.0 + 16.5) / 4.0
    assert "GEOMETRIC_ELONGATED_SLIT" in stats["morphology_distribution"]
    assert "SEALING_RIM" in stats["spatial_zone_distribution"]


def test_crud_cryptographic_tamper_detection(db_session):
    """
    Verifies that authentic certificates pass HMAC verification, and that any
    unauthorized modification to a database record (e.g. altering the decision or anomaly score)
    is caught as cryptographic tampering.
    """
    import hmac
    import hashlib
    secret_key = b"test-secret-key-1234"

    payload = _sample_inspection_payload("CERT-TAMPER-01", "FAIL", "CRITICAL", 22.5)
    payload["morphological_features"] = []

    # Pre-generate authentic HMAC and hash
    from src.inspection.defect_analysis import InspectionReportGenerator
    from src.inspection.decision_engine import InspectionVerdict, InspectionDecision, SeverityLevel, RoutingAction
    from src.inspection.unified_pipeline import UnifiedInspectionResult
    verdict = InspectionVerdict(
        decision=InspectionDecision.FAIL,
        severity=SeverityLevel.CRITICAL,
        routing=RoutingAction.AUTOMATIC_REJECT_BIN,
        primary_defect_type="glass_anomaly",
        confidence=1.0,
        physical_defect_area_mm2=150.25,
        anomaly_score=22.5,
        is_anomalous=True,
        engineering_rationale="Test failure rationale"
    )
    res = UnifiedInspectionResult(
        verdict=verdict,
        tier1_anomaly=None,
        tier2_classification=None,
        tier3_detection=None,
        tier3_segmentation=None,
        physical_metrics={"measurement_mode": "projected_sensor_plane", "scale_mm_per_pixel": 0.15},
        total_latency_ms=40.0,
        early_exit_applied=False,
        product_domain="GLASS_BOTTLE"
    )
    cert = InspectionReportGenerator.generate_certificate(
        res,
        morphological_features=[],
        config_version="CFG-v1.0.0",
        certificate_prefix="CERT-TAMPER",
        hmac_secret=secret_key
    )

    payload["certificate"] = cert.to_dict()
    rec = crud.create_inspection_record(db_session, payload)

    # 1. Untampered record verification
    ver_clean = crud.verify_stored_record_integrity(db_session, rec.id, secret_key=secret_key)
    assert ver_clean["is_valid"] is True
    assert ver_clean["sha256_integrity_match"] is True
    assert ver_clean["hmac_signature_match"] is True

    # 2. Tampered record: attacker changes decision from FAIL to PASS
    rec.decision = "PASS"
    db_session.commit()

    ver_tampered = crud.verify_stored_record_integrity(db_session, rec.id, secret_key=secret_key)
    assert ver_tampered["is_valid"] is False
    assert ver_tampered["sha256_integrity_match"] is False
    assert ver_tampered["hmac_signature_match"] is False


def test_crud_audit_log_record(db_session):
    audit_rec = crud.record_audit_log(
        db_session,
        actor="lead_quality_engineer",
        action="UPDATE_THRESHOLDS",
        changed_parameters={"anomaly_threshold": {"old": 15.5, "new": 16.0}},
        config_version="CFG-v1.0.1"
    )
    assert audit_rec.id is not None
    assert audit_rec.actor == "lead_quality_engineer"
    assert audit_rec.config_version == "CFG-v1.0.1"
    d = audit_rec.to_dict()
    assert d["changed_parameters"]["anomaly_threshold"]["new"] == 16.0
