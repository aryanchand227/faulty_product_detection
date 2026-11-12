"""
Database CRUD Repository Layer for Industrial AOI Inspections.
Handles relational persistence, relative URI storage, multi-filter historical queries,
cryptographic record verification with key rotation resolution, soft deletion lifecycles,
and aggregate analytics calculations.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hmac
import hashlib
import numpy as np
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.models import (
    InspectionRecord,
    DefectComponentRecord,
    AuditLogRecord,
    utc_now,
)

logger = logging.getLogger(__name__)


def create_inspection_record(
    db: Session,
    result_dict: Dict[str, Any],
    source_image_bytes: Optional[bytes] = None,
    annotated_image_bytes: Optional[bytes] = None,
) -> InspectionRecord:
    """
    Persists an inspection run, stores associated image files with relative URIs,
    and inserts all child defect components within a single database transaction.
    """
    verdict = result_dict.get("verdict", {})
    cert = result_dict.get("certificate", {})
    physical = result_dict.get("physical_metrics", {})
    domain = result_dict.get("product_domain", "GLASS_BOTTLE")
    cert_id = cert.get("certificate_id") or f"CERT-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # 1. Media File Storage (Decoupled Relative URIs)
    orig_uri = None
    annot_uri = None

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_rel_dir = Path(domain) / date_str
    target_abs_dir = settings.STORAGE_DIR / target_rel_dir
    target_abs_dir.mkdir(parents=True, exist_ok=True)

    if source_image_bytes:
        orig_filename = f"{cert_id}_original.jpg"
        orig_file = target_abs_dir / orig_filename
        orig_file.write_bytes(source_image_bytes)
        orig_uri = str(target_rel_dir / orig_filename).replace("\\", "/")

    if annotated_image_bytes:
        annot_filename = f"{cert_id}_annotated.jpg"
        annot_file = target_abs_dir / annot_filename
        annot_file.write_bytes(annotated_image_bytes)
        annot_uri = str(target_rel_dir / annot_filename).replace("\\", "/")

    # 2. Extract values with robust fallbacks
    zone_summary = cert.get("zone_summary", {})

    ts_str = cert.get("timestamp_utc")
    if ts_str:
        try:
            ts_dt = datetime.fromisoformat(ts_str)
        except Exception:
            ts_dt = datetime.now(timezone.utc)
            ts_str = ts_dt.isoformat()
    else:
        ts_dt = datetime.now(timezone.utc)
        ts_str = ts_dt.isoformat()

    record = InspectionRecord(
        certificate_id=cert_id,
        timestamp_utc=ts_dt,
        timestamp_iso=ts_str,
        product_domain=domain,
        decision=cert.get("decision", verdict.get("decision", "PASS")),
        severity=cert.get("severity", verdict.get("severity", "NONE")),
        routing_action=cert.get("routing_action", verdict.get("routing", "LINE_CONTINUE")),
        anomaly_score=float(cert.get("anomaly_score", verdict.get("anomaly_score", 0.0))),
        primary_defect_type=cert.get("primary_defect_type", verdict.get("primary_defect_type", "nominal")),
        dominant_morphology=cert.get("dominant_morphology", "NONE"),
        total_defect_area_mm2=float(cert.get("total_defect_area_mm2", verdict.get("physical_defect_area_mm2", 0.0))),
        component_defect_area_mm2=float(cert.get("component_defect_area_mm2", 0.0)),
        filtered_noise_area_mm2=float(cert.get("filtered_noise_area_mm2", 0.0)),
        is_area_reconciled=bool(cert.get("is_area_reconciled", True)),
        max_single_defect_area_mm2=float(cert.get("max_single_defect_area_mm2", 0.0)),
        max_defect_length_mm=float(cert.get("max_defect_length_mm", 0.0)),
        measurement_mode=physical.get("measurement_mode", cert.get("measurement_mode", "projected_sensor_plane")),
        calibration_scale_mm_per_pixel=float(physical.get("mm_per_pixel", cert.get("calibration_scale_mm_per_pixel", 0.15))),
        engineering_rationale=cert.get("engineering_rationale", verdict.get("engineering_rationale", "")),
        config_version=cert.get("config_version", "CFG-v1.0.0"),
        source_image_hash=cert.get("source_image_hash", hashlib.sha256(source_image_bytes or b"").hexdigest()),
        model_artifact_id=cert.get("model_artifact_id", settings.MODEL_ARTIFACT_ID),
        software_version=cert.get("software_version", settings.SOFTWARE_VERSION),
        calibration_profile_id=cert.get("calibration_profile_id", settings.CALIBRATION_PROFILE_ID),
        decision_policy_version=cert.get("decision_policy_version", settings.DECISION_POLICY_VERSION),
        key_id=cert.get("key_id", settings.ACTIVE_KEY_ID),
        sha256_integrity_hash=cert.get("sha256_integrity_hash", ""),
        hmac_sha256_signature=cert.get("hmac_sha256_signature", cert.get("sha256_audit_signature", "")),
        total_latency_ms=float(result_dict.get("total_latency_ms", 0.0)),
        early_exit_applied=bool(result_dict.get("early_exit_applied", False)),
        original_image_uri=orig_uri,
        annotated_image_uri=annot_uri,
        zone_summary_json=json.dumps(zone_summary),
        record_status="ACTIVE",
    )

    db.add(record)
    db.flush()  # Flush to populate record.id for foreign key

    # 3. Insert Defect Components
    morph_features = result_dict.get("morphological_features", [])
    for comp in morph_features:
        bbox = comp.get("bounding_box_xywh", [])
        centroid = comp.get("centroid_xy", [])
        overlaps = comp.get("zone_overlaps", {})

        c_rec = DefectComponentRecord(
            inspection_id=record.id,
            contour_id=int(comp.get("contour_id", 0)),
            area_pixels=int(comp.get("area_pixels", 0)),
            physical_area_mm2=float(comp.get("physical_area_mm2", 0.0)),
            perimeter_pixels=float(comp.get("perimeter_pixels", 0.0)),
            physical_perimeter_mm=float(comp.get("physical_perimeter_mm", 0.0)),
            bounding_box_json=json.dumps(bbox),
            centroid_xy_json=json.dumps(centroid),
            aspect_ratio=float(comp.get("aspect_ratio", 1.0)),
            circularity=float(min(1.0, max(0.0, comp.get("circularity", 0.0)))),
            is_shape_reliable=bool(comp.get("is_shape_reliable", True)),
            solidity=float(min(1.0, max(0.0, comp.get("solidity", 1.0)))),
            compactness=float(comp.get("compactness", 1.0)),
            extent=float(comp.get("extent", 1.0)),
            equivalent_diameter_mm=float(comp.get("equivalent_diameter_mm", 0.0)),
            major_axis_length_mm=float(comp.get("major_axis_length_mm", 0.0)),
            max_feret_diameter_mm=float(comp.get("max_feret_diameter_mm", 0.0)),
            morphology_type=str(comp.get("morphology_type", "GEOMETRIC_IRREGULAR")),
            spatial_zone=str(comp.get("spatial_zone", "UNKNOWN")),
            zone_overlaps_json=json.dumps(overlaps),
            zone_risk_weight=float(comp.get("zone_risk_weight", 1.0)),
            morphology_risk_weight=float(comp.get("morphology_risk_weight", 1.0)),
            heuristic_severity_score=float(comp.get("heuristic_severity_score", comp.get("adjusted_severity_score", 0.0))),
        )
        db.add(c_rec)

    db.commit()
    db.refresh(record)
    return record


def get_inspection(db: Session, inspection_id_or_cert: Union[int, str]) -> Optional[InspectionRecord]:
    """Retrieves an inspection by primary key integer or unique certificate string."""
    if isinstance(inspection_id_or_cert, int) or (isinstance(inspection_id_or_cert, str) and inspection_id_or_cert.isdigit()):
        return db.query(InspectionRecord).filter(InspectionRecord.id == int(inspection_id_or_cert)).first()
    return db.query(InspectionRecord).filter(InspectionRecord.certificate_id == str(inspection_id_or_cert)).first()


def list_inspections(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    domain: Optional[str] = None,
    decision: Optional[str] = None,
    severity: Optional[str] = None,
    defect_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    include_archived: bool = False,
    record_status: Optional[str] = None,
) -> Tuple[List[InspectionRecord], int]:
    """
    Queries inspection history with multi-parameter filtering and pagination.
    Defaults to returning ACTIVE records only unless include_archived is True.
    """
    query = db.query(InspectionRecord)

    if record_status:
        query = query.filter(InspectionRecord.record_status == record_status)
    elif not include_archived:
        query = query.filter(InspectionRecord.record_status == "ACTIVE")

    if domain:
        query = query.filter(InspectionRecord.product_domain == domain)
    if decision:
        query = query.filter(InspectionRecord.decision == decision)
    if severity:
        query = query.filter(InspectionRecord.severity == severity)
    if defect_type:
        query = query.filter(InspectionRecord.primary_defect_type == defect_type)
    if start_time:
        query = query.filter(InspectionRecord.timestamp_utc >= start_time)
    if end_time:
        query = query.filter(InspectionRecord.timestamp_utc <= end_time)

    total_count = query.count()
    records = query.order_by(desc(InspectionRecord.timestamp_utc)).offset(skip).limit(limit).all()
    return records, total_count


def archive_inspection_record(
    db: Session,
    inspection_id: int,
    actor: str,
    reason: str,
    new_status: str = "ARCHIVED",
) -> Optional[InspectionRecord]:
    """
    Performs soft archival of an inspection record, preserving permanent historical auditability.
    Transitions status to ARCHIVED or LEGAL_HOLD and logs the action immutably.
    """
    rec = db.query(InspectionRecord).filter(InspectionRecord.id == inspection_id).first()
    if not rec:
        return None

    prev_status = rec.record_status
    rec.record_status = new_status
    rec.archived_at = utc_now()
    rec.archived_by = actor
    rec.archive_reason = reason

    # Record permanent audit trail
    record_audit_log(
        db=db,
        actor=actor,
        action=f"RECORD_{new_status}",
        changed_parameters={
            "inspection_id": inspection_id,
            "certificate_id": rec.certificate_id,
            "previous_status": prev_status,
            "new_status": new_status,
            "reason": reason,
        },
        config_version=rec.config_version,
    )

    db.commit()
    db.refresh(rec)
    return rec


def delete_inspection(db: Session, inspection_id: int) -> bool:
    """
    Destructive hard deletion.
    STRICTLY RESTRICTED to development/test environments where settings.ALLOW_HARD_DELETE=True.
    Raises PermissionError in production to preserve audit trail compliance.
    """
    if not settings.ALLOW_HARD_DELETE:
        raise PermissionError(
            "Hard deletion is prohibited in compliance audit mode. "
            "Use archive_inspection_record() for soft retention management."
        )

    rec = db.query(InspectionRecord).filter(InspectionRecord.id == inspection_id).first()
    if not rec:
        return False

    # Clean up physical files on disk if they exist
    for p in [rec.original_image_path, rec.annotated_image_path]:
        if p and Path(p).exists():
            try:
                Path(p).unlink()
            except Exception as e:
                logger.warning(f"Failed to remove stored inspection image {p}: {e}")

    db.delete(rec)
    db.commit()
    return True


def get_analytics_summary(
    db: Session,
    domain: Optional[str] = None,
    start_time: Optional[datetime] = None,
    include_archived: bool = False,
) -> Dict[str, Any]:
    """
    Computes statistical factory KPIs, formal yield breakdowns (PASS, REVIEW, FAIL),
    and latency percentiles (p50, p95, p99).
    Excludes ARCHIVED records by default to prevent skewing operational metrics.
    """
    query = db.query(InspectionRecord)
    if not include_archived:
        query = query.filter(InspectionRecord.record_status == "ACTIVE")
    if domain:
        query = query.filter(InspectionRecord.product_domain == domain)
    if start_time:
        query = query.filter(InspectionRecord.timestamp_utc >= start_time)

    total_inspections = query.count()
    if total_inspections == 0:
        return {
            "total_inspections": 0,
            "pass_count": 0,
            "review_count": 0,
            "fail_count": 0,
            "pass_rate_pct": 0.0,
            "review_rate_pct": 0.0,
            "fail_rate_pct": 0.0,
            "early_exit_count": 0,
            "early_exit_rate_pct": 0.0,
            "mean_anomaly_score": 0.0,
            "latency_metrics": {
                "mean_ms": 0.0,
                "median_p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            },
            "defect_type_distribution": {},
            "morphology_distribution": {},
            "spatial_zone_distribution": {},
            "severity_distribution": {},
        }

    # Decision counts
    decisions = db.query(
        InspectionRecord.decision, func.count(InspectionRecord.id)
    )
    if not include_archived:
        decisions = decisions.filter(InspectionRecord.record_status == "ACTIVE")
    if domain:
        decisions = decisions.filter(InspectionRecord.product_domain == domain)
    if start_time:
        decisions = decisions.filter(InspectionRecord.timestamp_utc >= start_time)
    decision_map = dict(decisions.group_by(InspectionRecord.decision).all())

    pass_count = decision_map.get("PASS", 0)
    review_count = decision_map.get("REVIEW", 0)
    fail_count = decision_map.get("FAIL", 0)

    # Formal statistical factory yield metrics
    pass_rate = round((pass_count / total_inspections) * 100.0, 2)
    review_rate = round((review_count / total_inspections) * 100.0, 2)
    fail_rate = round((fail_count / total_inspections) * 100.0, 2)

    # Early exit count
    ee_query = query.filter(InspectionRecord.early_exit_applied == True).count()
    ee_rate = round((ee_query / total_inspections) * 100.0, 2)

    # Anomaly scores
    mean_score = float(query.with_entities(func.avg(InspectionRecord.anomaly_score)).scalar() or 0.0)

    # Latency percentiles calculation
    latencies = [
        r[0] for r in query.with_entities(InspectionRecord.total_latency_ms).all()
        if r[0] is not None
    ]
    if latencies:
        lat_arr = np.array(latencies, dtype=np.float64)
        mean_latency = float(np.mean(lat_arr))
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        min_lat = float(np.min(lat_arr))
        max_lat = float(np.max(lat_arr))
    else:
        mean_latency = p50 = p95 = p99 = min_lat = max_lat = 0.0

    latency_metrics = {
        "mean_ms": round(mean_latency, 2),
        "median_p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2),
    }

    # Defect type distribution
    defect_types = db.query(
        InspectionRecord.primary_defect_type, func.count(InspectionRecord.id)
    )
    if not include_archived:
        defect_types = defect_types.filter(InspectionRecord.record_status == "ACTIVE")
    if domain:
        defect_types = defect_types.filter(InspectionRecord.product_domain == domain)
    if start_time:
        defect_types = defect_types.filter(InspectionRecord.timestamp_utc >= start_time)
    defect_type_dist = {k: v for k, v in defect_types.group_by(InspectionRecord.primary_defect_type).all() if k != "nominal"}

    # Morphology distribution
    morph_query = db.query(
        InspectionRecord.dominant_morphology, func.count(InspectionRecord.id)
    )
    if not include_archived:
        morph_query = morph_query.filter(InspectionRecord.record_status == "ACTIVE")
    if domain:
        morph_query = morph_query.filter(InspectionRecord.product_domain == domain)
    if start_time:
        morph_query = morph_query.filter(InspectionRecord.timestamp_utc >= start_time)
    morph_dist = {k: v for k, v in morph_query.group_by(InspectionRecord.dominant_morphology).all() if k != "NONE"}

    # Spatial zone distribution (from defect component records)
    zone_query = db.query(
        DefectComponentRecord.spatial_zone, func.count(DefectComponentRecord.id)
    ).join(InspectionRecord, DefectComponentRecord.inspection_id == InspectionRecord.id)
    if not include_archived:
        zone_query = zone_query.filter(InspectionRecord.record_status == "ACTIVE")
    if domain:
        zone_query = zone_query.filter(InspectionRecord.product_domain == domain)
    if start_time:
        zone_query = zone_query.filter(InspectionRecord.timestamp_utc >= start_time)
    zone_dist = dict(zone_query.group_by(DefectComponentRecord.spatial_zone).all())

    # Severity distribution
    sev_query = db.query(
        InspectionRecord.severity, func.count(InspectionRecord.id)
    )
    if not include_archived:
        sev_query = sev_query.filter(InspectionRecord.record_status == "ACTIVE")
    if domain:
        sev_query = sev_query.filter(InspectionRecord.product_domain == domain)
    if start_time:
        sev_query = sev_query.filter(InspectionRecord.timestamp_utc >= start_time)
    sev_dist = dict(sev_query.group_by(InspectionRecord.severity).all())

    return {
        "total_inspections": total_inspections,
        "pass_count": pass_count,
        "review_count": review_count,
        "fail_count": fail_count,
        "pass_rate_pct": pass_rate,
        "review_rate_pct": review_rate,
        "fail_rate_pct": fail_rate,
        "early_exit_count": ee_query,
        "early_exit_rate_pct": ee_rate,
        "mean_anomaly_score": round(mean_score, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "latency_metrics": latency_metrics,
        "defect_type_distribution": defect_type_dist,
        "morphology_distribution": morph_dist,
        "spatial_zone_distribution": zone_dist,
        "severity_distribution": sev_dist,
    }


def verify_stored_record_integrity(
    db: Session,
    inspection_id: int,
    secret_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Cryptographically verifies the authenticity and tamper-evidence of a stored inspection record
    by resolving the appropriate key_id from the key registry and recomputing its canonical payload HMAC.
    """
    rec = db.query(InspectionRecord).filter(InspectionRecord.id == inspection_id).first()
    if not rec:
        return {"is_valid": False, "detail": "Inspection record not found."}

    # Resolve signing key for key rotation resilience
    signing_key = secret_key
    if signing_key is None:
        rec_key_id = rec.key_id or "KEY-2026-V1"
        if rec_key_id in settings.KEY_REGISTRY:
            signing_key = settings.KEY_REGISTRY[rec_key_id]
        elif rec_key_id == settings.ACTIVE_KEY_ID:
            signing_key = settings.DEFAULT_HMAC_SECRET
        else:
            return {
                "is_valid": False,
                "inspection_id": rec.id,
                "certificate_id": rec.certificate_id,
                "key_id": rec_key_id,
                "detail": f"Signing key identifier '{rec_key_id}' not found in active key registry. Cannot verify without corresponding secret.",
            }

    canonical_payload = rec.to_canonical_dict()
    canonical_bytes = json.dumps(canonical_payload, sort_keys=True).encode("utf-8")

    computed_hash = hashlib.sha256(canonical_bytes).hexdigest()
    computed_sig = hmac.new(signing_key, canonical_bytes, hashlib.sha256).hexdigest()

    is_hash_match = hmac.compare_digest(rec.sha256_integrity_hash, computed_hash)
    is_sig_match = hmac.compare_digest(rec.hmac_sha256_signature, computed_sig)

    is_valid = is_hash_match and is_sig_match

    return {
        "is_valid": is_valid,
        "inspection_id": rec.id,
        "certificate_id": rec.certificate_id,
        "key_id": rec.key_id,
        "sha256_integrity_match": is_hash_match,
        "hmac_signature_match": is_sig_match,
        "stored_hash": rec.sha256_integrity_hash,
        "computed_hash": computed_hash,
        "stored_signature": rec.hmac_sha256_signature,
        "computed_signature": computed_sig,
        "record_status": rec.record_status,
    }


def record_audit_log(
    db: Session,
    actor: str,
    action: str,
    changed_parameters: Dict[str, Any],
    config_version: str,
) -> AuditLogRecord:
    """Inserts an immutable audit record for configuration updates and lifecycle events."""
    log_rec = AuditLogRecord(
        timestamp_utc=datetime.now(timezone.utc),
        actor=actor,
        action=action,
        changed_parameters_json=json.dumps(changed_parameters),
        config_version=config_version,
    )
    db.add(log_rec)
    db.commit()
    db.refresh(log_rec)
    return log_rec
