"""
Historical Inspection Records and Cryptographic Audit Verification Endpoints.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.api.routes.inspection import verify_admin_authorization
from backend.app.database.connection import get_db
from backend.app.database import crud
from backend.app.schemas.history import (
    InspectionSummaryItem,
    PaginatedInspectionsResponse,
    VerificationResponse,
)

router = APIRouter()


@router.get(
    "/inspections",
    response_model=PaginatedInspectionsResponse,
    summary="Query Inspection History"
)
def list_inspections(
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(20, ge=1, le=100, description="Page size limit"),
    domain: Optional[str] = Query(None, description="Filter by product domain: GLASS_BOTTLE or STEEL_SHEET"),
    decision: Optional[str] = Query(None, description="Filter by triage decision: PASS, REVIEW, or FAIL"),
    severity: Optional[str] = Query(None, description="Filter by severity: NONE, LOW, MEDIUM, or CRITICAL"),
    defect_type: Optional[str] = Query(None, description="Filter by defect type: nominal, broken_large, etc."),
    start_date: Optional[datetime] = Query(None, description="Filter inspections on or after this UTC timestamp"),
    end_date: Optional[datetime] = Query(None, description="Filter inspections on or before this UTC timestamp"),
    db: Session = Depends(get_db),
):
    """
    Retrieves a paginated list of historical inspection records with multi-criteria filtering.
    """
    records, total = crud.list_inspections(
        db,
        skip=skip,
        limit=limit,
        domain=domain,
        decision=decision,
        severity=severity,
        defect_type=defect_type,
        start_time=start_date,
        end_time=end_date,
    )

    items = []
    for r in records:
        items.append(
            InspectionSummaryItem(
                id=r.id,
                certificate_id=r.certificate_id,
                timestamp_utc=r.timestamp_utc.isoformat(),
                product_domain=r.product_domain,
                decision=r.decision,
                severity=r.severity,
                routing_action=r.routing_action,
                anomaly_score=round(r.anomaly_score, 4),
                primary_defect_type=r.primary_defect_type,
                dominant_morphology=r.dominant_morphology,
                total_defect_area_mm2=round(r.total_defect_area_mm2, 3),
                max_defect_length_mm=round(r.max_defect_length_mm, 3),
                total_latency_ms=round(r.total_latency_ms, 2),
                early_exit_applied=r.early_exit_applied,
                num_defect_components=len(r.components),
                has_images=bool(r.original_image_path or r.annotated_image_path),
            )
        )

    total_pages = (total + limit - 1) // limit if total > 0 else 0
    current_page = (skip // limit) + 1 if limit > 0 else 1

    return PaginatedInspectionsResponse(
        total=total,
        page=current_page,
        page_size=limit,
        total_pages=total_pages,
        items=items,
    )


@router.get(
    "/inspections/{inspection_id}",
    summary="Get Detailed Inspection Record"
)
def get_inspection_detail(
    inspection_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieves comprehensive inspection details including individual defect contours,
    morphological descriptors, spatial zoning distribution, and cryptographic hashes.
    """
    rec = crud.get_inspection(db, inspection_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection record '{inspection_id}' not found."
        )
    return rec.to_dict(include_components=True)


@router.get(
    "/inspections/{inspection_id}/certificate",
    summary="Export Signed Inspection Certificate"
)
def export_certificate(
    inspection_id: str,
    format: str = Query("json", pattern="^(json|markdown)$", description="Export format: json or markdown"),
    db: Session = Depends(get_db),
):
    """
    Exports the authentic cryptographic inspection certificate.
    """
    rec = crud.get_inspection(db, inspection_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection record '{inspection_id}' not found."
        )

    cert_data = {
        "certificate_id": rec.certificate_id,
        "timestamp_utc": rec.timestamp_utc.isoformat(),
        "product_domain": rec.product_domain,
        "config_version": rec.config_version,
        "source_image_hash": rec.source_image_hash,
        "decision": rec.decision,
        "severity": rec.severity,
        "routing_action": rec.routing_action,
        "anomaly_score": round(rec.anomaly_score, 4),
        "primary_defect_type": rec.primary_defect_type,
        "dominant_morphology": rec.dominant_morphology,
        "total_defect_area_mm2": round(rec.total_defect_area_mm2, 3),
        "component_defect_area_mm2": round(rec.component_defect_area_mm2, 3),
        "filtered_noise_area_mm2": round(rec.filtered_noise_area_mm2, 3),
        "is_area_reconciled": rec.is_area_reconciled,
        "max_single_defect_area_mm2": round(rec.max_single_defect_area_mm2, 3),
        "max_defect_length_mm": round(rec.max_defect_length_mm, 3),
        "measurement_mode": rec.measurement_mode,
        "num_defect_components": len(rec.components),
        "zone_summary": json.loads(rec.zone_summary_json) if rec.zone_summary_json else {},
        "defect_components": [comp.to_dict() for comp in rec.components],
        "engineering_rationale": rec.engineering_rationale,
        "calibration_scale_mm_per_pixel": round(rec.calibration_scale_mm_per_pixel, 5),
        "sha256_integrity_hash": rec.sha256_integrity_hash,
        "hmac_sha256_signature": rec.hmac_sha256_signature,
        "sha256_audit_signature": rec.hmac_sha256_signature,
    }

    if format == "markdown":
        from src.inspection.defect_analysis import InspectionCertificate
        md_kwargs = dict(cert_data)
        md_kwargs.pop("sha256_audit_signature", None)
        cert_obj = InspectionCertificate(**md_kwargs)
        return Response(content=cert_obj.to_markdown(), media_type="text/markdown")

    return cert_data


@router.post(
    "/inspections/{inspection_id}/verify",
    response_model=VerificationResponse,
    summary="Cryptographically Verify Stored Inspection Record"
)
def verify_record_integrity(
    inspection_id: int,
    db: Session = Depends(get_db),
):
    """
    Recomputes the HMAC-SHA256 signature and SHA-256 integrity hash directly from stored database rows
    and verifies that the database records have not been tampered with or corrupted.
    """
    res = crud.verify_stored_record_integrity(db, inspection_id)
    if "is_valid" not in res or not res.get("stored_signature"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=res.get("detail", "Inspection record not found.")
        )
    return VerificationResponse(**res)


@router.get(
    "/inspections/{inspection_id}/image",
    summary="Retrieve Stored Inspection Image"
)
def get_inspection_image(
    inspection_id: str,
    image_type: str = Query("annotated", pattern="^(original|annotated)$", description="Image type to retrieve"),
    db: Session = Depends(get_db),
):
    """
    Serves the persisted original or annotated HUD inspection image from the server media repository.
    """
    rec = crud.get_inspection(db, inspection_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspection record not found.")

    target_path = rec.annotated_image_path if image_type == "annotated" else rec.original_image_path
    if not target_path or not Path(target_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Requested {image_type} image file is not stored for this record."
        )

    return FileResponse(target_path, media_type="image/jpeg")


@router.delete(
    "/inspections/{inspection_id}",
    summary="Delete Inspection Record (Admin Only)"
)
def delete_inspection(
    inspection_id: int,
    db: Session = Depends(get_db),
    authorized: bool = Depends(verify_admin_authorization),
):
    """
    Deletes an inspection record, cascaded child components, and stored images.
    Protected by admin API key.
    """
    deleted = crud.delete_inspection(db, inspection_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspection record not found.")
    return {"status": "success", "message": f"Inspection record {inspection_id} deleted successfully."}
