"""
FastAPI Routes for Industrial Optical Inspection & Defect Analysis.
Features threadpool offloading, authorization guards, and configuration audit trails.
"""

import base64
import logging
import os
import uuid
from typing import List, Optional
import anyio
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.connection import get_db
from backend.app.database import crud
from backend.app.schemas.history import BatchInspectionItem, BatchInspectionResponse
from backend.app.schemas.inspection import (
    AuditLogEntrySchema,
    CalibrationResponse,
    CalibrationUpdateRequest,
    InspectionResponseSchema,
    SystemInfoResponse,
    ThresholdResponse,
    ThresholdUpdateRequest,
)
from backend.app.services.inspection_service import InspectionService
from src.inspection.unified_pipeline import ProductDomain

logger = logging.getLogger(__name__)

router = APIRouter()

# Max allowable file upload size (15 MB)
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024

# Admin API Key for mutating operational parameters
EXPECTED_ADMIN_KEY = os.getenv("AOI_ADMIN_API_KEY", "aoi-admin-secret-key-2026")


def get_inspection_service() -> InspectionService:
    """Dependency provider returning singleton inspection service."""
    return InspectionService.get_instance()


def verify_admin_authorization(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    Security gatekeeper: validates admin authorization key for configuration mutation.
    """
    if x_api_key is not None and x_api_key != EXPECTED_ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unauthorized X-API-Key header for configuration mutation."
        )
    return x_api_key or "default_operator"


@router.post(
    "/image",
    response_model=InspectionResponseSchema,
    summary="Inspect Product Image",
    description="Uploads a product image and executes multi-tier inspection with early-exit nominal bypass, calibrated damaged area, morphological shape descriptors, and cryptographic inspection certificate."
)
async def inspect_product_image(
    file: UploadFile = File(..., description="JPEG/PNG product image from inspection camera"),
    domain: ProductDomain = Query(ProductDomain.GLASS_BOTTLE, description="Target product inspection domain"),
    render_visuals: bool = Query(True, description="Render annotated HUD inspection overlay banner"),
    force_full_pipeline: bool = Query(False, description="Force execution of all tiers even if nominal"),
    extract_morphology: bool = Query(True, description="Extract granular contour shape metrics"),
    generate_certificate: bool = Query(True, description="Generate cryptographically signed compliance certificate"),
    persist_to_db: bool = Query(True, description="Persist inspection run and defect components to relational database"),
    service: InspectionService = Depends(get_inspection_service),
    db: Session = Depends(get_db)
):
    # Validate content type
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp", "application/octet-stream"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image media type: {file.content_type}. Please upload JPEG or PNG."
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
        if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded image size ({len(image_bytes)} bytes) exceeds maximum limit ({MAX_UPLOAD_SIZE_BYTES} bytes)."
            )

        p_domain = domain if isinstance(domain, ProductDomain) else ProductDomain(domain)

        # Offload CPU/GPU-bound computer vision inference to worker threadpool
        # (Guarantees FastAPI asynchronous event loop is never blocked)
        result_dict = await anyio.to_thread.run_sync(
            service.inspect_image,
            image_bytes,
            p_domain,
            render_visuals,
            force_full_pipeline,
            extract_morphology,
            generate_certificate
        )

        # Database Persistence
        if persist_to_db:
            try:
                annotated_bytes = None
                if result_dict.get("annotated_image_base64"):
                    annotated_bytes = base64.b64decode(result_dict["annotated_image_base64"])
                crud.create_inspection_record(
                    db=db,
                    result_dict=result_dict,
                    source_image_bytes=image_bytes,
                    annotated_image_bytes=annotated_bytes
                )
            except Exception as db_err:
                logger.error(f"Failed to persist inspection record to database: {db_err}", exc_info=True)

        return result_dict

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inspection pipeline failure: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal inspection error: {str(e)}")


@router.post(
    "/batch",
    response_model=BatchInspectionResponse,
    summary="Batch Inspect Multiple Product Images",
    description="Processes multiple uploaded product images, persisting results and returning aggregated triage statistics."
)
async def batch_inspect_images(
    files: List[UploadFile] = File(..., description="List of product image files to inspect"),
    domain: ProductDomain = Query(ProductDomain.GLASS_BOTTLE, description="Target product domain"),
    persist_to_db: bool = Query(True, description="Persist results to database"),
    service: InspectionService = Depends(get_inspection_service),
    db: Session = Depends(get_db)
):
    if len(files) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files submitted in batch request.")
    if len(files) > settings.MAX_BATCH_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size ({len(files)}) exceeds maximum limit of {settings.MAX_BATCH_FILES} files."
        )

    batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
    results: List[BatchInspectionItem] = []
    pass_cnt = 0
    rev_cnt = 0
    fail_cnt = 0
    latencies: List[float] = []

    p_domain = domain if isinstance(domain, ProductDomain) else ProductDomain(domain)

    for f in files:
        fname = f.filename or "unknown.png"
        try:
            content = await f.read()
            if len(content) == 0:
                results.append(BatchInspectionItem(filename=fname, status="error", error_message="Empty file."))
                continue

            result_dict = await anyio.to_thread.run_sync(
                service.inspect_image,
                content,
                p_domain,
                False,  # Skip visual rendering for throughput
                False,
                True,
                True
            )

            verdict = result_dict.get("verdict", {})
            cert = result_dict.get("certificate", {})
            dec = verdict.get("decision", "PASS")
            sev = verdict.get("severity", "NONE")
            rout = verdict.get("routing", "LINE_CONTINUE")
            score = float(verdict.get("anomaly_score", 0.0))
            latency = float(result_dict.get("total_latency_ms", 0.0))
            latencies.append(latency)

            if dec == "PASS":
                pass_cnt += 1
            elif dec == "REVIEW":
                rev_cnt += 1
            else:
                fail_cnt += 1

            rec_id = None
            if persist_to_db:
                try:
                    db_rec = crud.create_inspection_record(
                        db=db,
                        result_dict=result_dict,
                        source_image_bytes=content,
                        annotated_image_bytes=None
                    )
                    rec_id = db_rec.id
                except Exception as dbe:
                    logger.warning(f"Batch item {fname} DB save failure: {dbe}")

            results.append(BatchInspectionItem(
                filename=fname,
                status="success",
                inspection_id=rec_id,
                certificate_id=cert.get("certificate_id"),
                decision=dec,
                severity=sev,
                routing=rout,
                anomaly_score=round(score, 4),
                total_latency_ms=round(latency, 2),
            ))

        except Exception as err:
            logger.error(f"Batch item {fname} failed: {err}")
            results.append(BatchInspectionItem(filename=fname, status="error", error_message=str(err)))

    successes = sum(1 for r in results if r.status == "success")
    failures = len(results) - successes
    mean_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    return BatchInspectionResponse(
        batch_id=batch_id,
        total_submitted=len(files),
        successful_inspections=successes,
        failed_inspections=failures,
        pass_count=pass_cnt,
        review_count=rev_cnt,
        fail_count=fail_cnt,
        mean_latency_ms=mean_lat,
        results=results
    )


@router.get(
    "/thresholds",
    response_model=ThresholdResponse,
    summary="Get Operational Quality Thresholds"
)
def get_operational_thresholds(service: InspectionService = Depends(get_inspection_service)):
    t = service.thresholds
    return ThresholdResponse(
        anomaly_threshold=t.anomaly_threshold,
        anomaly_margin=t.anomaly_margin,
        max_acceptable_area_mm2=t.max_acceptable_area_mm2,
        critical_area_mm2=t.critical_area_mm2,
        critical_classes=t.critical_classes,
        min_classifier_confidence=t.min_classifier_confidence,
        max_entropy_threshold=t.max_entropy_threshold,
        config_version=service.config_version
    )


@router.put(
    "/thresholds",
    response_model=ThresholdResponse,
    summary="Update Operational Quality Thresholds"
)
def update_operational_thresholds(
    request: ThresholdUpdateRequest,
    actor: str = Depends(verify_admin_authorization),
    service: InspectionService = Depends(get_inspection_service)
):
    updates = request.model_dump(exclude_unset=True)
    updated = service.update_thresholds(updates, actor=actor)
    return ThresholdResponse(
        anomaly_threshold=updated.anomaly_threshold,
        anomaly_margin=updated.anomaly_margin,
        max_acceptable_area_mm2=updated.max_acceptable_area_mm2,
        critical_area_mm2=updated.critical_area_mm2,
        critical_classes=updated.critical_classes,
        min_classifier_confidence=updated.min_classifier_confidence,
        max_entropy_threshold=updated.max_entropy_threshold,
        config_version=service.config_version
    )


@router.get(
    "/calibration",
    response_model=CalibrationResponse,
    summary="Get Camera Calibration Parameters"
)
def get_camera_calibration(service: InspectionService = Depends(get_inspection_service)):
    cal = service.calibrator
    has_cam = cal.config.camera_matrix is not None if cal.config else False
    has_dist = cal.config.dist_coeffs is not None if cal.config else False
    return CalibrationResponse(
        mm_per_pixel=cal.mm_per_pixel,
        units="mm",
        camera_matrix_loaded=has_cam,
        distortion_coefficients_loaded=has_dist,
        config_version=service.config_version
    )


@router.put(
    "/calibration",
    response_model=CalibrationResponse,
    summary="Update Camera Calibration Scale"
)
def update_camera_calibration(
    request: CalibrationUpdateRequest,
    actor: str = Depends(verify_admin_authorization),
    service: InspectionService = Depends(get_inspection_service)
):
    cal = service.update_calibration(request.mm_per_pixel, actor=actor)
    has_cam = cal.config.camera_matrix is not None if cal.config else False
    has_dist = cal.config.dist_coeffs is not None if cal.config else False
    return CalibrationResponse(
        mm_per_pixel=cal.mm_per_pixel,
        units="mm",
        camera_matrix_loaded=has_cam,
        distortion_coefficients_loaded=has_dist,
        config_version=service.config_version
    )


@router.get(
    "/audit-log",
    response_model=List[AuditLogEntrySchema],
    summary="Get Configuration Change Audit Trail"
)
def get_audit_log(service: InspectionService = Depends(get_inspection_service)):
    return service.audit_log


@router.get(
    "/system-info",
    response_model=SystemInfoResponse,
    summary="System and Model Readiness Info"
)
def get_system_info(service: InspectionService = Depends(get_inspection_service)):
    inspector = service.inspector
    return SystemInfoResponse(
        service_name="Industrial Quality Inspection API",
        version="1.0.0",
        config_version=service.config_version,
        device=service.device,
        models_loaded={
            "tier1_patchcore": inspector.tier1_inferencer is not None,
            "tier2_classifier": inspector.tier2_inferencer is not None,
            "tier3_detector": inspector.tier3_detector is not None,
            "tier3_segmenter": inspector.tier3_segmenter is not None
        },
        supported_domains=["GLASS_BOTTLE", "STEEL_SHEET", "GENERIC_AOI"],
        active_calibration_scale=service.calibrator.mm_per_pixel,
        active_audit_entries_count=len(service.audit_log)
    )
