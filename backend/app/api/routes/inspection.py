"""
FastAPI Routes for Industrial Optical Inspection & Defect Analysis.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from backend.app.schemas.inspection import (
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


def get_inspection_service() -> InspectionService:
    """Dependency provider returning singleton inspection service."""
    return InspectionService.get_instance()


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
    service: InspectionService = Depends(get_inspection_service)
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

        p_domain = domain if isinstance(domain, ProductDomain) else ProductDomain(domain)
        result_dict = service.inspect_image(
            image_bytes=image_bytes,
            domain=p_domain,
            render_visuals=render_visuals,
            force_full_pipeline=force_full_pipeline,
            extract_morphology=extract_morphology,
            generate_certificate=generate_certificate
        )
        return result_dict

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Inspection pipeline failure: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal inspection error: {str(e)}")


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
        max_entropy_threshold=t.max_entropy_threshold
    )


@router.put(
    "/thresholds",
    response_model=ThresholdResponse,
    summary="Update Operational Quality Thresholds"
)
def update_operational_thresholds(
    request: ThresholdUpdateRequest,
    service: InspectionService = Depends(get_inspection_service)
):
    updates = request.model_dump(exclude_unset=True)
    updated = service.update_thresholds(updates)
    return ThresholdResponse(
        anomaly_threshold=updated.anomaly_threshold,
        anomaly_margin=updated.anomaly_margin,
        max_acceptable_area_mm2=updated.max_acceptable_area_mm2,
        critical_area_mm2=updated.critical_area_mm2,
        critical_classes=updated.critical_classes,
        min_classifier_confidence=updated.min_classifier_confidence,
        max_entropy_threshold=updated.max_entropy_threshold
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
        distortion_coefficients_loaded=has_dist
    )


@router.put(
    "/calibration",
    response_model=CalibrationResponse,
    summary="Update Camera Calibration Scale"
)
def update_camera_calibration(
    request: CalibrationUpdateRequest,
    service: InspectionService = Depends(get_inspection_service)
):
    cal = service.update_calibration(request.mm_per_pixel)
    has_cam = cal.config.camera_matrix is not None if cal.config else False
    has_dist = cal.config.dist_coeffs is not None if cal.config else False
    return CalibrationResponse(
        mm_per_pixel=cal.mm_per_pixel,
        units="mm",
        camera_matrix_loaded=has_cam,
        distortion_coefficients_loaded=has_dist
    )


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
        device=service.device,
        models_loaded={
            "tier1_patchcore": inspector.tier1_inferencer is not None,
            "tier2_classifier": inspector.tier2_inferencer is not None,
            "tier3_detector": inspector.tier3_detector is not None,
            "tier3_segmenter": inspector.tier3_segmenter is not None
        },
        supported_domains=["GLASS_BOTTLE", "STEEL_SHEET", "GENERIC_AOI"],
        active_calibration_scale=service.calibrator.mm_per_pixel
    )
