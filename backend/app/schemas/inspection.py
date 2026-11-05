"""
Pydantic Schemas for Industrial Optical Inspection REST API.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ThresholdUpdateRequest(BaseModel):
    """Payload for updating operational quality thresholds."""
    anomaly_threshold: Optional[float] = Field(None, ge=0.0, le=100.0, description="PatchCore anomaly distance cutoff")
    anomaly_margin: Optional[float] = Field(None, ge=0.0, le=0.50, description="Uncertainty band margin delta")
    max_acceptable_area_mm2: Optional[float] = Field(None, ge=0.0, description="Maximum allowable cosmetic micro-flaw area")
    critical_area_mm2: Optional[float] = Field(None, ge=0.0, description="Critical damage area triggering immediate rejection")
    critical_classes: Optional[List[str]] = Field(None, description="Defect categories triggering automatic FAIL")
    min_classifier_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_entropy_threshold: Optional[float] = Field(None, ge=0.0)


class ThresholdResponse(BaseModel):
    """Operational quality threshold configuration."""
    anomaly_threshold: float
    anomaly_margin: float
    max_acceptable_area_mm2: float
    critical_area_mm2: float
    critical_classes: List[str]
    min_classifier_confidence: float
    max_entropy_threshold: float


class CalibrationUpdateRequest(BaseModel):
    """Payload for updating metric camera scale."""
    mm_per_pixel: float = Field(..., gt=0.0, description="Physical millimeter length per sensor pixel")
    units: Optional[str] = Field("mm", description="Measurement units")


class CalibrationResponse(BaseModel):
    """Active camera calibration parameters."""
    mm_per_pixel: float
    units: str
    camera_matrix_loaded: bool
    distortion_coefficients_loaded: bool


class MorphologicalFeatureItem(BaseModel):
    """Shape descriptors for an individual segmented defect component."""
    contour_id: int
    area_pixels: int
    physical_area_mm2: float
    perimeter_pixels: float
    physical_perimeter_mm: float
    bounding_box_xywh: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    aspect_ratio: float
    circularity: float
    solidity: float
    compactness: float
    extent: float
    equivalent_diameter_mm: float
    morphology_type: str
    spatial_zone: str
    zone_risk_weight: float
    morphology_risk_weight: float
    adjusted_severity_score: float


class InspectionVerdictSchema(BaseModel):
    """Decision engine quality triage verdict."""
    decision: str = Field(..., description="Operational triage: PASS, REVIEW, or FAIL")
    severity: str = Field(..., description="Damage severity: NONE, LOW, MEDIUM, or CRITICAL")
    routing: str = Field(..., description="Conveyor diverter action: LINE_CONTINUE, MANUAL_INSPECTION_BAY, or AUTOMATIC_REJECT_BIN")
    primary_defect_type: str
    confidence: float
    physical_defect_area_mm2: float
    anomaly_score: float
    is_anomalous: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    engineering_rationale: str = ""


class InspectionCertificateSchema(BaseModel):
    """Cryptographically verifiable inspection certificate."""
    certificate_id: str
    timestamp_utc: str
    product_domain: str
    source_image_hash: str
    decision: str
    severity: str
    routing_action: str
    anomaly_score: float
    primary_defect_type: str
    dominant_morphology: str
    total_defect_area_mm2: float
    max_single_defect_area_mm2: float
    measurement_mode: str
    num_defect_components: int
    zone_summary: Dict[str, Dict[str, Any]]
    defect_components: List[Dict[str, Any]]
    engineering_rationale: str
    calibration_scale_mm_per_pixel: float
    sha256_audit_signature: str


class InspectionResponseSchema(BaseModel):
    """Comprehensive inspection report returned by API."""
    verdict: InspectionVerdictSchema
    early_exit_applied: bool
    total_latency_ms: float
    product_domain: str
    physical_metrics: Dict[str, Any]
    morphological_features: List[MorphologicalFeatureItem] = Field(default_factory=list)
    certificate: Optional[InspectionCertificateSchema] = None
    annotated_image_base64: Optional[str] = None


class SystemInfoResponse(BaseModel):
    """System health, hardware device, and active pipeline configuration."""
    service_name: str
    version: str
    device: str
    models_loaded: Dict[str, bool]
    supported_domains: List[str]
    active_calibration_scale: float
