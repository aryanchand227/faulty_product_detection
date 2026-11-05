"""
Pydantic Schemas for Industrial Optical Inspection REST API.
Includes strict validation bounds, geometric descriptors, and audit logging.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ThresholdUpdateRequest(BaseModel):
    """Payload for updating operational quality thresholds with strict industrial bounds."""
    anomaly_threshold: Optional[float] = Field(None, ge=1.0, le=50.0, description="PatchCore anomaly distance cutoff")
    anomaly_margin: Optional[float] = Field(None, ge=0.0, le=0.40, description="Uncertainty band margin delta")
    max_acceptable_area_mm2: Optional[float] = Field(None, ge=0.0, le=500.0, description="Maximum allowable cosmetic micro-flaw area")
    critical_area_mm2: Optional[float] = Field(None, ge=0.01, le=10000.0, description="Critical damage area triggering immediate rejection")
    critical_classes: Optional[List[str]] = Field(None, description="Defect categories triggering automatic FAIL")
    min_classifier_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_entropy_threshold: Optional[float] = Field(None, ge=0.0, le=10.0)


class ThresholdResponse(BaseModel):
    """Operational quality threshold configuration."""
    anomaly_threshold: float
    anomaly_margin: float
    max_acceptable_area_mm2: float
    critical_area_mm2: float
    critical_classes: List[str]
    min_classifier_confidence: float
    max_entropy_threshold: float
    config_version: str


class CalibrationUpdateRequest(BaseModel):
    """Payload for updating metric camera scale with validation bounds."""
    mm_per_pixel: float = Field(..., gt=0.0001, le=50.0, description="Physical millimeter length per sensor pixel")
    units: Optional[str] = Field("mm", description="Measurement units")


class CalibrationResponse(BaseModel):
    """Active camera calibration parameters."""
    mm_per_pixel: float
    units: str
    camera_matrix_loaded: bool
    distortion_coefficients_loaded: bool
    config_version: str


class MorphologicalFeatureItem(BaseModel):
    """2D Geometric shape descriptors for an individual segmented defect component."""
    contour_id: int
    area_pixels: int
    physical_area_mm2: float
    perimeter_pixels: float
    physical_perimeter_mm: float
    bounding_box_xywh: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    aspect_ratio: float
    circularity: float
    is_shape_reliable: bool = True
    solidity: float
    compactness: float
    extent: float
    equivalent_diameter_mm: float
    major_axis_length_mm: float = 0.0
    max_feret_diameter_mm: float = 0.0
    morphology_type: str
    spatial_zone: str
    zone_overlaps: Dict[str, float] = Field(default_factory=dict)
    zone_risk_weight: float
    morphology_risk_weight: float
    heuristic_severity_score: float
    adjusted_severity_score: Optional[float] = None


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
    """Authenticated inspection certificate with HMAC-SHA256 and area reconciliation."""
    certificate_id: str
    timestamp_utc: str
    product_domain: str
    config_version: str = "CFG-v1.0.0"
    source_image_hash: str
    decision: str
    severity: str
    routing_action: str
    anomaly_score: float
    primary_defect_type: str
    dominant_morphology: str
    total_defect_area_mm2: float
    component_defect_area_mm2: float = 0.0
    filtered_noise_area_mm2: float = 0.0
    is_area_reconciled: bool = True
    max_single_defect_area_mm2: float
    max_defect_length_mm: float = 0.0
    measurement_mode: str
    num_defect_components: int
    zone_summary: Dict[str, Dict[str, Any]]
    defect_components: List[Dict[str, Any]]
    engineering_rationale: str
    calibration_scale_mm_per_pixel: float
    sha256_integrity_hash: str
    hmac_sha256_signature: str
    sha256_audit_signature: Optional[str] = None


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


class AuditLogEntrySchema(BaseModel):
    """Audit record for configuration changes."""
    timestamp_utc: str
    actor: str
    action: str
    changed_parameters: Dict[str, Any]
    config_version: str


class SystemInfoResponse(BaseModel):
    """System health, hardware device, and active pipeline configuration."""
    service_name: str
    version: str
    config_version: str
    device: str
    models_loaded: Dict[str, bool]
    supported_domains: List[str]
    active_calibration_scale: float
    active_audit_entries_count: int
