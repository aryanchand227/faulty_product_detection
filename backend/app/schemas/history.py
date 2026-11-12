"""
Pydantic Schemas for Historical Inspection Queries, Quality Analytics,
Batch Inspection, Archival Retention, and Model Registry.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InspectionSummaryItem(BaseModel):
    """Compact summary of an inspection run for paginated table views."""
    id: int
    certificate_id: str
    timestamp_utc: str
    product_domain: str
    decision: str
    severity: str
    routing_action: str
    anomaly_score: float
    primary_defect_type: str
    dominant_morphology: str
    total_defect_area_mm2: float
    max_defect_length_mm: float
    total_latency_ms: float
    early_exit_applied: bool
    num_defect_components: int
    has_images: bool = False
    key_id: Optional[str] = "KEY-2026-V1"
    record_status: str = "ACTIVE"


class PaginatedInspectionsResponse(BaseModel):
    """Paginated collection of historical inspections."""
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[InspectionSummaryItem]


class VerificationResponse(BaseModel):
    """Cryptographic tamper verification status for a stored inspection record."""
    is_valid: bool
    inspection_id: int
    certificate_id: str
    key_id: Optional[str] = "KEY-2026-V1"
    sha256_integrity_match: bool
    hmac_signature_match: bool
    stored_hash: str
    computed_hash: str
    stored_signature: str
    computed_signature: str
    record_status: Optional[str] = "ACTIVE"
    detail: Optional[str] = None


class LatencyPercentilesSchema(BaseModel):
    """Statistical latency percentiles across historical inspection runs."""
    mean_ms: float
    median_p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


class AnalyticsSummaryResponse(BaseModel):
    """Aggregate quality analytics and KPI metrics for factory production monitoring."""
    total_inspections: int
    pass_count: int
    review_count: int
    fail_count: int
    pass_rate_pct: float
    review_rate_pct: float = 0.0
    fail_rate_pct: float = 0.0
    early_exit_count: int
    early_exit_rate_pct: float
    mean_anomaly_score: float
    mean_latency_ms: float
    latency_metrics: Optional[LatencyPercentilesSchema] = None
    defect_type_distribution: Dict[str, int] = Field(default_factory=dict)
    morphology_distribution: Dict[str, int] = Field(default_factory=dict)
    spatial_zone_distribution: Dict[str, int] = Field(default_factory=dict)
    severity_distribution: Dict[str, int] = Field(default_factory=dict)


class BatchInspectionItem(BaseModel):
    """Result of an individual image processed in a batch."""
    filename: str
    status: str
    inspection_id: Optional[int] = None
    certificate_id: Optional[str] = None
    decision: Optional[str] = None
    severity: Optional[str] = None
    routing: Optional[str] = None
    anomaly_score: Optional[float] = None
    total_latency_ms: Optional[float] = None
    error_message: Optional[str] = None


class BatchInspectionResponse(BaseModel):
    """Consolidated summary for multi-image batch inspection."""
    batch_id: str
    total_submitted: int
    successful_inspections: int
    failed_inspections: int
    pass_count: int
    review_count: int
    fail_count: int
    mean_latency_ms: float
    results: List[BatchInspectionItem]


class ArchiveRequest(BaseModel):
    """Request payload to soft archive an inspection record with compliance justification."""
    reason: str = Field(..., min_length=5, max_length=255, description="Auditor justification for status change")
    status: Optional[str] = Field("ARCHIVED", pattern="^(ARCHIVED|LEGAL_HOLD)$", description="Target retention status")


class ArchiveResponse(BaseModel):
    """Confirmation payload for inspection record archival."""
    inspection_id: int
    certificate_id: str
    record_status: str
    archived_at: str
    archived_by: str
    archive_reason: str


class ModelRegistryItem(BaseModel):
    """Metadata for an individual machine learning inspection model."""
    tier: str
    model_name: str
    architecture: str
    checkpoint_source: str
    device: str
    is_loaded: bool
    primary_metrics: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistryResponse(BaseModel):
    """Inspection pipeline model registry and runtime architecture information."""
    service: str
    device: str
    models: List[ModelRegistryItem]
