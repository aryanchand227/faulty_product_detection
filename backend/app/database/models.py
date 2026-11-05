"""
SQLAlchemy Relational ORM Models for Industrial Quality Inspection System.
Defines persistent schema for inspection runs, defect components, and audit trails.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Index
)
from sqlalchemy.orm import relationship

from backend.app.database.connection import Base


def utc_now() -> datetime:
    """Returns timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class InspectionRecord(Base):
    """
    Primary persistent record for an optical inspection event.
    Stores complete operational triage verdicts, physical metrics, and cryptographic hashes.
    """
    __tablename__ = "inspection_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    certificate_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp_utc = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)
    timestamp_iso = Column(String(64), nullable=True)
    product_domain = Column(String(32), index=True, nullable=False)
    
    # Operational Triage Verdict
    decision = Column(String(16), index=True, nullable=False)  # PASS, REVIEW, FAIL
    severity = Column(String(16), index=True, nullable=False)  # NONE, LOW, MEDIUM, CRITICAL
    routing_action = Column(String(32), nullable=False)        # LINE_CONTINUE, MANUAL_INSPECTION_BAY, AUTOMATIC_REJECT_BIN
    
    # Anomaly and Defect Diagnostics
    anomaly_score = Column(Float, nullable=False)
    primary_defect_type = Column(String(64), index=True, nullable=False)
    dominant_morphology = Column(String(64), nullable=False)
    
    # Metric Physical Damage Measures & Area Reconciliation
    total_defect_area_mm2 = Column(Float, nullable=False, default=0.0)
    component_defect_area_mm2 = Column(Float, nullable=False, default=0.0)
    filtered_noise_area_mm2 = Column(Float, nullable=False, default=0.0)
    is_area_reconciled = Column(Boolean, nullable=False, default=True)
    max_single_defect_area_mm2 = Column(Float, nullable=False, default=0.0)
    max_defect_length_mm = Column(Float, nullable=False, default=0.0)
    measurement_mode = Column(String(32), nullable=False, default="projected_sensor_plane")
    calibration_scale_mm_per_pixel = Column(Float, nullable=False, default=0.15)
    
    # Engineering Context
    engineering_rationale = Column(Text, nullable=True)
    config_version = Column(String(32), nullable=False, default="CFG-v1.0.0")
    source_image_hash = Column(String(64), index=True, nullable=False)
    
    # Cryptographic Authentication Signatures
    sha256_integrity_hash = Column(String(64), nullable=False)
    hmac_sha256_signature = Column(String(64), nullable=False)
    
    # Operational Pipeline Telemetry
    total_latency_ms = Column(Float, nullable=False)
    early_exit_applied = Column(Boolean, nullable=False, default=False)
    
    # Media Storage Paths
    original_image_path = Column(String(512), nullable=True)
    annotated_image_path = Column(String(512), nullable=True)
    
    # Zone Summary Breakdown (stored as serialized JSON)
    zone_summary_json = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relational Child: Individual Segmented Defect Components
    components = relationship(
        "DefectComponentRecord",
        back_populates="inspection",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    __table_args__ = (
        Index("idx_inspection_domain_decision", "product_domain", "decision"),
        Index("idx_inspection_created_at", "created_at"),
    )

    def to_dict(self, include_components: bool = True) -> Dict[str, Any]:
        """Serializes inspection record to python dictionary."""
        data = {
            "id": self.id,
            "certificate_id": self.certificate_id,
            "timestamp_utc": self.timestamp_iso or (self.timestamp_utc.isoformat() if self.timestamp_utc else None),
            "product_domain": self.product_domain,
            "decision": self.decision,
            "severity": self.severity,
            "routing_action": self.routing_action,
            "anomaly_score": round(self.anomaly_score, 4),
            "primary_defect_type": self.primary_defect_type,
            "dominant_morphology": self.dominant_morphology,
            "total_defect_area_mm2": round(self.total_defect_area_mm2, 3),
            "component_defect_area_mm2": round(self.component_defect_area_mm2, 3),
            "filtered_noise_area_mm2": round(self.filtered_noise_area_mm2, 3),
            "is_area_reconciled": self.is_area_reconciled,
            "max_single_defect_area_mm2": round(self.max_single_defect_area_mm2, 3),
            "max_defect_length_mm": round(self.max_defect_length_mm, 3),
            "measurement_mode": self.measurement_mode,
            "calibration_scale_mm_per_pixel": round(self.calibration_scale_mm_per_pixel, 5),
            "engineering_rationale": self.engineering_rationale,
            "config_version": self.config_version,
            "source_image_hash": self.source_image_hash,
            "sha256_integrity_hash": self.sha256_integrity_hash,
            "hmac_sha256_signature": self.hmac_sha256_signature,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "early_exit_applied": self.early_exit_applied,
            "original_image_path": self.original_image_path,
            "annotated_image_path": self.annotated_image_path,
            "zone_summary": json.loads(self.zone_summary_json) if self.zone_summary_json else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_components:
            data["defect_components"] = [comp.to_dict() for comp in self.components]
            data["num_defect_components"] = len(self.components)
        return data


class DefectComponentRecord(Base):
    """
    Relational record storing morphological parameters for an individual segmented defect contour.
    """
    __tablename__ = "defect_component_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    inspection_id = Column(
        Integer,
        ForeignKey("inspection_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    contour_id = Column(Integer, nullable=False)
    area_pixels = Column(Integer, nullable=False)
    physical_area_mm2 = Column(Float, nullable=False)
    perimeter_pixels = Column(Float, nullable=False)
    physical_perimeter_mm = Column(Float, nullable=False)
    
    # Coordinates & Centroid
    bounding_box_json = Column(String(64), nullable=False)  # [bx, by, bw, bh]
    centroid_xy_json = Column(String(64), nullable=False)   # [cx, cy]
    
    # Geometric Descriptors
    aspect_ratio = Column(Float, nullable=False)
    circularity = Column(Float, nullable=False)
    is_shape_reliable = Column(Boolean, nullable=False, default=True)
    solidity = Column(Float, nullable=False)
    compactness = Column(Float, nullable=False)
    extent = Column(Float, nullable=False)
    equivalent_diameter_mm = Column(Float, nullable=False)
    major_axis_length_mm = Column(Float, nullable=False, default=0.0)
    max_feret_diameter_mm = Column(Float, nullable=False, default=0.0)
    
    # Classification & Functional Risk
    morphology_type = Column(String(48), index=True, nullable=False)
    spatial_zone = Column(String(32), index=True, nullable=False)
    zone_overlaps_json = Column(Text, nullable=False, default="{}")
    zone_risk_weight = Column(Float, nullable=False, default=1.0)
    morphology_risk_weight = Column(Float, nullable=False, default=1.0)
    heuristic_severity_score = Column(Float, nullable=False, default=0.0)

    # Relationship to parent
    inspection = relationship("InspectionRecord", back_populates="components")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes component record to python dictionary."""
        return {
            "id": self.id,
            "contour_id": self.contour_id,
            "area_pixels": self.area_pixels,
            "physical_area_mm2": round(self.physical_area_mm2, 3),
            "perimeter_pixels": round(self.perimeter_pixels, 1),
            "physical_perimeter_mm": round(self.physical_perimeter_mm, 2),
            "bounding_box_xywh": json.loads(self.bounding_box_json) if self.bounding_box_json else [],
            "centroid_xy": json.loads(self.centroid_xy_json) if self.centroid_xy_json else [],
            "aspect_ratio": round(self.aspect_ratio, 2),
            "circularity": round(self.circularity, 3),
            "is_shape_reliable": self.is_shape_reliable,
            "solidity": round(self.solidity, 3),
            "compactness": round(self.compactness, 2),
            "extent": round(self.extent, 3),
            "equivalent_diameter_mm": round(self.equivalent_diameter_mm, 2),
            "major_axis_length_mm": round(self.major_axis_length_mm, 2),
            "max_feret_diameter_mm": round(self.max_feret_diameter_mm, 2),
            "morphology_type": self.morphology_type,
            "spatial_zone": self.spatial_zone,
            "zone_overlaps": json.loads(self.zone_overlaps_json) if self.zone_overlaps_json else {},
            "zone_risk_weight": self.zone_risk_weight,
            "morphology_risk_weight": self.morphology_risk_weight,
            "heuristic_severity_score": round(self.heuristic_severity_score, 2),
            "adjusted_severity_score": round(self.heuristic_severity_score, 2),
        }

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Returns exact dictionary format matching MorphologicalFeatures.to_dict() for cryptographic canonical hashing."""
        return {
            "contour_id": self.contour_id,
            "area_pixels": self.area_pixels,
            "physical_area_mm2": round(self.physical_area_mm2, 3),
            "perimeter_pixels": round(self.perimeter_pixels, 1),
            "physical_perimeter_mm": round(self.physical_perimeter_mm, 2),
            "bounding_box_xywh": tuple(json.loads(self.bounding_box_json)) if self.bounding_box_json else (),
            "centroid_xy": tuple(json.loads(self.centroid_xy_json)) if self.centroid_xy_json else (),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "circularity": round(self.circularity, 3),
            "is_shape_reliable": self.is_shape_reliable,
            "solidity": round(self.solidity, 3),
            "compactness": round(self.compactness, 2),
            "extent": round(self.extent, 3),
            "equivalent_diameter_mm": round(self.equivalent_diameter_mm, 2),
            "major_axis_length_mm": round(self.major_axis_length_mm, 2),
            "max_feret_diameter_mm": round(self.max_feret_diameter_mm, 2),
            "morphology_type": self.morphology_type,
            "spatial_zone": self.spatial_zone,
            "zone_overlaps": json.loads(self.zone_overlaps_json) if self.zone_overlaps_json else {},
            "zone_risk_weight": self.zone_risk_weight,
            "morphology_risk_weight": self.morphology_risk_weight,
            "heuristic_severity_score": round(self.heuristic_severity_score, 2),
            "adjusted_severity_score": round(self.heuristic_severity_score, 2),
        }


class AuditLogRecord(Base):
    """
    Persistent audit trail for all operational configuration updates.
    """
    __tablename__ = "audit_log_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp_utc = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)
    actor = Column(String(64), nullable=False)
    action = Column(String(64), nullable=False)
    changed_parameters_json = Column(Text, nullable=False)
    config_version = Column(String(32), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
            "actor": self.actor,
            "action": self.action,
            "changed_parameters": json.loads(self.changed_parameters_json) if self.changed_parameters_json else {},
            "config_version": self.config_version,
        }
