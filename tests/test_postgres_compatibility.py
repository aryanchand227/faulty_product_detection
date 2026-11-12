"""
PostgreSQL Parity & DDL Compilation Compatibility Test Suite.
Validates that all SQLAlchemy relational models, column types, foreign keys,
indexes, and CheckConstraints compile without errors under PostgreSQL dialect.
"""

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable, CreateIndex
from backend.app.database.models import (
    InspectionRecord,
    DefectComponentRecord,
    AuditLogRecord,
)


def test_postgresql_ddl_compilation_inspection_records():
    """Validates PostgreSQL DDL compilation for primary inspection records table."""
    dialect = postgresql.dialect()
    ddl = str(CreateTable(InspectionRecord.__table__).compile(dialect=dialect))

    assert "CREATE TABLE inspection_records" in ddl
    assert "SERIAL NOT NULL" in ddl or "id INTEGER NOT NULL" in ddl
    assert "TIMESTAMP WITH TIME ZONE" in ddl
    assert "key_id VARCHAR(32)" in ddl
    assert "model_artifact_id VARCHAR(128)" in ddl
    assert "record_status VARCHAR(16)" in ddl
    assert "CONSTRAINT chk_inspection_anomaly_non_negative CHECK (anomaly_score >= 0.0)" in ddl
    assert "CONSTRAINT chk_inspection_area_non_negative CHECK (total_defect_area_mm2 >= 0.0)" in ddl
    assert "CONSTRAINT chk_inspection_scale_positive CHECK (calibration_scale_mm_per_pixel > 0.0)" in ddl
    assert "CONSTRAINT chk_inspection_valid_decision CHECK (decision IN ('PASS', 'REVIEW', 'FAIL'))" in ddl
    assert "CONSTRAINT chk_inspection_valid_status CHECK (record_status IN ('ACTIVE', 'ARCHIVED', 'LEGAL_HOLD'))" in ddl


def test_postgresql_ddl_compilation_defect_components():
    """Validates PostgreSQL DDL compilation for child defect components table."""
    dialect = postgresql.dialect()
    ddl = str(CreateTable(DefectComponentRecord.__table__).compile(dialect=dialect))

    assert "CREATE TABLE defect_component_records" in ddl
    assert "FOREIGN KEY(inspection_id) REFERENCES inspection_records (id) ON DELETE CASCADE" in ddl
    assert "CONSTRAINT chk_component_area_pixels_non_negative CHECK (area_pixels >= 0)" in ddl
    assert "CONSTRAINT chk_component_circularity_range CHECK (circularity >= 0.0 AND circularity <= 1.0)" in ddl
    assert "CONSTRAINT chk_component_solidity_range CHECK (solidity >= 0.0 AND solidity <= 1.0)" in ddl


def test_postgresql_ddl_compilation_audit_log():
    """Validates PostgreSQL DDL compilation for audit trail table."""
    dialect = postgresql.dialect()
    ddl = str(CreateTable(AuditLogRecord.__table__).compile(dialect=dialect))

    assert "CREATE TABLE audit_log_records" in ddl
    assert "actor VARCHAR(64) NOT NULL" in ddl
    assert "action VARCHAR(64) NOT NULL" in ddl
    assert "changed_parameters_json TEXT NOT NULL" in ddl


def test_postgresql_index_compilation():
    """Validates PostgreSQL index syntax compilation."""
    dialect = postgresql.dialect()
    for index in InspectionRecord.__table__.indexes:
        compiled_idx = str(CreateIndex(index).compile(dialect=dialect))
        assert "CREATE INDEX" in compiled_idx or "CREATE UNIQUE INDEX" in compiled_idx
