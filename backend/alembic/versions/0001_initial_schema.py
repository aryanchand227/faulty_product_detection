"""Initial schema for industrial inspection records, defect components, and audit log

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-11-06 01:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Inspection Records Table
    op.create_table(
        'inspection_records',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('certificate_id', sa.String(length=64), nullable=False),
        sa.Column('timestamp_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('timestamp_iso', sa.String(length=64), nullable=True),
        sa.Column('product_domain', sa.String(length=32), nullable=False),
        sa.Column('decision', sa.String(length=16), nullable=False),
        sa.Column('severity', sa.String(length=16), nullable=False),
        sa.Column('routing_action', sa.String(length=32), nullable=False),
        sa.Column('anomaly_score', sa.Float(), nullable=False),
        sa.Column('primary_defect_type', sa.String(length=64), nullable=False),
        sa.Column('dominant_morphology', sa.String(length=64), nullable=False),
        sa.Column('total_defect_area_mm2', sa.Float(), nullable=False),
        sa.Column('component_defect_area_mm2', sa.Float(), nullable=False),
        sa.Column('filtered_noise_area_mm2', sa.Float(), nullable=False),
        sa.Column('is_area_reconciled', sa.Boolean(), nullable=False),
        sa.Column('max_single_defect_area_mm2', sa.Float(), nullable=False),
        sa.Column('max_defect_length_mm', sa.Float(), nullable=False),
        sa.Column('measurement_mode', sa.String(length=32), nullable=False),
        sa.Column('calibration_scale_mm_per_pixel', sa.Float(), nullable=False),
        sa.Column('engineering_rationale', sa.Text(), nullable=True),
        sa.Column('config_version', sa.String(length=32), nullable=False),
        sa.Column('source_image_hash', sa.String(length=64), nullable=False),
        sa.Column('sha256_integrity_hash', sa.String(length=64), nullable=False),
        sa.Column('hmac_sha256_signature', sa.String(length=64), nullable=False),
        sa.Column('total_latency_ms', sa.Float(), nullable=False),
        sa.Column('early_exit_applied', sa.Boolean(), nullable=False),
        sa.Column('original_image_path', sa.String(length=512), nullable=True),
        sa.Column('annotated_image_path', sa.String(length=512), nullable=True),
        sa.Column('zone_summary_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_inspection_records_id', 'inspection_records', ['id'], unique=False)
    op.create_index('ix_inspection_records_certificate_id', 'inspection_records', ['certificate_id'], unique=True)
    op.create_index('ix_inspection_records_timestamp_utc', 'inspection_records', ['timestamp_utc'], unique=False)
    op.create_index('ix_inspection_records_product_domain', 'inspection_records', ['product_domain'], unique=False)
    op.create_index('ix_inspection_records_decision', 'inspection_records', ['decision'], unique=False)
    op.create_index('ix_inspection_records_severity', 'inspection_records', ['severity'], unique=False)
    op.create_index('ix_inspection_records_primary_defect_type', 'inspection_records', ['primary_defect_type'], unique=False)
    op.create_index('ix_inspection_records_source_image_hash', 'inspection_records', ['source_image_hash'], unique=False)
    op.create_index('idx_inspection_domain_decision', 'inspection_records', ['product_domain', 'decision'], unique=False)
    op.create_index('idx_inspection_created_at', 'inspection_records', ['created_at'], unique=False)

    # 2. Defect Component Records Table
    op.create_table(
        'defect_component_records',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('inspection_id', sa.Integer(), nullable=False),
        sa.Column('contour_id', sa.Integer(), nullable=False),
        sa.Column('area_pixels', sa.Integer(), nullable=False),
        sa.Column('physical_area_mm2', sa.Float(), nullable=False),
        sa.Column('perimeter_pixels', sa.Float(), nullable=False),
        sa.Column('physical_perimeter_mm', sa.Float(), nullable=False),
        sa.Column('bounding_box_json', sa.String(length=64), nullable=False),
        sa.Column('centroid_xy_json', sa.String(length=64), nullable=False),
        sa.Column('aspect_ratio', sa.Float(), nullable=False),
        sa.Column('circularity', sa.Float(), nullable=False),
        sa.Column('is_shape_reliable', sa.Boolean(), nullable=False),
        sa.Column('solidity', sa.Float(), nullable=False),
        sa.Column('compactness', sa.Float(), nullable=False),
        sa.Column('extent', sa.Float(), nullable=False),
        sa.Column('equivalent_diameter_mm', sa.Float(), nullable=False),
        sa.Column('major_axis_length_mm', sa.Float(), nullable=False),
        sa.Column('max_feret_diameter_mm', sa.Float(), nullable=False),
        sa.Column('morphology_type', sa.String(length=48), nullable=False),
        sa.Column('spatial_zone', sa.String(length=32), nullable=False),
        sa.Column('zone_overlaps_json', sa.Text(), nullable=False),
        sa.Column('zone_risk_weight', sa.Float(), nullable=False),
        sa.Column('morphology_risk_weight', sa.Float(), nullable=False),
        sa.Column('heuristic_severity_score', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['inspection_id'], ['inspection_records.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_defect_component_records_id', 'defect_component_records', ['id'], unique=False)
    op.create_index('ix_defect_component_records_inspection_id', 'defect_component_records', ['inspection_id'], unique=False)
    op.create_index('ix_defect_component_records_morphology_type', 'defect_component_records', ['morphology_type'], unique=False)
    op.create_index('ix_defect_component_records_spatial_zone', 'defect_component_records', ['spatial_zone'], unique=False)

    # 3. Audit Log Records Table
    op.create_table(
        'audit_log_records',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('actor', sa.String(length=64), nullable=False),
        sa.Column('action', sa.String(length=64), nullable=False),
        sa.Column('changed_parameters_json', sa.Text(), nullable=False),
        sa.Column('config_version', sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_log_records_id', 'audit_log_records', ['id'], unique=False)
    op.create_index('ix_audit_log_records_timestamp_utc', 'audit_log_records', ['timestamp_utc'], unique=False)


def downgrade() -> None:
    op.drop_table('audit_log_records')
    op.drop_table('defect_component_records')
    op.drop_table('inspection_records')

