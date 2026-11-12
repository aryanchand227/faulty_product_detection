"""Soft delete retention, cryptographic key identifier, and provenance metadata

Revision ID: 0002_provenance_and_retention
Revises: 0001_initial_schema
Create Date: 2026-11-06 14:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002_provenance_and_retention'
down_revision: Union[str, None] = '0001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update inspection_records with provenance, key_id, relative URIs, and soft-delete retention
    with op.batch_alter_table('inspection_records') as batch_op:
        batch_op.add_column(sa.Column('key_id', sa.String(length=32), server_default='KEY-2026-V1', nullable=False))
        batch_op.add_column(sa.Column('model_artifact_id', sa.String(length=128), server_default='TAC13/industrial-damage-patchcore-bottle@main', nullable=False))
        batch_op.add_column(sa.Column('software_version', sa.String(length=32), server_default='1.0.0', nullable=False))
        batch_op.add_column(sa.Column('calibration_profile_id', sa.String(length=64), server_default='CALIB-BOTTLE-0.15MM-V1', nullable=False))
        batch_op.add_column(sa.Column('decision_policy_version', sa.String(length=32), server_default='POL-v1.0.0', nullable=False))
        batch_op.add_column(sa.Column('record_status', sa.String(length=16), server_default='ACTIVE', nullable=False))
        batch_op.add_column(sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('archived_by', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('archive_reason', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('original_image_uri', sa.String(length=256), nullable=True))
        batch_op.add_column(sa.Column('annotated_image_uri', sa.String(length=256), nullable=True))

        batch_op.create_index('ix_inspection_records_record_status', ['record_status'], unique=False)
        batch_op.create_check_constraint('chk_inspection_anomaly_non_negative', 'anomaly_score >= 0.0')
        batch_op.create_check_constraint('chk_inspection_area_non_negative', 'total_defect_area_mm2 >= 0.0')
        batch_op.create_check_constraint('chk_inspection_latency_non_negative', 'total_latency_ms >= 0.0')
        batch_op.create_check_constraint('chk_inspection_scale_positive', 'calibration_scale_mm_per_pixel > 0.0')
        batch_op.create_check_constraint('chk_inspection_valid_decision', "decision IN ('PASS', 'REVIEW', 'FAIL')")
        batch_op.create_check_constraint('chk_inspection_valid_status', "record_status IN ('ACTIVE', 'ARCHIVED', 'LEGAL_HOLD')")

    # 2. Update defect_component_records with geometric bounds constraints
    with op.batch_alter_table('defect_component_records') as batch_op:
        batch_op.create_check_constraint('chk_component_area_pixels_non_negative', 'area_pixels >= 0')
        batch_op.create_check_constraint('chk_component_area_non_negative', 'physical_area_mm2 >= 0.0')
        batch_op.create_check_constraint('chk_component_circularity_range', 'circularity >= 0.0 AND circularity <= 1.0')
        batch_op.create_check_constraint('chk_component_solidity_range', 'solidity >= 0.0 AND solidity <= 1.0')


def downgrade() -> None:
    with op.batch_alter_table('defect_component_records') as batch_op:
        batch_op.drop_constraint('chk_component_solidity_range', type_='check')
        batch_op.drop_constraint('chk_component_circularity_range', type_='check')
        batch_op.drop_constraint('chk_component_area_non_negative', type_='check')
        batch_op.drop_constraint('chk_component_area_pixels_non_negative', type_='check')

    with op.batch_alter_table('inspection_records') as batch_op:
        batch_op.drop_constraint('chk_inspection_valid_status', type_='check')
        batch_op.drop_constraint('chk_inspection_valid_decision', type_='check')
        batch_op.drop_constraint('chk_inspection_scale_positive', type_='check')
        batch_op.drop_constraint('chk_inspection_latency_non_negative', type_='check')
        batch_op.drop_constraint('chk_inspection_area_non_negative', type_='check')
        batch_op.drop_constraint('chk_inspection_anomaly_non_negative', type_='check')
        batch_op.drop_index('ix_inspection_records_record_status')
        batch_op.drop_column('annotated_image_uri')
        batch_op.drop_column('original_image_uri')
        batch_op.drop_column('archive_reason')
        batch_op.drop_column('archived_by')
        batch_op.drop_column('archived_at')
        batch_op.drop_column('record_status')
        batch_op.drop_column('decision_policy_version')
        batch_op.drop_column('calibration_profile_id')
        batch_op.drop_column('software_version')
        batch_op.drop_column('model_artifact_id')
        batch_op.drop_column('key_id')

