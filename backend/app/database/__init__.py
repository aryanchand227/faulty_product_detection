"""
Database package initialization for Industrial Damage Detection.
Exports ORM models, session dependencies, and engine initializers.
"""

from backend.app.database.connection import (
    Base,
    engine,
    SessionLocal,
    get_db,
    init_db,
    check_db_health,
)
from backend.app.database.models import (
    InspectionRecord,
    DefectComponentRecord,
    AuditLogRecord,
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "check_db_health",
    "InspectionRecord",
    "DefectComponentRecord",
    "AuditLogRecord",
]

