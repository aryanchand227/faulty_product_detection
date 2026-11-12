"""
Core configuration settings for Industrial Inspection Backend Microservice.
Supports environment variable overrides and sensible defaults for local development.
"""

from pathlib import Path
import os
import json
from typing import Dict, List


# Base directories
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = DATA_DIR / "inspection_media"

# Ensure data storage directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class Settings:
    PROJECT_NAME: str = "Industrial Optical Inspection Microservice"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database Configuration: Defaults to SQLite for seamless local execution;
    # Can be overridden via DATABASE_URL to PostgreSQL (e.g., postgresql://user:pass@localhost:5432/inspection_db)
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'inspections.db'}")

    # Security & Role-Based API Keys (Stored outside database via environment variables)
    ADMIN_API_KEY: str = os.getenv("AOI_ADMIN_API_KEY", "aoi-admin-secret-key-2026")
    AUDITOR_API_KEY: str = os.getenv("AOI_AUDITOR_API_KEY", "aoi-auditor-secret-key-2026")
    OPERATOR_API_KEY: str = os.getenv("AOI_OPERATOR_API_KEY", "aoi-operator-secret-key-2026")

    # Cryptographic HMAC Key Management & Lifecycle
    # Secret keys are NEVER stored in the database.
    ACTIVE_KEY_ID: str = os.getenv("AOI_ACTIVE_KEY_ID", "KEY-2026-V1")
    DEFAULT_HMAC_SECRET: bytes = os.getenv(
        "AOI_HMAC_SECRET_KEY", "aoi-production-secret-key-2026-compliance"
    ).encode("utf-8")

    # Key Registry for Historical Rotation: Maps key_id -> secret bytes
    # Allows historical records signed with previous keys to remain verifiable
    KEY_REGISTRY: Dict[str, bytes] = {
        "KEY-2026-V1": DEFAULT_HMAC_SECRET,
        "KEY-2025-V0": b"aoi-legacy-secret-key-2025-baseline",
    }

    # Load additional keys from JSON env var if provided: e.g. '{"KEY-2027-V1": "secret2027"}'
    if os.getenv("AOI_KEY_REGISTRY_JSON"):
        try:
            extra_keys = json.loads(os.getenv("AOI_KEY_REGISTRY_JSON", "{}"))
            for k, v in extra_keys.items():
                KEY_REGISTRY[k] = v.encode("utf-8") if isinstance(v, str) else bytes(v)
        except Exception:
            pass

    # Model and Engineering Provenance Metadata
    MODEL_ARTIFACT_ID: str = os.getenv("AOI_MODEL_ARTIFACT_ID", "TAC13/industrial-damage-patchcore-bottle@main")
    BACKBONE_ARCHITECTURE: str = "Wide-ResNet-50-2"
    SOFTWARE_VERSION: str = "1.0.0"
    CALIBRATION_PROFILE_ID: str = os.getenv("AOI_CALIBRATION_PROFILE_ID", "CALIB-BOTTLE-0.15MM-V1")
    DECISION_POLICY_VERSION: str = os.getenv("AOI_DECISION_POLICY_VERSION", "POL-v1.0.0")

    # Audit & Deletion Policy
    # In production, hard deletion is prohibited to preserve permanent compliance records (FDA 21 CFR Part 11 / ISO 9001).
    ALLOW_HARD_DELETE: bool = os.getenv("AOI_ALLOW_HARD_DELETE", "false").lower() in ("true", "1")

    # Concurrency and Upload Resource Bounds
    STORAGE_DIR: Path = STORAGE_DIR
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    MAX_BATCH_FILES: int = 32
    BATCH_CONCURRENCY_LIMIT: int = int(os.getenv("AOI_BATCH_CONCURRENCY_LIMIT", "4"))

    # Inference defaults
    DEVICE: str = os.getenv("INFERENCE_DEVICE", "cpu")
    HF_REPO_ID: str = "TAC13/industrial-damage-patchcore-bottle"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()
