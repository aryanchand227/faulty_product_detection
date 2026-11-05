"""
Core configuration settings for Industrial Inspection Backend Microservice.
Supports environment variable overrides and sensible defaults for local development.
"""

from pathlib import Path
import os
from typing import List


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

    # Security & Administration
    ADMIN_API_KEY: str = os.getenv("AOI_ADMIN_API_KEY", "aoi-admin-secret-key-2026")
    DEFAULT_HMAC_SECRET: bytes = os.getenv(
        "AOI_HMAC_SECRET_KEY", "aoi-production-secret-key-2026-compliance"
    ).encode("utf-8")

    # File uploads & storage
    STORAGE_DIR: Path = STORAGE_DIR
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    MAX_BATCH_FILES: int = 32

    # Inference defaults
    DEVICE: str = os.getenv("INFERENCE_DEVICE", "cpu")
    HF_REPO_ID: str = "TAC13/industrial-damage-patchcore-bottle"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()

