"""
Database Connection and Session Management for Industrial AOI Service.
Supports SQLite (local development and tests) and PostgreSQL (production).
"""

import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Engine configuration based on dialect
connect_args = {}
engine_kwargs = {}

if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
else:
    # PostgreSQL production pooling parameters
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    **engine_kwargs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a thread-safe database session with automatic teardown."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates all registered database tables if they do not already exist."""
    try:
        from backend.app.database import models  # Ensure all models are imported
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized successfully with URL: {settings.DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise


def check_db_health() -> bool:
    """Pings database connection via lightweight query."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return False

