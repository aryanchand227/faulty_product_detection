"""
FastAPI Application Entry Point.
Configures middleware, routes, lifespan database initialization, and system health checks.
"""

from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.database.connection import init_db, check_db_health
from backend.app.api.routes import inspection, history, analytics, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initializes database tables and storage directories on startup."""
    init_db()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade automated optical inspection, defect morphology analysis, and historical quality analytics backend.",
    version=settings.VERSION,
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(inspection.router, prefix="/api/v1/inspect", tags=["Inspection"])
app.include_router(history.router, prefix="/api/v1", tags=["History & Audit"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics & KPIs"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Model Registry"])


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    python_version: str
    service: str
    version: str
    message: str


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint to probe backend operational readiness and database connectivity."""
    db_ok = check_db_health()
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        database_connected=db_ok,
        python_version=sys.version.split()[0],
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        message="Industrial Damage Detection API is running"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
