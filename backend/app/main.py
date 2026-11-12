"""
FastAPI Application Entry Point.
Configures middleware, routes, lifespan database initialization, system health checks,
and serves the industrial inspection frontend dashboard.
"""

from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.app.core.config import settings, PROJECT_ROOT
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


# Serve Frontend Web Dashboard
frontend_path = PROJECT_ROOT / "frontend"
if frontend_path.exists() and (frontend_path / "index.html").exists():
    app.mount("/dashboard", StaticFiles(directory=str(frontend_path), html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/dashboard/")
