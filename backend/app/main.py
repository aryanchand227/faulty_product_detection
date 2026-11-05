"""
FastAPI Application Entry Point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

app = FastAPI(
    title="Industrial Quality Inspection API",
    description="Automated defect detection, segmentation, and anomaly detection backend.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.app.api.routes import inspection
app.include_router(inspection.router, prefix="/api/v1/inspect", tags=["Inspection"])

class HealthResponse(BaseModel):
    status: str
    python_version: str
    message: str

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint to verify backend operational readiness."""
    return HealthResponse(
        status="healthy",
        python_version=sys.version.split()[0],
        message="Industrial Damage Detection API is running"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)

