"""
Quality Analytics and Operational Factory KPI Endpoints.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database import crud
from backend.app.schemas.history import AnalyticsSummaryResponse

router = APIRouter()


@router.get(
    "/summary",
    response_model=AnalyticsSummaryResponse,
    summary="Get Production Quality Analytics"
)
def get_production_analytics(
    domain: Optional[str] = Query(None, description="Filter by product domain: GLASS_BOTTLE or STEEL_SHEET"),
    days: int = Query(7, ge=1, le=365, description="Lookback window in days"),
    db: Session = Depends(get_db),
):
    """
    Computes factory quality statistics over the requested time window:
    pass/review/fail rates, early-exit rate, anomaly score distribution,
    defect category breakdown, and spatial zone vulnerabilities.
    """
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    summary_data = crud.get_analytics_summary(db, domain=domain, start_time=start_time)
    return AnalyticsSummaryResponse(**summary_data)


@router.get(
    "/zone-heatmap",
    summary="Spatial Zone Vulnerability Breakdown"
)
def get_zone_vulnerability_heatmap(
    domain: Optional[str] = Query(None, description="Product domain"),
    db: Session = Depends(get_db),
):
    """
    Analyzes spatial distribution of defects across functional zones,
    identifying critical stress concentrations such as sealing rim chips or edge tears.
    """
    from backend.app.database.models import DefectComponentRecord, InspectionRecord
    from sqlalchemy import func

    q = db.query(
        DefectComponentRecord.spatial_zone,
        func.count(DefectComponentRecord.id).label("defect_count"),
        func.sum(DefectComponentRecord.physical_area_mm2).label("total_area_mm2"),
        func.avg(DefectComponentRecord.heuristic_severity_score).label("mean_severity"),
        func.max(DefectComponentRecord.zone_risk_weight).label("risk_multiplier")
    ).join(InspectionRecord, DefectComponentRecord.inspection_id == InspectionRecord.id)

    if domain:
        q = q.filter(InspectionRecord.product_domain == domain)

    results = q.group_by(DefectComponentRecord.spatial_zone).all()

    items = []
    for row in results:
        items.append({
            "zone": row.spatial_zone,
            "defect_count": row.defect_count,
            "total_area_mm2": round(float(row.total_area_mm2 or 0.0), 3),
            "mean_severity_score": round(float(row.mean_severity or 0.0), 2),
            "risk_multiplier": float(row.risk_multiplier or 1.0)
        })

    return {"zones": items, "count": len(items)}

