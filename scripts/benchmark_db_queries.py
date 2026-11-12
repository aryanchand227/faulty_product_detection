"""
Comprehensive Database Query and Cryptographic Audit Performance Benchmark.
Populates 1,000 synthetic inspection records and measures query latency percentiles
(p50, p95, p99) and throughput (QPS) across indexing, filtering, analytics, and HMAC verification.
"""

from datetime import datetime, timezone, timedelta
import json
import logging
import sys
from pathlib import Path
import random
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.database.connection import Base
from backend.app.database import crud
from backend.app.database.models import InspectionRecord, DefectComponentRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("db_benchmark")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_benchmark_dataset(session, num_records=1000):
    """Inserts a representative distribution of 1,000 inspection records."""
    logger.info(f"Generating and persisting {num_records} benchmark inspection records...")
    domains = ["GLASS_BOTTLE", "STEEL_SHEET"]
    decisions = [("PASS", "NONE", "LINE_CONTINUE", 0.70), 
                 ("REVIEW", "LOW", "MANUAL_INSPECTION_BAY", 0.15), 
                 ("FAIL", "CRITICAL", "AUTOMATIC_REJECT_BIN", 0.15)]
    zones = ["SEALING_RIM", "NECK", "BODY_SIDEWALL", "BASE_HEEL"]
    morphologies = ["GEOMETRIC_ELONGATED_SLIT", "GEOMETRIC_CIRCULAR_PIT", "GEOMETRIC_DIFFUSE_CLUSTER"]

    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    records_to_insert = []

    for i in range(num_records):
        cert_id = f"CERT-BENCH-{i:05d}"
        dom = random.choice(domains)
        r_val = random.random()
        cum = 0.0
        dec, sev, rout = "PASS", "NONE", "LINE_CONTINUE"
        for d, s, r, prob in decisions:
            cum += prob
            if r_val <= cum:
                dec, sev, rout = d, s, r
                break

        anomaly_score = random.uniform(5.0, 14.0) if dec == "PASS" else random.uniform(15.5, 35.0)
        total_area = 0.0 if dec == "PASS" else random.uniform(5.0, 150.0)
        ts = base_time + timedelta(minutes=i * 40)

        # Synthetic payload
        payload = {
            "product_domain": dom,
            "total_latency_ms": random.uniform(22.0, 55.0),
            "early_exit_applied": (dec == "PASS"),
            "verdict": {
                "decision": dec,
                "severity": sev,
                "routing": rout,
                "anomaly_score": round(anomaly_score, 4),
                "primary_defect_type": "nominal" if dec == "PASS" else "glass_damage",
                "physical_defect_area_mm2": round(total_area, 3),
                "engineering_rationale": "Automated benchmark entry"
            },
            "physical_metrics": {
                "measurement_mode": "projected_sensor_plane",
                "mm_per_pixel": 0.15
            },
            "certificate": {
                "certificate_id": cert_id,
                "timestamp_utc": ts.isoformat(),
                "dominant_morphology": "NONE" if dec == "PASS" else random.choice(morphologies),
                "total_defect_area_mm2": round(total_area, 3),
                "component_defect_area_mm2": round(total_area * 0.95, 3),
                "filtered_noise_area_mm2": round(total_area * 0.05, 3),
                "is_area_reconciled": True,
                "max_single_defect_area_mm2": round(total_area * 0.8, 3),
                "max_defect_length_mm": round(np.sqrt(total_area) * 2.5, 3),
                "config_version": "CFG-v1.0.0",
                "key_id": "KEY-2026-V1",
                "model_artifact_id": "TAC13/industrial-damage-patchcore-bottle@main",
                "software_version": "1.0.0",
                "calibration_profile_id": "CALIB-BOTTLE-0.15MM-V1",
                "decision_policy_version": "POL-v1.0.0",
                "source_image_hash": f"hash_{i:08x}",
                "zone_summary": {random.choice(zones): {"count": 1, "total_area_mm2": round(total_area, 3)}} if total_area > 0 else {}
            },
            "morphological_features": [
                {
                    "contour_id": 1,
                    "area_pixels": int(total_area / (0.15 ** 2)),
                    "physical_area_mm2": round(total_area * 0.95, 3),
                    "perimeter_pixels": 150.0,
                    "physical_perimeter_mm": 22.5,
                    "bounding_box_xywh": [100, 100, 50, 50],
                    "centroid_xy": [125.0, 125.0],
                    "aspect_ratio": 1.2,
                    "circularity": 0.75,
                    "is_shape_reliable": True,
                    "solidity": 0.90,
                    "compactness": 15.0,
                    "extent": 0.70,
                    "equivalent_diameter_mm": round(np.sqrt(total_area), 2),
                    "major_axis_length_mm": round(np.sqrt(total_area) * 1.5, 2),
                    "max_feret_diameter_mm": round(np.sqrt(total_area) * 1.6, 2),
                    "morphology_type": random.choice(morphologies),
                    "spatial_zone": random.choice(zones),
                    "zone_overlaps": {random.choice(zones): 1.0},
                    "zone_risk_weight": 2.0,
                    "morphology_risk_weight": 1.8,
                    "heuristic_severity_score": round(total_area * 2.0, 2)
                }
            ] if total_area > 0 else []
        }

        # Generate authentic HMAC
        from src.inspection.defect_analysis import InspectionReportGenerator
        from src.inspection.decision_engine import InspectionVerdict, InspectionDecision, SeverityLevel, RoutingAction
        from src.inspection.unified_pipeline import UnifiedInspectionResult

        verdict_obj = InspectionVerdict(
            decision=InspectionDecision(dec),
            severity=SeverityLevel(sev),
            routing=RoutingAction(rout),
            primary_defect_type="nominal" if dec == "PASS" else "glass_damage",
            confidence=0.99,
            physical_defect_area_mm2=round(total_area, 3),
            anomaly_score=round(anomaly_score, 4),
            is_anomalous=(dec != "PASS"),
            engineering_rationale="Automated benchmark entry"
        )
        res_obj = UnifiedInspectionResult(
            verdict=verdict_obj,
            tier1_anomaly=None,
            tier2_classification=None,
            tier3_detection=None,
            tier3_segmentation=None,
            physical_metrics={"measurement_mode": "projected_sensor_plane", "scale_mm_per_pixel": 0.15},
            total_latency_ms=30.0,
            early_exit_applied=(dec == "PASS"),
            product_domain=dom
        )
        cert_obj = InspectionReportGenerator.generate_certificate(
            inspection_result=res_obj,
            morphological_features=[],
            config_version="CFG-v1.0.0",
            certificate_prefix="CERT-BENCH",
            hmac_secret=settings.DEFAULT_HMAC_SECRET,
            key_id="KEY-2026-V1"
        )

        payload["certificate"]["sha256_integrity_hash"] = cert_obj.sha256_integrity_hash
        payload["certificate"]["hmac_sha256_signature"] = cert_obj.hmac_sha256_signature
        payload["certificate"]["certificate_id"] = cert_obj.certificate_id

        crud.create_inspection_record(session, payload)

    logger.info(f"Successfully inserted {num_records} records.")


def run_benchmarks(session, iterations=100):
    """Executes timed query benchmarks and returns percentile stats."""
    logger.info(f"Running query benchmarks ({iterations} timed iterations per pattern)...")

    # Fetch existing IDs for random queries
    all_recs = session.query(InspectionRecord.id, InspectionRecord.certificate_id).all()
    sample_ids = [r[0] for r in all_recs]
    sample_certs = [r[1] for r in all_recs]

    benchmark_results = {}

    def measure(fn, name):
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

        arr = np.array(times)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean = float(np.mean(arr))
        qps = float(1000.0 / mean) if mean > 0 else 0.0

        benchmark_results[name] = {
            "mean_ms": round(mean, 3),
            "median_p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
            "min_ms": round(float(np.min(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
            "throughput_qps": round(qps, 1)
        }
        logger.info(f"[{name}] Mean: {mean:.2f}ms | p50: {p50:.2f}ms | p95: {p95:.2f}ms | p99: {p99:.2f}ms | QPS: {qps:.1f}")

    # 1. Single Record Lookup by Indexed Primary Key
    measure(
        lambda: crud.get_inspection(session, random.choice(sample_ids)),
        "single_lookup_primary_key"
    )

    # 2. Single Record Lookup by Indexed Unique Certificate ID
    measure(
        lambda: crud.get_inspection(session, random.choice(sample_certs)),
        "single_lookup_certificate_id"
    )

    # 3. Multi-Filter Historical Query with Pagination (Domain + Decision + Sort)
    measure(
        lambda: crud.list_inspections(
            session,
            domain="GLASS_BOTTLE",
            decision="FAIL",
            skip=random.randint(0, 50),
            limit=20
        ),
        "multi_filter_paginated_query"
    )

    # 4. Factory Analytics KPI Aggregation (Pass Rate, Scores, Distributions)
    measure(
        lambda: crud.get_analytics_summary(session, domain="GLASS_BOTTLE"),
        "analytics_kpi_aggregation"
    )

    # 5. Cryptographic Record Verification (HMAC-SHA256 reconstruction from database rows)
    measure(
        lambda: crud.verify_stored_record_integrity(session, random.choice(sample_ids)),
        "cryptographic_hmac_verification"
    )

    return benchmark_results


def main():
    logger.info("Initializing isolated SQLite memory benchmark database...")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()

    try:
        generate_benchmark_dataset(session, num_records=1000)
        results = run_benchmarks(session, iterations=100)

        out_file = REPORTS_DIR / "db_query_benchmark.json"
        out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Benchmark results successfully saved to {out_file}")

    finally:
        session.close()


if __name__ == "__main__":
    main()
