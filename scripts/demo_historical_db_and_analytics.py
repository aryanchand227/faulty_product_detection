"""
Demonstration script executing the Phase 10 Inspection Database Persistence & Quality Analytics Engine.
Inspects benchmark samples, persists them with defect components to SQLite/PostgreSQL,
executes cryptographic HMAC tamper audits on stored records, and displays factory analytics.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pprint

from backend.app.core.config import settings
from backend.app.database.connection import init_db, SessionLocal
from backend.app.database import crud
from src.inspection.unified_pipeline import ProductDomain, UnifiedQualityInspector
from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import DecisionEngine, QualityThresholds
from src.inference.anomaly_inference import PatchCoreInferencer
from src.models.anomaly_detection.patchcore import PatchCore
import torch
from huggingface_hub import hf_hub_download


def run_phase10_demo():
    print("=" * 80)
    print("PHASE 10: INSPECTION DATABASE PERSISTENCE, AUDIT & QUALITY ANALYTICS DEMO")
    print("=" * 80)

    # 1. Initialize Database Schema
    print("\n[1] Initializing Relational Database Schema...")
    init_db()
    db = SessionLocal()
    print(f"    Connected to: {settings.DATABASE_URL.split('@')[-1]}")

    # 2. Initialize Inspector Pipeline
    print("\n[2] Initializing Tier 1 PatchCore Gatekeeper from HuggingFace Checkpoint...")
    device = "cpu"
    ckpt_path = hf_hub_download(
        repo_id=settings.HF_REPO_ID,
        filename="checkpoints/patchcore_bottle_wide_resnet50_2.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PatchCore(backbone_name="wide_resnet50_2", coreset_ratio=0.10, device=device)
    model.memory_bank = ckpt["memory_bank"].to(device)
    model.backbone.eval()
    t1_inferencer = PatchCoreInferencer(patchcore_instance=model, device=device)

    calibrator = CameraCalibrator(mm_per_pixel=0.15)
    thresholds = QualityThresholds(
        anomaly_threshold=15.50,
        anomaly_margin=0.10,
        critical_area_mm2=25.0,
        critical_classes=["broken_large", "crazing"]
    )
    engine = DecisionEngine(thresholds=thresholds)

    inspector = UnifiedQualityInspector(
        domain=ProductDomain.GLASS_BOTTLE,
        tier1_inferencer=t1_inferencer,
        calibrator=calibrator,
        decision_engine=engine,
        early_exit=True,
        device=device
    )

    # 3. Process Benchmark Samples and Persist
    samples = [
        ("nominal_part", Path("datasets/raw/mvtec_ad/bottle/test/good/000.png")),
        ("broken_small_part", Path("datasets/raw/mvtec_ad/bottle/test/broken_small/000.png")),
        ("broken_large_part", Path("datasets/raw/mvtec_ad/bottle/test/broken_large/000.png")),
        ("contamination_part", Path("datasets/raw/mvtec_ad/bottle/test/contamination/000.png")),
    ]

    print("\n[3] Executing Inspections and Persisting Relational Records...")
    saved_ids = []

    for name, path in samples:
        if not path.exists():
            continue

        raw_bytes = path.read_bytes()
        res = inspector.inspect(
            path,
            render_visuals=True,
            force_full_pipeline=(name != "nominal_part"),
            extract_morphology=True,
            generate_certificate=True
        )

        res_dict = {
            "verdict": {
                "decision": res.verdict.decision.value,
                "severity": res.verdict.severity.value,
                "routing": res.verdict.routing.value,
                "anomaly_score": res.verdict.anomaly_score,
                "primary_defect_type": res.verdict.primary_defect_type,
                "physical_defect_area_mm2": res.verdict.physical_defect_area_mm2,
                "engineering_rationale": res.verdict.engineering_rationale,
            },
            "product_domain": res.product_domain,
            "total_latency_ms": res.total_latency_ms,
            "early_exit_applied": res.early_exit_applied,
            "physical_metrics": res.physical_metrics,
            "morphological_features": res.morphological_features,
            "certificate": res.certificate,
        }

        # Persist to database
        annotated_bytes = None
        if res.annotated_image is not None:
            import cv2
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(res.annotated_image, cv2.COLOR_RGB2BGR))
            annotated_bytes = buf.tobytes()

        record = crud.create_inspection_record(
            db=db,
            result_dict=res_dict,
            source_image_bytes=raw_bytes,
            annotated_image_bytes=annotated_bytes
        )
        saved_ids.append(record.id)

        print(f"    --> Persisted [{record.id}] {record.certificate_id}: {record.decision} [{record.routing_action}] | Score: {record.anomaly_score:.2f} | Defect: {record.primary_defect_type} | Components: {len(record.components)}")

    # 4. Query Historical Inspections
    print("\n[4] Querying Historical Inspections (Decision = FAIL)...")
    fails, fail_total = crud.list_inspections(db, decision="FAIL", limit=5)
    print(f"    Found {fail_total} FAIL records in database:")
    for f in fails:
        print(f"    - ID: {f.id} | Cert: {f.certificate_id} | Defect: {f.primary_defect_type} | Morphology: {f.dominant_morphology} | Damage Area: {f.total_defect_area_mm2:.2f} mm²")

    # 5. Cryptographic Record Tamper Audit
    print("\n[5] Executing Cryptographic HMAC-SHA256 Integrity Verification on Stored Records...")
    for rec_id in saved_ids:
        ver = crud.verify_stored_record_integrity(db, rec_id)
        status_tag = "VALID (Untampered)" if ver["is_valid"] else "TAMPERED / INVALID"
        print(f"    - Record [{rec_id}] ({ver['certificate_id']}): {status_tag} | HMAC: {ver['stored_signature'][:18]}...")

    # 6. Aggregate Factory Quality KPIs
    print("\n[6] Computing Factory Floor Quality Analytics & Yield KPIs...")
    analytics = crud.get_analytics_summary(db, domain="GLASS_BOTTLE")
    print("-" * 60)
    print(f"Total Inspected       : {analytics['total_inspections']}")
    print(f"Pass Rate             : {analytics['pass_rate_pct']}% ({analytics['pass_count']} Pass / {analytics['review_count']} Review / {analytics['fail_count']} Fail)")
    print(f"Early Exit Bypass Rate: {analytics['early_exit_rate_pct']}% ({analytics['early_exit_count']} parts bypassed Tier 2/3)")
    print(f"Mean Anomaly Score    : {analytics['mean_anomaly_score']:.4f}")
    print(f"Mean Pipeline Latency : {analytics['mean_latency_ms']:.2f} ms")
    print(f"Defect Categories     : {analytics['defect_type_distribution']}")
    print(f"Dominant Morphologies : {analytics['morphology_distribution']}")
    print(f"Vulnerable Zones      : {analytics['spatial_zone_distribution']}")
    print("-" * 60)

    db.close()
    print("\n" + "=" * 80)
    print("PHASE 10 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase10_demo()

