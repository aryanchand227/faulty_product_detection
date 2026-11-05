"""
Demo script executing the Unified Multi-Tier Decision Pipeline
on sample normal, broken, and contaminated industrial parts.
Saves annotated visual HUD inspection overlays and quad-panel dashboards.
"""

from pathlib import Path
import cv2
import torch
from huggingface_hub import hf_hub_download

from src.models.anomaly_detection.patchcore import PatchCore
from src.inference.anomaly_inference import PatchCoreInferencer
from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import DecisionEngine, QualityThresholds
from src.inspection.unified_pipeline import ProductDomain, UnifiedQualityInspector
from src.inspection.visualizer import InspectionVisualizer

def run_demo():
    print("=" * 70)
    print("UNIFIED MULTI-TIER INSPECTION PIPELINE DEMO (GLASS_BOTTLE DOMAIN)")
    print("=" * 70)

    device = "cpu"
    output_dir = Path("reports/figures/unified_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tier 1 PatchCore from cached checkpoint
    print("[1] Initializing Tier 1 PatchCore Anomaly Gatekeeper & Localization...")
    ckpt_path = hf_hub_download(
        repo_id="TAC13/industrial-damage-patchcore-bottle",
        filename="checkpoints/patchcore_bottle_wide_resnet50_2.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    patchcore_model = PatchCore(backbone_name="wide_resnet50_2", coreset_ratio=0.10, device=device)
    patchcore_model.memory_bank = ckpt["memory_bank"].to(device)
    patchcore_model.backbone.eval()
    t1_inferencer = PatchCoreInferencer(patchcore_instance=patchcore_model, device=device)

    # 2. Camera Calibrator (0.15 mm per pixel: e.g. 224 px = 33.6 mm field of view)
    calibrator = CameraCalibrator(mm_per_pixel=0.15)

    # 3. Decision Engine
    # Using the operational separation threshold: tau = 15.50 (safety band: [13.95, 17.05])
    # Distinguishes nominal test samples (scores <= 15.37) from defective samples (scores >= 18.10)
    thresholds = QualityThresholds(
        anomaly_threshold=15.50,
        anomaly_margin=0.10,
        critical_area_mm2=25.0,  # Demonstration threshold for reject
        critical_classes=["broken_large", "crazing"]
    )
    decision_engine = DecisionEngine(thresholds=thresholds)

    # 4. Unified Inspector configured for GLASS_BOTTLE domain
    inspector = UnifiedQualityInspector(
        domain=ProductDomain.GLASS_BOTTLE,
        tier1_inferencer=t1_inferencer,
        calibrator=calibrator,
        decision_engine=decision_engine,
        early_exit=True,
        device=device
    )

    # Test Samples
    samples = [
        ("nominal_part", Path("datasets/raw/mvtec_ad/bottle/test/good/000.png")),
        ("broken_small_part", Path("datasets/raw/mvtec_ad/bottle/test/broken_small/000.png")),
        ("broken_large_part", Path("datasets/raw/mvtec_ad/bottle/test/broken_large/000.png")),
        ("contamination_part", Path("datasets/raw/mvtec_ad/bottle/test/contamination/000.png"))
    ]

    for name, img_path in samples:
        if not img_path.exists():
            print(f"Skipping {name}: file {img_path} not found.")
            continue

        print(f"\n---> Inspecting: {name} ({img_path.name})")
        result = inspector.inspect(img_path, render_visuals=True, force_full_pipeline=(name != "nominal_part"))

        print(f"     Verdict               : {result.verdict.decision.value} [{result.verdict.routing.value}]")
        print(f"     Severity              : {result.verdict.severity.value}")
        print(f"     Anomaly Score         : {result.verdict.anomaly_score:.2f}")
        print(f"     Defect Class          : {result.verdict.primary_defect_type}")
        print(f"     Projected Damage Area : {result.verdict.physical_defect_area_mm2:.2f} mm² ({result.physical_metrics['measurement_mode']})")
        print(f"     Early Exit Applied?   : {result.early_exit_applied}")
        print(f"     Total Latency         : {result.total_latency_ms:.1f} ms")
        if result.certificate:
            print(f"     Dominant Morphology   : {result.certificate.get('dominant_morphology')}")
            print(f"     Max Defect Length     : {result.certificate.get('max_defect_length_mm', 0.0):.2f} mm")
            print(f"     Area Reconciliation   : Total={result.certificate.get('total_defect_area_mm2', 0.0)} mm², Components={result.certificate.get('component_defect_area_mm2', 0.0)} mm², Noise={result.certificate.get('filtered_noise_area_mm2', 0.0)} mm² (Reconciled: {result.certificate.get('is_area_reconciled')})")
            print(f"     HMAC-SHA256 Signature : {result.certificate.get('hmac_sha256_signature')[:24]}...")
            print(f"     Integrity Hash        : {result.certificate.get('sha256_integrity_hash')[:24]}...")
        print(f"     Rationale             : {result.verdict.engineering_rationale}")

        # Save Certificate to reports/certificates/
        cert_dir = Path("reports/certificates")
        cert_dir.mkdir(parents=True, exist_ok=True)
        if result.certificate:
            cert_file = cert_dir / f"{name}_certificate.json"
            import json
            with open(cert_file, "w") as f:
                json.dump(result.certificate, f, indent=2)
            print(f"     Saved Certificate     : {cert_file}")

        # Save annotated HUD image
        if result.annotated_image is not None:
            hud_path = output_dir / f"hud_{name}.png"
            cv2.imwrite(str(hud_path), cv2.cvtColor(result.annotated_image, cv2.COLOR_RGB2BGR))
            print(f"     Saved HUD visual      : {hud_path}")

        # Generate and save Quad-Panel Dashboard
        img_rgb = inspector._load_image_rgb(img_path)
        quad = InspectionVisualizer.create_quad_panel_dashboard(
            image_rgb=img_rgb,
            verdict=result.verdict,
            anomaly_heatmap=None,
            boxes=None,
            segmentation_mask=None,
            latency_ms=result.total_latency_ms
        )
        quad_path = output_dir / f"quad_{name}.png"
        cv2.imwrite(str(quad_path), cv2.cvtColor(quad, cv2.COLOR_RGB2BGR))
        print(f"     Saved Quad Panel      : {quad_path}")

    print("\n" + "=" * 70)
    print(f"DEMO COMPLETE: All visual inspection reports saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
