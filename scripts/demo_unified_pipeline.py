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
from src.inference.classifier_inference import ClassifierInferencer, MVTEC_BOTTLE_CLASSES
from src.inference.segmenter_inference import UNetSegmenter
from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import DecisionEngine, QualityThresholds
from src.inspection.unified_pipeline import UnifiedQualityInspector
from src.inspection.visualizer import InspectionVisualizer

def run_demo():
    print("=" * 70)
    print("UNIFIED MULTI-TIER INSPECTION PIPELINE DEMO")
    print("=" * 70)

    device = "cpu"
    output_dir = Path("reports/figures/unified_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tier 1 PatchCore from cached checkpoint
    print("[1] Initializing Tier 1 PatchCore Anomaly Gatekeeper...")
    ckpt_path = hf_hub_download(
        repo_id="TAC13/industrial-damage-patchcore-bottle",
        filename="checkpoints/patchcore_bottle_wide_resnet50_2.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    patchcore_model = PatchCore(backbone_name="wide_resnet50_2", coreset_ratio=0.10, device=device)
    patchcore_model.memory_bank = ckpt["memory_bank"].to(device)
    patchcore_model.backbone.eval()
    t1_inferencer = PatchCoreInferencer(patchcore_instance=patchcore_model, device=device)

    # 2. Initialize Tier 2 Classifier
    print("[2] Initializing Tier 2 Multi-Class Classifier...")
    t2_inferencer = ClassifierInferencer(
        class_names=MVTEC_BOTTLE_CLASSES,
        device=device,
        nominal_class_name="good"
    )

    # 3. Initialize Tier 3 Segmenter
    print("[3] Initializing Tier 3 Semantic Segmenter...")
    t3_segmenter = UNetSegmenter(device=device)

    # 4. Camera Calibrator (0.15 mm per pixel: e.g. 224 px = 33.6 mm field of view)
    calibrator = CameraCalibrator(mm_per_pixel=0.15)

    # 5. Decision Engine
    thresholds = QualityThresholds(
        anomaly_threshold=15.5,
        anomaly_margin=0.15,
        critical_area_mm2=3.0,
        critical_classes=["broken_large", "crazing"]
    )
    decision_engine = DecisionEngine(thresholds=thresholds)

    # 6. Unified Inspector
    inspector = UnifiedQualityInspector(
        tier1_inferencer=t1_inferencer,
        tier2_inferencer=t2_inferencer,
        tier3_detector=None,  # Fallback to U-Net segmentation
        tier3_segmenter=t3_segmenter,
        calibrator=calibrator,
        decision_engine=decision_engine,
        early_exit=True,
        device=device
    )

    # Test Samples
    samples = [
        ("nominal_part", Path("datasets/raw/mvtec_ad/bottle/test/good/000.png")),
        ("broken_large_part", Path("datasets/raw/mvtec_ad/bottle/test/broken_large/000.png")),
        ("contamination_part", Path("datasets/raw/mvtec_ad/bottle/test/contamination/000.png"))
    ]

    for name, img_path in samples:
        if not img_path.exists():
            print(f"Skipping {name}: file {img_path} not found.")
            continue

        print(f"\n---> Inspecting: {name} ({img_path.name})")
        result = inspector.inspect(img_path, render_visuals=True, force_full_pipeline=(name != "nominal_part"))

        print(f"     Verdict          : {result.verdict.decision.value} [{result.verdict.routing.value}]")
        print(f"     Severity         : {result.verdict.severity.value}")
        print(f"     Anomaly Score    : {result.verdict.anomaly_score:.2f}")
        print(f"     Defect Class     : {result.verdict.primary_defect_type}")
        print(f"     Physical Area    : {result.verdict.physical_defect_area_mm2:.2f} mm²")
        print(f"     Early Exit?      : {result.early_exit_applied}")
        print(f"     Total Latency    : {result.total_latency_ms:.1f} ms")
        print(f"     Rationale        : {result.verdict.engineering_rationale}")

        # Save annotated HUD image
        if result.annotated_image is not None:
            hud_path = output_dir / f"hud_{name}.png"
            cv2.imwrite(str(hud_path), cv2.cvtColor(result.annotated_image, cv2.COLOR_RGB2BGR))
            print(f"     Saved HUD visual : {hud_path}")

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
        print(f"     Saved Quad Panel : {quad_path}")

    print("\n" + "=" * 70)
    print(f"DEMO COMPLETE: All visual inspection reports saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
