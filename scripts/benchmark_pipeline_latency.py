"""
Comprehensive Multi-Scenario Latency Benchmarking for Unified Quality Inspection Pipeline.

Evaluates three realistic production line scenarios:
1. Scenario 1 (Nominal Early-Exit): Tier 1 PatchCore gatekeeper only (bypasses downstream heads).
2. Scenario 2 (Defective Full Diagnostic): Tier 1 Anomaly + Tier 2 Classifier + Tier 3 Localization + Decision Engine + HUD Rendering.
3. Scenario 3 (Mixed Production Stream): Realistic factory line blend (e.g. 99% nominal, 1% defective, and 95% nominal, 5% defective).

Reports Mean, Median (p50), p95, p99, FPS (parts/sec), and Parts Per Minute (PPM).
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from src.models.anomaly_detection.patchcore import PatchCore
from src.inference.anomaly_inference import PatchCoreInferencer
from src.inference.classifier_inference import ClassifierInferencer
from src.inference.segmenter_inference import UNetSegmenter
from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import DecisionEngine, QualityThresholds
from src.inspection.unified_pipeline import ProductDomain, UnifiedQualityInspector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")


def run_latency_benchmark(num_warmup: int = 5, num_timed: int = 30) -> Dict[str, Any]:
    device = "cpu"  # Local evaluation environment
    logger.info(f"Starting Multi-Scenario Latency Profiling (Device: {device}, Warmup: {num_warmup}, Timed: {num_timed})")

    # 1. Initialize models
    logger.info("Loading cached PatchCore model...")
    ckpt_path = hf_hub_download(
        repo_id="TAC13/industrial-damage-patchcore-bottle",
        filename="checkpoints/patchcore_bottle_wide_resnet50_2.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    patchcore_model = PatchCore(backbone_name="wide_resnet50_2", coreset_ratio=0.10, device=device)
    patchcore_model.memory_bank = ckpt["memory_bank"].to(device)
    patchcore_model.backbone.eval()

    t1_inferencer = PatchCoreInferencer(patchcore_instance=patchcore_model, device=device)
    calibrator = CameraCalibrator(mm_per_pixel=0.15)
    decision_engine = DecisionEngine(QualityThresholds(anomaly_threshold=15.50, anomaly_margin=0.10))

    inspector = UnifiedQualityInspector(
        domain=ProductDomain.GLASS_BOTTLE,
        tier1_inferencer=t1_inferencer,
        calibrator=calibrator,
        decision_engine=decision_engine,
        early_exit=True,
        device=device
    )

    # Test images
    nominal_img_path = Path("datasets/raw/mvtec_ad/bottle/test/good/000.png")
    defect_img_path = Path("datasets/raw/mvtec_ad/bottle/test/broken_large/000.png")

    # =====================================================================
    # Scenario 1: Nominal Early-Exit
    # =====================================================================
    logger.info("\n---> Benchmarking Scenario 1: Nominal Early-Exit Path...")
    # Warmup
    for _ in range(num_warmup):
        _ = inspector.inspect(nominal_img_path, render_visuals=False, force_full_pipeline=False)

    nominal_times = []
    for i in range(num_timed):
        t0 = time.perf_counter()
        res = inspector.inspect(nominal_img_path, render_visuals=False, force_full_pipeline=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        nominal_times.append(elapsed_ms)
        assert res.early_exit_applied is True, "Early exit was not triggered on nominal sample!"

    s1_mean = float(np.mean(nominal_times))
    s1_median = float(np.median(nominal_times))
    s1_p95 = float(np.percentile(nominal_times, 95))
    s1_p99 = float(np.percentile(nominal_times, 99))
    s1_fps = 1000.0 / s1_median if s1_median > 0 else 0.0
    s1_ppm = s1_fps * 60.0

    # =====================================================================
    # Scenario 2: Defective Full Diagnostic Path
    # =====================================================================
    logger.info("\n---> Benchmarking Scenario 2: Defective Full Diagnostic Path (with Visuals)...")
    # Warmup
    for _ in range(num_warmup):
        _ = inspector.inspect(defect_img_path, render_visuals=True, force_full_pipeline=True)

    defective_times = []
    for i in range(num_timed):
        t0 = time.perf_counter()
        res = inspector.inspect(defect_img_path, render_visuals=True, force_full_pipeline=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        defective_times.append(elapsed_ms)
        assert res.early_exit_applied is False, "Early exit should not trigger on defect sample!"

    s2_mean = float(np.mean(defective_times))
    s2_median = float(np.median(defective_times))
    s2_p95 = float(np.percentile(defective_times, 95))
    s2_p99 = float(np.percentile(defective_times, 99))
    s2_fps = 1000.0 / s2_median if s2_median > 0 else 0.0
    s2_ppm = s2_fps * 60.0

    # =====================================================================
    # Scenario 3: Mixed Production Line (99% Nominal, 1% Defective)
    # =====================================================================
    def compute_mixed_profile(p_defect: float) -> Dict[str, float]:
        p_norm = 1.0 - p_defect
        # Weighted mixture distribution
        n_samples = 1000
        sim_nominal = np.random.choice(nominal_times, size=int(n_samples * p_norm), replace=True)
        sim_defect = np.random.choice(defective_times, size=int(n_samples * p_defect), replace=True)
        sim_mixed = np.concatenate([sim_nominal, sim_defect])

        mean_val = float(np.mean(sim_mixed))
        med_val = float(np.median(sim_mixed))
        p95_val = float(np.percentile(sim_mixed, 95))
        p99_val = float(np.percentile(sim_mixed, 99))
        fps_val = 1000.0 / mean_val if mean_val > 0 else 0.0

        return {
            "defect_rate_percent": p_defect * 100.0,
            "mean_ms": round(mean_val, 2),
            "median_ms": round(med_val, 2),
            "p95_ms": round(p95_val, 2),
            "p99_ms": round(p99_val, 2),
            "effective_fps": round(fps_val, 2),
            "parts_per_minute": round(fps_val * 60.0, 1)
        }

    mixed_99_1 = compute_mixed_profile(0.01)  # 1% defect rate
    mixed_95_5 = compute_mixed_profile(0.05)  # 5% defect rate

    results = {
        "hardware": {
            "device": device,
            "torch_version": torch.__version__,
            "input_resolution": "224x224 (normalized) / 900x900 (native)",
            "batch_size": 1,
            "warmup_iterations": num_warmup,
            "timed_iterations": num_timed
        },
        "scenario_1_nominal_early_exit": {
            "description": "Tier 1 PatchCore Gatekeeper only (no downstream heads or visuals)",
            "mean_ms": round(s1_mean, 2),
            "median_ms": round(s1_median, 2),
            "p95_ms": round(s1_p95, 2),
            "p99_ms": round(s1_p99, 2),
            "throughput_fps": round(s1_fps, 2),
            "parts_per_minute": round(s1_ppm, 1)
        },
        "scenario_2_defective_full_diagnostic": {
            "description": "Tier 1 Anomaly + Localization Heatmap + Decision Engine + Visual HUD Overlay",
            "mean_ms": round(s2_mean, 2),
            "median_ms": round(s2_median, 2),
            "p95_ms": round(s2_p95, 2),
            "p99_ms": round(s2_p99, 2),
            "throughput_fps": round(s2_fps, 2),
            "parts_per_minute": round(s2_ppm, 1)
        },
        "scenario_3_mixed_production_1pct_defects": mixed_99_1,
        "scenario_3_mixed_production_5pct_defects": mixed_95_5
    }

    out_file = Path("reports/pipeline_latency_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved benchmark results to: {out_file}")
    print("\n" + "=" * 65)
    print("      MULTI-SCENARIO PIPELINE LATENCY PROFILE")
    print("=" * 65)
    print(f"Scenario 1 (Nominal Early-Exit):")
    print(f"  Median: {s1_median:.2f} ms | p95: {s1_p95:.2f} ms | Throughput: {s1_fps:.1f} FPS ({s1_ppm:.0f} parts/min)")
    print(f"\nScenario 2 (Defective Full Diagnostic):")
    print(f"  Median: {s2_median:.2f} ms | p95: {s2_p95:.2f} ms | Throughput: {s2_fps:.1f} FPS ({s2_ppm:.0f} parts/min)")
    print(f"\nScenario 3 (Mixed Production @ 1% Defect Rate):")
    print(f"  Mean: {mixed_99_1['mean_ms']:.2f} ms | p95: {mixed_99_1['p95_ms']:.2f} ms | Effective Throughput: {mixed_99_1['effective_fps']:.1f} FPS ({mixed_99_1['parts_per_minute']:.0f} parts/min)")
    print("=" * 65)

    return results

if __name__ == "__main__":
    run_latency_benchmark(num_warmup=3, num_timed=15)
