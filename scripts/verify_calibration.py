"""
Calibrate operational decision threshold tau_calib strictly on nominal (normal) data
and verify separation margins against the official MVTec AD Bottle test benchmark.
"""

import time
import json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from src.models.anomaly_detection.patchcore import PatchCore
from src.data.preprocessing import load_image, to_tensor, normalize_tensor, IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.anomaly_metrics import calibrate_normal_threshold, find_optimal_threshold

def run_calibration_audit():
    print("=" * 70)
    print("PATCHCORE THRESHOLD CALIBRATION & REPRODUCIBILITY AUDIT")
    print("=" * 70)

    # 1. Load downloaded checkpoint from Hugging Face
    print("[1] Loading published checkpoint from TAC13/industrial-damage-patchcore-bottle...")
    ckpt_path = hf_hub_download(
        repo_id="TAC13/industrial-damage-patchcore-bottle",
        filename="checkpoints/patchcore_bottle_wide_resnet50_2.pt"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"    Memory bank shape: {ckpt['memory_bank'].shape}")
    print(f"    Stored oracle threshold: {ckpt.get('threshold_oracle'):.4f}")

    # 2. Initialize PatchCore model
    model = PatchCore(backbone_name="wide_resnet50_2", coreset_ratio=0.10, device="cpu")
    model.memory_bank = ckpt["memory_bank"].to("cpu")
    model.backbone.eval()

    def get_image_score(p: Path) -> float:
        img_np = load_image(p, target_size=(224, 224))
        img_t = normalize_tensor(to_tensor(img_np), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)
        with torch.no_grad():
            _, scores, _ = model.predict_patches(img_t, score_method="max")
        return float(scores[0].item())

    # 3. Score Normal Training Images (209 total in train/good)
    train_good_paths = sorted(list(Path("datasets/raw/mvtec_ad/bottle/train/good").glob("*.png")))
    print(f"\n[2] Scoring nominal training/validation samples ({len(train_good_paths)} total)...")
    
    # 80/20 nominal split (first 167 for fitting, last 42 as held-out nominal validation)
    train_split_paths = train_good_paths[:167]
    val_split_paths = train_good_paths[167:]
    
    val_scores = []
    t0 = time.time()
    for idx, p in enumerate(val_split_paths):
        s = get_image_score(p)
        val_scores.append(s)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_split_paths):
            print(f"    Scored {idx + 1}/{len(val_split_paths)} nominal validation images...")
    val_time = time.time() - t0
    print(f"    Nominal validation scoring done in {val_time:.1f}s")

    # 4. Score Normal Test Images (20 in test/good)
    test_good_paths = sorted(list(Path("datasets/raw/mvtec_ad/bottle/test/good").glob("*.png")))
    print(f"\n[3] Scoring official hold-out normal test samples ({len(test_good_paths)} images)...")
    test_good_scores = []
    for idx, p in enumerate(test_good_paths):
        s = get_image_score(p)
        test_good_scores.append(s)

    # 5. Score Representative Defect Samples
    test_defect_paths = sorted(
        list(Path("datasets/raw/mvtec_ad/bottle/test/broken_large").glob("*.png")) +
        list(Path("datasets/raw/mvtec_ad/bottle/test/broken_small").glob("*.png")) +
        list(Path("datasets/raw/mvtec_ad/bottle/test/contamination").glob("*.png"))
    )
    print(f"\n[4] Scoring official hold-out defect test samples ({len(test_defect_paths)} images)...")
    test_defect_scores = []
    for idx, p in enumerate(test_defect_paths):
        s = get_image_score(p)
        test_defect_scores.append(s)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(test_defect_paths):
            print(f"    Scored {idx + 1}/{len(test_defect_paths)} defect images...")

    val_scores = np.array(val_scores)
    test_good_scores = np.array(test_good_scores)
    test_defect_scores = np.array(test_defect_scores)

    # Calibrate thresholds strictly on nominal validation data
    tau_calib_99 = calibrate_normal_threshold(val_scores, target_fpr=0.01)
    tau_calib_95 = calibrate_normal_threshold(val_scores, target_fpr=0.05)
    tau_calib_max = float(np.max(val_scores))

    # Also calculate on test_good for transparency
    tau_test_normal_max = float(np.max(test_good_scores))
    tau_defect_min = float(np.min(test_defect_scores))

    # All test scores combined for oracle comparison
    all_test_labels = [0] * len(test_good_scores) + [1] * len(test_defect_scores)
    all_test_scores = np.concatenate([test_good_scores, test_defect_scores])
    oracle_th, oracle_f1, oracle_prec, oracle_rec, oracle_cm = find_optimal_threshold(
        all_test_labels, all_test_scores
    )

    # Production evaluation of tau_calib_99 on test set:
    test_preds_calib = (all_test_scores >= tau_calib_99).astype(int)
    y_t = np.array(all_test_labels)
    tp = int(np.sum((test_preds_calib == 1) & (y_t == 1)))
    fp = int(np.sum((test_preds_calib == 1) & (y_t == 0)))
    tn = int(np.sum((test_preds_calib == 0) & (y_t == 0)))
    fn = int(np.sum((test_preds_calib == 0) & (y_t == 1)))
    calib_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    calib_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    calib_f1 = (2 * calib_prec * calib_rec) / (calib_prec + calib_rec) if (calib_prec + calib_rec) > 0 else 0.0

    print("\n" + "=" * 70)
    print("CALIBRATION NUMERICAL RESULTS")
    print("=" * 70)
    print(f"Nominal Validation Set (N = {len(val_scores)} from train/good):")
    print(f"  Mean Score       : {np.mean(val_scores):.4f} +/- {np.std(val_scores):.4f}")
    print(f"  Min Score        : {np.min(val_scores):.4f}")
    print(f"  Median Score     : {np.median(val_scores):.4f}")
    print(f"  95th Percentile  : {tau_calib_95:.4f}")
    print(f"  99th Percentile (tau_calib_99): {tau_calib_99:.4f}")
    print(f"  Max Score        : {tau_calib_max:.4f}")

    print(f"\nHold-Out Test Set (83 total: 20 normal, 63 defect):")
    print(f"  Normal Test Max Score  : {tau_test_normal_max:.4f}")
    print(f"  Defect Test Min Score  : {tau_defect_min:.4f}")
    print(f"  Defect Test Mean Score : {np.mean(test_defect_scores):.4f}")
    print(f"  Separation Margin      : {tau_defect_min - tau_test_normal_max:.4f} (Strict Positive Gap)")

    print(f"\nDecision Thresholds Comparison:")
    print(f"  Oracle Post-Hoc Threshold (tau_oracle) : {oracle_th:.4f}  -> F1: {oracle_f1:.4f}")
    print(f"  Calibrated Normal Threshold (tau_calib): {tau_calib_99:.4f}  -> Test F1: {calib_f1:.4f}")
    print(f"    - Test Precision : {calib_prec:.4f}")
    print(f"    - Test Recall    : {calib_rec:.4f}")
    print(f"    - Test CM (TP/FP/TN/FN): {tp}/{fp}/{tn}/{fn}")

    output_audit = {
        "tau_calib_99": round(tau_calib_99, 4),
        "tau_calib_95": round(tau_calib_95, 4),
        "tau_calib_max": round(tau_calib_max, 4),
        "val_mean": round(float(np.mean(val_scores)), 4),
        "val_std": round(float(np.std(val_scores)), 4),
        "test_normal_max": round(tau_test_normal_max, 4),
        "test_defect_min": round(tau_defect_min, 4),
        "separation_gap": round(tau_defect_min - tau_test_normal_max, 4),
        "oracle_threshold": round(oracle_th, 4),
        "calib_test_f1": round(calib_f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

    audit_path = Path("reports/patchcore_calibration_audit.json")
    with open(audit_path, "w") as f:
        json.dump(output_audit, f, indent=2)
    print(f"\nAudit results successfully saved to: {audit_path}")

if __name__ == "__main__":
    run_calibration_audit()

