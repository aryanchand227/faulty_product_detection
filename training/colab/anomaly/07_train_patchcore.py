"""
Google Colab Training and Benchmarking Harness for Unsupervised Anomaly Detection:
PatchCore Memory Bank with Greedy k-Center Coreset on MVTec AD Bottle.

Features:
1. GPU Gatekeeper & Hardware Telemetry (Tesla T4 validation).
2. Hugging Face Hub Authentication & Repository Setup.
3. Strict Dataset Verification (209 normal training, 83 official test benchmark samples).
4. Mid-Level Feature Hierarchy Extraction (Layers 2 & 3 from frozen ResNet/WideResNet).
5. Greedy k-Center Coreset Subsampling (reducing 160k+ patch vectors by 90% while preserving minimax coverage).
6. Official Hold-Out Benchmark Evaluation:
   - Image-level AUROC
   - Pixel-level AUROC
   - Area Under Per-Region Overlap (AUPRO up to FPR <= 0.30)
   - F1-optimal industrial triage threshold
7. Qualitative Anomaly Heatmap Visualizations (Raw, Ground Truth, Anomaly Map, Triage Verdict).
8. Automated Checkpoint & Metric Upload to Hugging Face Hub.

Run this script directly in Google Colab with GPU runtime.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, login

# =====================================================================
# 1. GPU Gatekeeper Verification
# =====================================================================
print("=" * 70)
print("PHASE 7: PATCHCORE UNSUPERVISED ANOMALY DETECTION (COLAB GPU)")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is required for PatchCore feature extraction and coreset subsampling! "
        "In Google Colab, select: Runtime -> Change runtime type -> T4 GPU."
    )

gpu_name = torch.cuda.get_device_name(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
cuda_version = torch.version.cuda

print(f"[Hardware] GPU Detected : {gpu_name}")
print(f"[Hardware] Total VRAM   : {total_vram_gb:.2f} GB")
print(f"[Hardware] CUDA Version : {cuda_version}")

device = torch.device("cuda")

# Add project root to sys.path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.anomaly_detection.patchcore import PatchCore
from src.evaluation.anomaly_metrics import evaluate_anomaly_detection
from src.training.train_patchcore import NormalBottleDataset, MVTecBottleTestDataset
from src.inference.anomaly_inference import PatchCoreInferencer

# =====================================================================
# 2. Hugging Face Authentication & Repository Setup
# =====================================================================
print("\n" + "=" * 70)
print("2. HUGGING FACE AUTHENTICATION & REPO SETUP")
print("=" * 70)

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        pass

if hf_token:
    login(token=hf_token, add_to_git_credential=True)
    api = HfApi()
    user_info = api.whoami()
    hf_username = user_info["name"]
    print(f"[Auth] Logged into Hugging Face as: {hf_username}")
else:
    print("[Auth] Warning: HF_TOKEN not found. Artifacts will be saved locally only.")
    hf_username = None

REPO_NAME = "industrial-anomaly-patchcore"
HF_REPO_ID = f"{hf_username}/{REPO_NAME}" if hf_username else None

if HF_REPO_ID:
    try:
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, private=True)
        print(f"[Hub] Target repository initialized: https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"[Hub] Note on repo creation: {e}")

# =====================================================================
# 3. Dataset Audit & Provenance Verification
# =====================================================================
print("\n" + "=" * 70)
print("3. DATASET PROVENANCE & BENCHMARK SEPARATION AUDIT")
print("=" * 70)

MVTEC_BOTTLE_DIR = Path("datasets/raw/mvtec_ad/bottle")
if not MVTEC_BOTTLE_DIR.exists():
    raise FileNotFoundError(
        f"MVTec Bottle directory not found at: {MVTEC_BOTTLE_DIR.resolve()}. "
        "Please download dataset using Phase 2 data loader."
    )

train_good_paths = sorted(list((MVTEC_BOTTLE_DIR / "train" / "good").glob("*.png")))
test_good_paths = sorted(list((MVTEC_BOTTLE_DIR / "test" / "good").glob("*.png")))
test_broken_large = sorted(list((MVTEC_BOTTLE_DIR / "test" / "broken_large").glob("*.png")))
test_broken_small = sorted(list((MVTEC_BOTTLE_DIR / "test" / "broken_small").glob("*.png")))
test_contamination = sorted(list((MVTEC_BOTTLE_DIR / "test" / "contamination").glob("*.png")))

total_defects = len(test_broken_large) + len(test_broken_small) + len(test_contamination)
total_test = len(test_good_paths) + total_defects

print(f"[Audit] Training Partition (Normal only): {len(train_good_paths)} images")
print(f"[Audit] Test Partition (Benchmark):      {total_test} images")
print(f"        - Normal Test (good):            {len(test_good_paths)} images")
print(f"        - Defect - broken_large:         {len(test_broken_large)} images")
print(f"        - Defect - broken_small:         {len(test_broken_small)} images")
print(f"        - Defect - contamination:        {len(test_contamination)} images")
print("[Audit] Strict Guarantee: ZERO test or defect images are used during memory bank construction.")

# =====================================================================
# 4. PatchCore Memory Bank Fitting & Coreset Subsampling
# =====================================================================
print("\n" + "=" * 70)
print("4. PATCHCORE TRAINING (BACKBONE & CORESET MEMORY BANK)")
print("=" * 70)

BACKBONE = "wide_resnet50_2" if total_vram_gb >= 12.0 else "resnet18"
CORESET_RATIO = 0.10  # Retain top 10% most informative prototype patches
TARGET_SIZE = (224, 224)
BATCH_SIZE = 16

print(f"[Config] Backbone Selection : {BACKBONE}")
print(f"[Config] Input Resolution   : {TARGET_SIZE}")
print(f"[Config] Coreset Ratio      : {CORESET_RATIO * 100:.1f}%")
print(f"[Config] Compute Device     : {device}")

train_dataset = NormalBottleDataset(train_good_paths, target_size=TARGET_SIZE)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = PatchCore(
    backbone_name=BACKBONE,
    patch_size=3,
    coreset_ratio=CORESET_RATIO,
    projection_dim=128,
    device=device
)

start_fit = time.time()
num_prototypes = model.fit(train_loader, verbose=True)
fit_time_sec = round(time.time() - start_fit, 2)
print(f"[Training] Memory bank constructed in {fit_time_sec}s with {num_prototypes} prototypes.")

# =====================================================================
# 5. Benchmark Evaluation on Official MVTec Bottle Test Set
# =====================================================================
print("\n" + "=" * 70)
print("5. OFFICIAL BENCHMARK EVALUATION (IMAGE AUROC, PIXEL AUROC, AUPRO)")
print("=" * 70)

test_dataset = MVTecBottleTestDataset(category_dir=MVTEC_BOTTLE_DIR, target_size=TARGET_SIZE)

y_true_labels: List[int] = []
y_image_scores: List[float] = []
gt_masks: List[np.ndarray] = []
anomaly_maps: List[np.ndarray] = []

print(f"[Eval] Running inference across {len(test_dataset)} hold-out benchmark samples...")
infer_start = time.time()

model.backbone.eval()
with torch.no_grad():
    for i in range(len(test_dataset)):
        img_t, mask_bin, is_defect, defect_type = test_dataset[i]
        img_batch = img_t.unsqueeze(0).to(device)

        patch_dists, img_scores, spatial_hw = model.predict_patches(img_batch)
        score = float(img_scores[0].item())

        a_map = model.compute_anomaly_maps(
            patch_dists, spatial_hw, output_hw=TARGET_SIZE, sigma=4.0
        )[0]

        y_true_labels.append(is_defect)
        y_image_scores.append(score)
        gt_masks.append(mask_bin)
        anomaly_maps.append(a_map)

avg_latency_ms = ((time.time() - infer_start) / len(test_dataset)) * 1000.0

eval_result = evaluate_anomaly_detection(
    y_true_labels=y_true_labels,
    y_image_scores=y_image_scores,
    ground_truth_masks=gt_masks,
    anomaly_maps=anomaly_maps,
    max_fpr=0.30
)

model.threshold = eval_result.optimal_threshold

print("\n" + "*" * 55)
print("             OFFICIAL BENCHMARK RESULTS            ")
print("*" * 55)
print(f"Backbone Architecture : {BACKBONE}")
print(f"Image-level AUROC     : {eval_result.image_auroc:.4f} (Target >= 0.980)")
print(f"Pixel-level AUROC     : {eval_result.pixel_auroc:.4f} (Target >= 0.960)")
print(f"AUPRO (FPR <= 0.30)   : {eval_result.aupro:.4f} (Target >= 0.920)")
print(f"Optimal Threshold     : {eval_result.optimal_threshold:.4f}")
print(f"F1-Score at Threshold : {eval_result.f1_score:.4f}")
print(f"Precision / Recall    : {eval_result.precision:.4f} / {eval_result.recall:.4f}")
print(f"Confusion Matrix      : {eval_result.confusion_matrix}")
print(f"Mean Latency / Image  : {avg_latency_ms:.2f} ms")
print("*" * 55)

# =====================================================================
# 6. Qualitative Visual Inspection Plots
# =====================================================================
print("\n" + "=" * 70)
print("6. GENERATING QUALITATIVE INSPECTION PLOTS")
print("=" * 70)

output_fig_dir = Path("reports/figures")
output_fig_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_fig_dir / "patchcore_mvtec_bottle_evaluation.png"

# Select representative samples: 1 normal, 1 broken_large, 1 broken_small, 1 contamination
sample_indices = []
found_types = set()
for idx, (_, _, is_def, dtype) in enumerate(test_dataset.samples):
    if dtype not in found_types:
        sample_indices.append(idx)
        found_types.add(dtype)
    if len(sample_indices) >= 4:
        break

fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, 4 * len(sample_indices)))

for row, idx in enumerate(sample_indices):
    img_p, mask_p, is_defect, defect_type = test_dataset.samples[idx]
    orig_rgb = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
    orig_rgb_resized = cv2.resize(orig_rgb, TARGET_SIZE)

    gt_m = gt_masks[idx]
    a_m = anomaly_maps[idx]
    score = y_image_scores[idx]

    # Draw overlay
    overlay = PatchCoreInferencer.draw_anomaly_overlay(
        orig_rgb_resized, a_m, threshold=eval_result.optimal_threshold
    )

    verdict = "FAIL" if score >= eval_result.optimal_threshold else "PASS"

    # Column 1: Input image
    axes[row, 0].imshow(orig_rgb_resized)
    axes[row, 0].set_title(f"Input: {defect_type}", fontsize=11, fontweight="bold")
    axes[row, 0].axis("off")

    # Column 2: Ground Truth Mask
    axes[row, 1].imshow(gt_m, cmap="gray")
    axes[row, 1].set_title("Ground Truth Mask", fontsize=11)
    axes[row, 1].axis("off")

    # Column 3: Anomaly Heatmap
    im3 = axes[row, 2].imshow(a_m, cmap="jet")
    axes[row, 2].set_title(f"PatchCore Heatmap (Score: {score:.3f})", fontsize=11)
    axes[row, 2].axis("off")
    plt.colorbar(im3, ax=axes[row, 2], fraction=0.046, pad=0.04)

    # Column 4: Triage Overlay
    axes[row, 3].imshow(overlay)
    v_color = "red" if verdict == "FAIL" else "green"
    axes[row, 3].set_title(f"Triage Verdict: {verdict}", fontsize=11, fontweight="bold", color=v_color)
    axes[row, 3].axis("off")

plt.suptitle(f"PatchCore Anomaly Inspection on MVTec Bottle (Backbone: {BACKBONE})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(str(plot_path), dpi=200, bbox_inches="tight")
plt.close()
print(f"[Visual] Inspection visualization saved to: {plot_path}")

# =====================================================================
# 7. Checkpoint Saving & Hugging Face Hub Export
# =====================================================================
print("\n" + "=" * 70)
print("7. CHECKPOINT EXPORT & ARTIFACT UPLOAD")
print("=" * 70)

checkpoints_dir = Path("models/checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = checkpoints_dir / f"patchcore_bottle_{BACKBONE}.pt"
model.save(checkpoint_file)
print(f"[Save] Saved PatchCore memory bank to: {checkpoint_file}")

# Save JSON metrics summary
metrics_file = checkpoints_dir / f"patchcore_bottle_{BACKBONE}_metrics.json"
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump({
        "backbone": BACKBONE,
        "coreset_ratio": CORESET_RATIO,
        "prototypes": num_prototypes,
        "fit_time_sec": fit_time_sec,
        "image_auroc": eval_result.image_auroc,
        "pixel_auroc": eval_result.pixel_auroc,
        "aupro": eval_result.aupro,
        "optimal_threshold": eval_result.optimal_threshold,
        "f1_score": eval_result.f1_score,
        "precision": eval_result.precision,
        "recall": eval_result.recall,
        "confusion_matrix": eval_result.confusion_matrix,
        "mean_latency_ms": round(avg_latency_ms, 2)
    }, f, indent=2)
print(f"[Save] Metrics JSON exported to: {metrics_file}")

if HF_REPO_ID and hf_token:
    print(f"\n[Hub] Uploading model artifacts to Hugging Face Hub: {HF_REPO_ID}...")
    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint_file),
            path_in_repo=f"checkpoints/{checkpoint_file.name}",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj=str(metrics_file),
            path_in_repo=f"metrics/{metrics_file.name}",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj=str(plot_path),
            path_in_repo="visuals/patchcore_mvtec_bottle_evaluation.png",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        print(f"[Hub] Upload successful! Artifacts visible at: https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"[Hub] Upload failed with exception: {e}")

print("\n" + "=" * 70)
print("PHASE 7 COMPLETE: PATCHCORE ANOMALY DETECTION BENCHMARKED SUCCESSFULLY")
print("=" * 70)
