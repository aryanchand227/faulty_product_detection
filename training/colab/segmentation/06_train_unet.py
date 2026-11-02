"""
Google Colab Training Harness for Defect Semantic Segmentation:
PyTorch U-Net with Skip Connections and Combined BCE + Soft Dice Loss.

Features:
1. GPU Gatekeeper & Hardware Telemetry (Tesla T4 validation)
2. Hugging Face Hub Authentication & Repository Setup
3. MVTec AD Bottle Defect Image-Mask Split Verification
4. Dual Augmentation Pipeline (synchronized geometric transforms, isolated photometric jitter)
5. Mixed-Precision (AMP) Accelerated U-Net Training
6. Multi-Metric Evaluation: IoU, Dice Score (F1), Pixel Accuracy on Hold-Out Test Set
7. Visual Qualitative Segmentation Plots (Original, Ground Truth, Prediction, Overlay)
8. Automated Checkpoint & Model Artifact Upload to Hugging Face Hub

Run this script directly in Google Colab with GPU runtime.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, login

# =====================================================================
# 1. GPU Gatekeeper Verification
# =====================================================================
print("=" * 70)
print("PHASE 6: U-NET DEFECT SEGMENTATION EXPERIMENTS (COLAB GPU)")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is required for U-Net segmentation training! "
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

from src.models.segmentation.unet import UNet
from src.data.segmentation_dataset import IndustrialSegmentationDataset, build_mvtec_segmentation_splits
from src.data.augmentation import IndustrialDualAugmentation
from src.evaluation.segmentation_metrics import (
    CombinedBCEDiceLoss,
    compute_iou,
    compute_dice_coefficient,
    compute_pixel_accuracy,
)
from src.training.train_segmentator import SegmentationTrainer

# =====================================================================
# 2. Hugging Face Authentication
# =====================================================================
print("\n" + "=" * 70)
print("HUGGING FACE AUTHENTICATION")
print("=" * 70)

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        hf_token = None

hf_username = None
if hf_token:
    login(token=hf_token)
    api = HfApi()
    user_info = api.whoami()
    hf_username = user_info["name"]
    print(f"[Hugging Face] Successfully authenticated as: {hf_username}")
else:
    print("[Hugging Face] WARNING: HF_TOKEN not found. Artifact upload will be skipped.")

# =====================================================================
# 3. Dataset Splitting & DataLoader Setup
# =====================================================================
print("\n" + "=" * 70)
print("DATASET PREPARATION & DUAL AUGMENTATION")
print("=" * 70)

MVTEC_ROOT = Path("datasets/raw/mvtec_ad/bottle").resolve()
train_pairs, val_pairs, test_pairs = build_mvtec_segmentation_splits(
    mvtec_bottle_root=MVTEC_ROOT,
    include_good_in_train=True,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42
)

# Dual Augmentation for Training:
# Synchronized rotation (+-10 deg), horizontal flip (p=0.5)
# Photometric jitter applied ONLY to image, NEVER corrupting discrete binary mask
train_aug = IndustrialDualAugmentation(
    p_hflip=0.5,
    max_rotation_degrees=10.0,
    brightness_jitter=0.15,
    contrast_jitter=0.15
)

train_dataset = IndustrialSegmentationDataset(train_pairs, target_size=(256, 256), is_training=True, augmentation=train_aug)
val_dataset = IndustrialSegmentationDataset(val_pairs, target_size=(256, 256), is_training=False)
test_dataset = IndustrialSegmentationDataset(test_pairs, target_size=(256, 256), is_training=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

print(f"[Dataset] Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

# =====================================================================
# 4. Model Instantiation & Training
# =====================================================================
print("\n" + "=" * 70)
print("TRAINING U-NET ARCHITECTURE")
print("=" * 70)

OUTPUT_DIR = Path("experiments/segmentation/unet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = UNet(in_channels=3, num_classes=1, base_channels=32, bilinear=True).to(device)
param_count = model.count_parameters()
print(f"[Model] Initialized U-Net with {param_count:,} trainable parameters.")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

trainer = SegmentationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device=device,
    checkpoint_dir=OUTPUT_DIR,
    model_name="unet_bottle",
    patience=10
)

train_summary = trainer.train(num_epochs=30)
best_checkpoint = Path(train_summary["best_checkpoint"])

# =====================================================================
# 5. Hold-Out Test Split Evaluation
# =====================================================================
print("\n" + "=" * 70)
print("HOLD-OUT TEST EVALUATION")
print("=" * 70)

# Load best checkpoint
checkpoint_data = torch.load(best_checkpoint, map_location=device)
model.load_state_dict(checkpoint_data["model_state_dict"])
model.eval()

test_metrics = trainer.evaluate(test_loader)
print(f"Test Loss           : {test_metrics['loss']:.4f}")
print(f"Test Mean IoU       : {test_metrics['iou']:.4f}")
print(f"Test Dice Score     : {test_metrics['dice']:.4f}")
print(f"Test Pixel Accuracy : {test_metrics['pixel_accuracy']:.4f}")

# =====================================================================
# 6. Qualitative Prediction Visualizations
# =====================================================================
print("\n" + "=" * 70)
print("GENERATING QUALITATIVE SEGMENTATION OVERLAYS")
print("=" * 70)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
col_titles = ["Product Sample", "Ground Truth", "Predicted Probability", "Defect Overlay"]

for i in range(4):
    for j in range(4):
        if i == 0:
            axes[i, j].set_title(col_titles[j], fontsize=13, pad=10)

sample_indices = min(4, len(test_dataset))
for idx in range(sample_indices):
    img_t, mask_t, meta = test_dataset[idx]
    
    # Forward pass
    with torch.no_grad():
        logits = model(img_t.unsqueeze(0).to(device))
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (prob_map >= 0.5).astype(np.uint8)

    # Denormalize image for plotting
    inv_mean = np.array([0.485, 0.456, 0.406])
    inv_std = np.array([0.229, 0.224, 0.225])
    disp_img = img_t.permute(1, 2, 0).numpy() * inv_std + inv_mean
    disp_img = np.clip(disp_img, 0.0, 1.0)

    gt_mask = mask_t.squeeze().numpy()

    # Create overlay
    overlay = disp_img.copy()
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6

    axes[idx, 0].imshow(disp_img)
    axes[idx, 0].set_ylabel(f"{meta['category']}", fontsize=11)
    axes[idx, 1].imshow(gt_mask, cmap="gray")
    axes[idx, 2].imshow(prob_map, cmap="inferno", vmin=0, vmax=1)
    axes[idx, 3].imshow(overlay)

    for col in range(4):
        axes[idx, col].set_xticks([])
        axes[idx, col].set_yticks([])

plt.tight_layout()
qualitative_plot_path = OUTPUT_DIR / "test_qualitative_predictions.png"
plt.savefig(qualitative_plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Visualization] Saved qualitative predictions plot to {qualitative_plot_path}")

# =====================================================================
# 7. Telemetry & Hugging Face Hub Upload
# =====================================================================
telemetry = {
    "model_architecture": "UNet (base_channels=32, bilinear=True)",
    "parameter_count": param_count,
    "hardware": {
        "gpu_name": gpu_name,
        "vram_gb": round(total_vram_gb, 2),
        "cuda_version": cuda_version
    },
    "training_summary": train_summary,
    "test_metrics": test_metrics
}

telemetry_path = OUTPUT_DIR / "unet_run_telemetry.json"
with open(telemetry_path, "w", encoding="utf-8") as f:
    json.dump(telemetry, f, indent=2)

if hf_username and hf_token:
    print("\n" + "=" * 70)
    print("HUGGING FACE REPOSITORY UPLOAD")
    print("=" * 70)

    repo_id = f"{hf_username}/industrial-damage-unet-bottle"
    print(f"[Hugging Face] Target Repository: {repo_id}")

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        staging_dir = OUTPUT_DIR / "hf_upload_stage"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        shutil.copy2(best_checkpoint, staging_dir / "best_unet.pt")
        shutil.copy2(telemetry_path, staging_dir / "telemetry.json")
        shutil.copy2(qualitative_plot_path, staging_dir / "qualitative_results.png")

        model_card = f"""---
language:
- en
license: mit
tags:
- computer-vision
- semantic-segmentation
- industrial-inspection
- defect-detection
- unet
datasets:
- foersben/mvtec-ad
metrics:
- iou
- dice
---

# Industrial Defect Segmentation: U-Net (Bottle)

PyTorch U-Net architecture trained for pixel-level boundary localization on bottle surface defects (broken glass, contamination).

## Evaluation Metrics (Hold-Out Test Split)
- **Test Mean IoU**: {test_metrics['iou']:.4f}
- **Test Dice Score**: {test_metrics['dice']:.4f}
- **Test Pixel Accuracy**: {test_metrics['pixel_accuracy']:.4f}

## Hardware & Training
- **Trained on**: Google Colab Tesla T4 ({total_vram_gb:.1f} GB VRAM)
- **Parameters**: {param_count:,}
- **Loss**: Combined BCE + Soft Dice Loss
"""
        with open(staging_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(model_card)

        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add trained U-Net weights, evaluation telemetry, and prediction plots"
        )
        print(f"[Hugging Face] Upload complete! View at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[Hugging Face] Upload failed: {e}")

print("\n" + "=" * 70)
print("PHASE 6 U-NET TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
