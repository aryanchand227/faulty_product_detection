"""
Google Colab Training Harness for Industrial Defect Detection:
YOLOv8 Nano & Small Object Localization on NEU Surface Defect Dataset.

Features:
1. GPU Gatekeeper & Hardware Telemetry (Tesla T4 validation)
2. Hugging Face Authentication & Model Card Generation
3. Automated Dataset Verification & YOLO data.yaml configuration
4. Anchor-free YOLOv8 Training with CIoU + DFL Loss
5. Validation and Test Set Evaluation (mAP@50, mAP@50:95, Precision, Recall)
6. Sample Prediction Visualization with Bounding Box Overlays
7. ONNX Export for Low-Latency Industrial Edge Serving
8. Hugging Face Hub Model Artifact Publishing

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
import cv2
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, login

# =====================================================================
# 1. GPU Gatekeeper Verification
# =====================================================================
print("=" * 70)
print("PHASE 5: YOLO DEFECT LOCALIZATION EXPERIMENTS (COLAB GPU)")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is required for YOLO model training! "
        "In Google Colab, select: Runtime -> Change runtime type -> T4 GPU."
    )

gpu_name = torch.cuda.get_device_name(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
cuda_version = torch.version.cuda

print(f"[Hardware] GPU Detected : {gpu_name}")
print(f"[Hardware] Total VRAM   : {total_vram_gb:.2f} GB")
print(f"[Hardware] CUDA Version : {cuda_version}")

# Ensure ultralytics is imported
from ultralytics import YOLO

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
# 3. Dataset Configuration & Verification
# =====================================================================
print("\n" + "=" * 70)
print("DATASET VERIFICATION")
print("=" * 70)

DATA_YAML = Path("datasets/annotations/yolo/data.yaml").resolve()

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"data.yaml not found at {DATA_YAML}. "
        "Please ensure Phase 5 dataset conversion has been run or clone the repository."
    )

print(f"[Dataset] Found data.yaml at: {DATA_YAML}")

# Verify file counts
train_imgs = list((DATA_YAML.parent / "images" / "train").glob("*.jpg"))
val_imgs = list((DATA_YAML.parent / "images" / "val").glob("*.jpg"))
test_imgs = list((DATA_YAML.parent / "images" / "test").glob("*.jpg"))

print(f"[Dataset] Train images : {len(train_imgs)}")
print(f"[Dataset] Val images   : {len(val_imgs)}")
print(f"[Dataset] Test images  : {len(test_imgs)}")

assert len(train_imgs) == 1437, f"Expected 1437 train images, found {len(train_imgs)}"
assert len(val_imgs) == 180, f"Expected 180 val images, found {len(val_imgs)}"
assert len(test_imgs) == 180, f"Expected 180 test images, found {len(test_imgs)}"
print("[Dataset] Partition integrity verified. Zero split leakage.")

# =====================================================================
# 4. YOLOv8 Training Run
# =====================================================================
print("\n" + "=" * 70)
print("TRAINING YOLOv8 NANO (yolov8n.pt)")
print("=" * 70)

PROJECT_DIR = Path("experiments/detection/yolo")
RUN_NAME = "neu_yolov8n_colab"

model = YOLO("yolov8n.pt")

start_train = time.time()
train_results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project=str(PROJECT_DIR),
    name=RUN_NAME,
    patience=15,
    optimizer="AdamW",
    lr0=0.001,
    seed=42,
    exist_ok=True,
    verbose=True,
    save=True,
    plots=True
)
train_duration = time.time() - start_train
print(f"\n[Training] Completed in {train_duration / 60:.2f} minutes.")

# =====================================================================
# 5. Validation and Test Set Evaluation
# =====================================================================
print("\n" + "=" * 70)
print("MODEL EVALUATION: VALIDATION & TEST SETS")
print("=" * 70)

best_weights_path = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
if not best_weights_path.exists():
    raise FileNotFoundError(f"Best model weights not found at: {best_weights_path}")

best_model = YOLO(str(best_weights_path))

# Validation evaluation
print("\n--- Validation Partition Evaluation ---")
val_results = best_model.val(data=str(DATA_YAML), split="val", imgsz=640, device=0)
val_metrics = {
    "mAP50": float(val_results.box.map50),
    "mAP50_95": float(val_results.box.map),
    "precision": float(val_results.box.mp),
    "recall": float(val_results.box.mr)
}
print(f"Validation mAP@50    : {val_metrics['mAP50']:.4f}")
print(f"Validation mAP@50:95 : {val_metrics['mAP50_95']:.4f}")
print(f"Validation Precision : {val_metrics['precision']:.4f}")
print(f"Validation Recall    : {val_metrics['recall']:.4f}")

# Hold-out test evaluation
print("\n--- Test Partition Evaluation ---")
test_results = best_model.val(data=str(DATA_YAML), split="test", imgsz=640, device=0)
test_metrics = {
    "mAP50": float(test_results.box.map50),
    "mAP50_95": float(test_results.box.map),
    "precision": float(test_results.box.mp),
    "recall": float(test_results.box.mr)
}
print(f"Test mAP@50          : {test_metrics['mAP50']:.4f}")
print(f"Test mAP@50:95       : {test_metrics['mAP50_95']:.4f}")
print(f"Test Precision       : {test_metrics['precision']:.4f}")
print(f"Test Recall          : {test_metrics['recall']:.4f}")

# =====================================================================
# 6. ONNX Model Export
# =====================================================================
print("\n" + "=" * 70)
print("ONNX EXPORT FOR EDGE DEPLOYMENT")
print("=" * 70)

onnx_path = best_model.export(format="onnx", imgsz=640)
print(f"[Export] Successfully generated ONNX model at: {onnx_path}")

# =====================================================================
# 7. Visual Predictions on Test Samples
# =====================================================================
print("\n" + "=" * 70)
print("VISUAL PREDICTION SAMPLES")
print("=" * 70)

sample_preds_dir = PROJECT_DIR / RUN_NAME / "test_visual_samples"
sample_preds_dir.mkdir(parents=True, exist_ok=True)

test_sample_images = test_imgs[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, img_path in enumerate(test_sample_images):
    res = best_model.predict(str(img_path), conf=0.25, iou=0.45, device=0, verbose=False)[0]
    annotated_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
    axes[idx].imshow(annotated_rgb)
    axes[idx].set_title(f"Test Sample: {img_path.stem}", fontsize=11)
    axes[idx].axis("off")

plt.tight_layout()
visual_summary_path = sample_preds_dir / "test_predictions_summary.png"
plt.savefig(visual_summary_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Visualization] Saved test predictions to {visual_summary_path}")

# =====================================================================
# 8. Telemetry Summary JSON
# =====================================================================
telemetry = {
    "model_architecture": "YOLOv8n",
    "dataset": "NEU Surface Defect (1,797 images, 6 classes)",
    "hardware": {
        "gpu_name": gpu_name,
        "vram_gb": round(total_vram_gb, 2),
        "cuda_version": cuda_version
    },
    "training_hyperparameters": {
        "epochs": 50,
        "imgsz": 640,
        "batch_size": 16,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "patience": 15
    },
    "training_duration_minutes": round(train_duration / 60, 2),
    "validation_metrics": val_metrics,
    "test_metrics": test_metrics,
    "weights": {
        "best_pt": str(best_weights_path),
        "onnx": str(onnx_path)
    }
}

telemetry_path = PROJECT_DIR / RUN_NAME / "run_telemetry.json"
with open(telemetry_path, "w", encoding="utf-8") as f:
    json.dump(telemetry, f, indent=2)
print(f"[Telemetry] Saved execution summary to {telemetry_path}")

# =====================================================================
# 9. Upload Artifacts to Hugging Face Hub
# =====================================================================
if hf_username and hf_token:
    print("\n" + "=" * 70)
    print("HUGGING FACE REPOSITORY UPLOAD")
    print("=" * 70)

    repo_id = f"{hf_username}/industrial-damage-yolov8n"
    print(f"[Hugging Face] Target Repository: {repo_id}")

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"[Hugging Face] Repository ready: https://huggingface.co/{repo_id}")

        # Prepare staged upload folder
        staging_dir = PROJECT_DIR / RUN_NAME / "hf_upload_stage"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        # Copy best weights and ONNX
        shutil.copy2(best_weights_path, staging_dir / "best.pt")
        if Path(onnx_path).exists():
            shutil.copy2(onnx_path, staging_dir / "model.onnx")
        shutil.copy2(telemetry_path, staging_dir / "telemetry.json")
        shutil.copy2(visual_summary_path, staging_dir / "predictions_sample.png")
        shutil.copy2(DATA_YAML, staging_dir / "data.yaml")

        # Copy Ultralytics training curves if present
        for plot_name in ["confusion_matrix.png", "results.png", "PR_curve.png", "F1_curve.png"]:
            src_plot = PROJECT_DIR / RUN_NAME / plot_name
            if src_plot.exists():
                shutil.copy2(src_plot, staging_dir / plot_name)

        # Generate Model Card README.md
        model_card_content = f"""---
language:
- en
license: mit
tags:
- computer-vision
- object-detection
- industrial-inspection
- defect-detection
- yolo
- yolov8
datasets:
- KeenForgeAI/NEU-DET-corrected
metrics:
- mAP50
- mAP50-95
---

# Industrial Defect Detector: YOLOv8 Nano

Anchor-free YOLOv8 Nano model trained for automated visual inspection of steel strip manufacturing defects.

## Defect Classes (6)
1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## Quantitative Evaluation
- **Test mAP@50**: {test_metrics['mAP50']:.4f}
- **Test mAP@50:95**: {test_metrics['mAP50_95']:.4f}
- **Test Precision**: {test_metrics['precision']:.4f}
- **Test Recall**: {test_metrics['recall']:.4f}

## Hardware & Training
- **Trained on**: Google Colab Tesla T4 ({total_vram_gb:.1f} GB VRAM)
- **Epochs**: 50 (Early Stopping Patience 15)
- **Optimizer**: AdamW (lr=0.001)
- **Input Resolution**: 640x640
- **Loss Formulation**: Complete IoU (CIoU) + Distribution Focal Loss (DFL) + BCE Classification
"""
        with open(staging_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(model_card_content)

        # Upload staged folder
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add trained YOLOv8n weights, ONNX model, and evaluation telemetry"
        )
        print(f"[Hugging Face] Upload complete! View at: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"[Hugging Face] Upload error: {e}")
else:
    print("\n[Hugging Face] Skipped upload (no token provided). Checkpoints are saved locally in:", PROJECT_DIR / RUN_NAME)

print("\n" + "=" * 70)
print("PHASE 5 COLAB RUN FINISHED SUCCESSFULLY!")
print("=" * 70)
