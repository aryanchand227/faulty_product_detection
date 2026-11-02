"""
Google Colab Training Harness for Classification Models:
1. SimpleCNN (Baseline from Scratch)
2. ResNet-18 (Transfer Learning with Skip Connections)
3. EfficientNet-B0 (Compound Scaling & Depthwise Convolutions)

Run this script in Google Colab with GPU runtime (Tesla T4).
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt

# 1. GPU Gatekeeper Check
print("=" * 70)
print("PHASE 4: SUPERVISED CLASSIFICATION EXPERIMENTS (COLAB GPU)")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is required for model training! Go to Runtime -> Change runtime type -> T4 GPU."
    )

device = torch.device("cuda")
print(f"[Hardware] GPU Detected: {torch.cuda.get_device_name(0)}")
print(f"[Hardware] VRAM: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")
print(f"[Hardware] CUDA Version: {torch.version.cuda}")

# 2. Add Project Root to Path
PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.models.classification.simple_cnn import SimpleCNN
from src.models.classification.resnet_classifier import ResNetClassifier
from src.models.classification.efficientnet_classifier import EfficientNetClassifier
from src.training.train_classifier import ClassificationTrainer
from src.evaluation.classification_metrics import compute_classification_metrics, plot_confusion_matrix
from src.data.neu_dataset import NEU_CLASSES, NEUSurfaceDataset, download_neu_surface_dataset

CHECKPOINT_DIR = Path("./models/classification")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("./docs/assets/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 25,
    lr: float = 1e-3,
    is_transfer_learning: bool = False
) -> Dict[str, Union[float, str, int]]:
    """
    Trains a model, logs training metrics, and evaluates on held-out test split.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {model_name}")
    print(f"{'='*60}")

    param_info = model.get_num_parameters()
    print(f"Total Parameters:     {param_info['total_parameters']:,}")
    print(f"Trainable Parameters: {param_info['trainable_parameters']:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        use_amp=True,
        checkpoint_dir=CHECKPOINT_DIR,
        model_name=model_name
    )

    t_start = time.time()

    if is_transfer_learning:
        # Two-Stage Fine-Tuning:
        # Stage 1: Train classification head for 5 epochs with frozen backbone
        print("\n[Transfer Learning] Stage 1: Training classifier head (Backbone Frozen)...")
        model.freeze_backbone()
        trainer.fit(num_epochs=5, early_stopping_patience=5, checkpoint_prefix="stage1")

        # Stage 2: Unfreeze backbone with a 10x lower learning rate
        print("\n[Transfer Learning] Stage 2: Fine-Tuning full backbone (Backbone Unfrozen)...")
        model.unfreeze_backbone()
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * 0.1  # Lower LR for fine-tuning
        trainer.fit(num_epochs=num_epochs - 5, early_stopping_patience=8, checkpoint_prefix="best")
    else:
        # Standard training from scratch
        trainer.fit(num_epochs=num_epochs, early_stopping_patience=8, checkpoint_prefix="best")

    total_training_time = time.time() - t_start

    # Load Best Checkpoint for Test Evaluation
    best_ckpt_path = CHECKPOINT_DIR / f"best_{model_name}.pt"
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n[Test] Loaded best checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Evaluate on Held-Out Test Set
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            latencies.append((time.time() - t0) / images.size(0))

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    test_metrics = compute_classification_metrics(all_labels, all_preds, y_prob=all_probs, class_names=list(NEU_CLASSES))
    mean_latency_ms = float(np.mean(latencies) * 1000.0)
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    print(f"\n[Test Results - {model_name}]")
    print(f"  Accuracy:         {test_metrics['accuracy']:.2%}")
    print(f"  Macro Precision:  {test_metrics['precision_macro']:.4f}")
    print(f"  Macro Recall:     {test_metrics['recall_macro']:.4f}")
    print(f"  Macro F1-Score:   {test_metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:      {test_metrics['f1_weighted']:.4f}")
    print(f"  Inference Latency: {mean_latency_ms:.2f} ms/image ({fps:.1f} FPS)")

    # Save Confusion Matrix Plot
    cm_path = RESULTS_DIR / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=list(NEU_CLASSES),
        title=f"Confusion Matrix: {model_name}",
        save_path=str(cm_path)
    )
    print(f"  Saved confusion matrix to: {cm_path}")

    return {
        "model_name": model_name,
        "total_params": param_info["total_parameters"],
        "trainable_params": param_info["trainable_parameters"],
        "training_time_sec": round(total_training_time, 1),
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "latency_ms": round(mean_latency_ms, 2),
        "fps": round(fps, 1)
    }


def main():
    # 1. Download / Load Dataset
    data_dir = download_neu_surface_dataset()
    full_dataset = NEUSurfaceDataset(root_dir=data_dir, split="train", img_size=224)

    if len(full_dataset) == 0:
        print("[Warning] No images found in standard structure; scanning raw folder...")
        full_dataset.root_dir = data_dir
        full_dataset._index_samples()

    print(f"Total catalogued images: {len(full_dataset)}")

    # 2. Stratified / Seeded Train-Val-Test Split (70% train, 15% val, 15% test)
    total_samples = len(full_dataset)
    train_len = int(0.70 * total_samples)
    val_len = int(0.15 * total_samples)
    test_len = total_samples - train_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_len, val_len, test_len], generator=generator
    )

    print(f"Split sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model 1: Baseline SimpleCNN
    model_cnn = SimpleCNN(num_classes=len(NEU_CLASSES), dropout=0.3)
    results_cnn = train_and_evaluate_model(
        model=model_cnn,
        model_name="SimpleCNN_Baseline",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=25,
        lr=1e-3,
        is_transfer_learning=False
    )

    # 4. Model 2: ResNet-18 Transfer Learning
    model_resnet = ResNetClassifier(num_classes=len(NEU_CLASSES), pretrained=True, dropout=0.3)
    results_resnet = train_and_evaluate_model(
        model=model_resnet,
        model_name="ResNet18_Transfer",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=25,
        lr=5e-4,
        is_transfer_learning=True
    )

    # 5. Model 3: EfficientNet-B0 Transfer Learning
    model_efficientnet = EfficientNetClassifier(num_classes=len(NEU_CLASSES), pretrained=True, dropout=0.3)
    results_enet = train_and_evaluate_model(
        model=model_efficientnet,
        model_name="EfficientNetB0_Transfer",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=25,
        lr=5e-4,
        is_transfer_learning=True
    )

    # 6. Consolidated Model Comparison Table
    all_results = [results_cnn, results_resnet, results_enet]
    comparison_df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("CONSOLIDATED CLASSIFICATION BENCHMARK COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Save to disk
    comparison_df.to_csv(RESULTS_DIR / "classification_model_comparison.csv", index=False)
    print(f"\n[OK] Results saved to: {RESULTS_DIR / 'classification_model_comparison.csv'}")


if __name__ == "__main__":
    main()

