"""
PatchCore Training and Evaluation Pipeline.

Fits the nominal prototype memory bank on normal training images
and evaluates on the official MVTec AD Bottle benchmark.

Guarantees:
- Strict zero-leakage benchmark separation: Memory bank is built ONLY from
  datasets/raw/mvtec_ad/bottle/train/good/ (209 normal images).
- Test evaluation uses official hold-out partition: 20 good images + 63 defect images.
- Reports Image AUROC, Pixel AUROC, AUPRO, and F1-optimal threshold.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import json
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_image,
    normalize_tensor,
    to_tensor,
)
from src.evaluation.anomaly_metrics import evaluate_anomaly_detection
from src.models.anomaly_detection.patchcore import PatchCore


class NormalBottleDataset(Dataset):
    """Dataset providing normal training images for memory bank construction."""

    def __init__(self, image_paths: List[Path], target_size: Tuple[int, int] = (224, 224)) -> None:
        self.image_paths = sorted(image_paths)
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_p = self.image_paths[idx]
        img_np = load_image(img_p, target_size=self.target_size)
        img_t = to_tensor(img_np)
        return normalize_tensor(img_t, IMAGENET_MEAN, IMAGENET_STD)


class MVTecBottleTestDataset(Dataset):
    """Official MVTec Bottle test benchmark dataset."""

    def __init__(
        self,
        category_dir: Union[str, Path] = "datasets/raw/mvtec_ad/bottle",
        target_size: Tuple[int, int] = (224, 224)
    ) -> None:
        self.category_dir = Path(category_dir)
        self.target_size = target_size
        self.samples: List[Tuple[Path, Optional[Path], int, str]] = []  # (img_p, mask_p, is_defect, defect_type)

        test_dir = self.category_dir / "test"
        gt_dir = self.category_dir / "ground_truth"

        if not test_dir.exists():
            raise FileNotFoundError(f"MVTec test directory not found: {test_dir}")

        for folder in sorted(test_dir.iterdir()):
            if not folder.is_dir():
                continue
            defect_type = folder.name
            is_defect = 0 if defect_type == "good" else 1

            for img_p in sorted(folder.glob("*.png")):
                mask_p = None
                if is_defect:
                    mask_candidate = gt_dir / defect_type / f"{img_p.stem}_mask.png"
                    if mask_candidate.exists():
                        mask_p = mask_candidate
                self.samples.append((img_p, mask_p, is_defect, defect_type))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray, int, str]:
        img_p, mask_p, is_defect, defect_type = self.samples[idx]

        # Load image
        img_np = load_image(img_p, target_size=self.target_size)
        img_t = to_tensor(img_np)
        img_norm = normalize_tensor(img_t, IMAGENET_MEAN, IMAGENET_STD)

        # Load ground truth mask
        w, h = self.target_size
        if mask_p is not None and mask_p.exists():
            mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 0).astype(np.uint8)
        else:
            mask_bin = np.zeros((h, w), dtype=np.uint8)

        return img_norm, mask_bin, is_defect, defect_type


def run_patchcore_training(
    mvtec_bottle_root: Union[str, Path] = "datasets/raw/mvtec_ad/bottle",
    backbone_name: str = "resnet18",
    coreset_ratio: float = 0.10,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    device: str = "cpu",
    save_path: Optional[Union[str, Path]] = "models/checkpoints/patchcore_bottle_resnet18.pt"
) -> Dict[str, Union[float, int, str]]:
    """
    Fits and evaluates PatchCore on MVTec Bottle benchmark.
    """
    root = Path(mvtec_bottle_root)
    train_good_dir = root / "train" / "good"

    if not train_good_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_good_dir}")

    train_paths = sorted(list(train_good_dir.glob("*.png")))
    print(f"[Phase 7] Found {len(train_paths)} normal training images in {train_good_dir}")

    train_dataset = NormalBottleDataset(train_paths, target_size=target_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    print(f"[Phase 7] Initializing PatchCore with backbone '{backbone_name}' on {device}...")
    model = PatchCore(
        backbone_name=backbone_name,
        coreset_ratio=coreset_ratio,
        device=device
    )

    # Fit memory bank
    start_time = time.time()
    num_prototypes = model.fit(train_loader, verbose=True)
    fit_duration_sec = round(time.time() - start_time, 2)
    print(f"[Phase 7] Fitting completed in {fit_duration_sec}s. Prototypes stored: {num_prototypes}")

    # Evaluate on official test benchmark
    test_dataset = MVTecBottleTestDataset(category_dir=root, target_size=target_size)
    print(f"[Phase 7] Evaluating on {len(test_dataset)} official test benchmark samples...")

    y_true_labels: List[int] = []
    y_image_scores: List[float] = []
    gt_masks: List[np.ndarray] = []
    anomaly_maps: List[np.ndarray] = []

    model.backbone.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img_t, mask_bin, is_defect, defect_type = test_dataset[i]
            img_batch = img_t.unsqueeze(0).to(device)

            patch_dists, img_scores, spatial_hw = model.predict_patches(img_batch)
            score = float(img_scores[0].item())

            a_map = model.compute_anomaly_maps(
                patch_dists, spatial_hw, output_hw=target_size, sigma=4.0
            )[0]

            y_true_labels.append(is_defect)
            y_image_scores.append(score)
            gt_masks.append(mask_bin)
            anomaly_maps.append(a_map)

    # Compute comprehensive evaluation metrics
    eval_result = evaluate_anomaly_detection(
        y_true_labels=y_true_labels,
        y_image_scores=y_image_scores,
        ground_truth_masks=gt_masks,
        anomaly_maps=anomaly_maps,
        max_fpr=0.30
    )

    model.threshold = eval_result.optimal_threshold

    print("\n" + "=" * 50)
    print("           PATCHCORE BENCHMARK RESULTS           ")
    print("=" * 50)
    print(f"Backbone:               {backbone_name}")
    print(f"Coreset Ratio:          {coreset_ratio * 100:.1f}%")
    print(f"Image AUROC:            {eval_result.image_auroc:.4f}")
    print(f"Pixel AUROC:            {eval_result.pixel_auroc:.4f}")
    print(f"AUPRO (FPR <= 0.30):    {eval_result.aupro:.4f}")
    print(f"Optimal Threshold:      {eval_result.optimal_threshold:.4f}")
    print(f"F1-Score:               {eval_result.f1_score:.4f}")
    print(f"Precision:              {eval_result.precision:.4f}")
    print(f"Recall:                 {eval_result.recall:.4f}")
    print(f"Confusion Matrix:       {eval_result.confusion_matrix}")
    print("=" * 50 + "\n")

    # Save model checkpoint
    if save_path:
        saved_file = model.save(save_path)
        print(f"[Phase 7] Checkpoint saved successfully to: {saved_file}")

    return {
        "backbone": backbone_name,
        "coreset_ratio": coreset_ratio,
        "prototypes": num_prototypes,
        "fit_duration_sec": fit_duration_sec,
        "image_auroc": eval_result.image_auroc,
        "pixel_auroc": eval_result.pixel_auroc,
        "aupro": eval_result.aupro if eval_result.aupro is not None else 0.0,
        "optimal_threshold": eval_result.optimal_threshold,
        "f1_score": eval_result.f1_score,
        "precision": eval_result.precision,
        "recall": eval_result.recall,
        "confusion_matrix": eval_result.confusion_matrix
    }


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore Anomaly Detection")
    parser.add_argument("--root", type=str, default="datasets/raw/mvtec_ad/bottle", help="MVTec Bottle root")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50", "wide_resnet50_2"])
    parser.add_argument("--coreset-ratio", type=float, default=0.10, help="Fraction of patch memory bank to retain")
    parser.add_argument("--target-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="models/checkpoints/patchcore_bottle_resnet18.pt")
    args = parser.parse_args()

    run_patchcore_training(
        mvtec_bottle_root=args.root,
        backbone_name=args.backbone,
        coreset_ratio=args.coreset_ratio,
        target_size=(args.target_size, args.target_size),
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
