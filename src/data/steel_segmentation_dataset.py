"""
Model B: NEU Steel Weakly-Supervised Segmentation Dataset.

Provides a completely isolated dataset pipeline for steel surface defect segmentation:
- Domain: NEU Steel Strips (6 defect categories)
- Supervision: Weakly-Supervised Pseudo-Labels derived from Pascal VOC bounding boxes
  and localized Otsu intensity refinement.
- Preserves complete separation from Model A (MVTec Bottle U-Net).

ACADEMIC DEFENSE NOTICE:
-----------------------
This dataset uses WEAK PSEUDO-MASKS. It allows exploring weak-to-dense segmentation learning,
but does not represent human ground-truth pixel annotations. Evaluation metrics must be reported
specifically under the "Weakly-Supervised Steel Experiment" heading, never combined with
human-annotated benchmarks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import IndustrialDualAugmentation
from src.data.preprocessing import (
    load_image,
    load_and_preprocess_mask,
    to_tensor,
    normalize_tensor,
)

logger = logging.getLogger(__name__)


class NEUSteelSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Weakly-Supervised Steel Defect Segmentation (Model B).
    """

    def __init__(
        self,
        image_mask_pairs: List[Tuple[Path, Path, str]],
        target_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        augmentation: Optional[IndustrialDualAugmentation] = None
    ) -> None:
        self.image_mask_pairs = image_mask_pairs
        self.target_size = target_size
        self.is_training = is_training
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.image_mask_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, int]]]:
        img_path, mask_path, category = self.image_mask_pairs[idx]

        # 1. Load image (RGB)
        img_np = load_image(img_path, target_size=self.target_size)
        img_tensor = to_tensor(img_np)

        # 2. Load pseudo-mask (discrete nearest-neighbor resize)
        if mask_path is not None and Path(mask_path).exists():
            mask_tensor = load_and_preprocess_mask(mask_path, target_size=self.target_size)
        else:
            mask_tensor = load_and_preprocess_mask(None, target_size=self.target_size)

        # 3. Synchronized Dual Augmentation
        if self.is_training and self.augmentation is not None:
            img_tensor, mask_tensor = self.augmentation(img_tensor, mask_tensor)
        else:
            img_tensor = normalize_tensor(img_tensor)

        metadata = {
            "image_path": str(img_path),
            "category": category,
            "supervision_type": "weak_pseudo_label"
        }

        return img_tensor, mask_tensor, metadata


def build_neu_steel_segmentation_pairs(
    yolo_dir: Union[str, Path] = "datasets/annotations/yolo",
    pseudo_mask_dir: Union[str, Path] = "datasets/processed/neu_pseudo_masks",
    split: str = "train"
) -> List[Tuple[Path, Path, str]]:
    """
    Pairs steel images from the YOLO split directory with generated weak pseudo-masks.

    Returns:
        List[Tuple[Path, Path, str]]: [(image_path, pseudo_mask_path, defect_class), ...]
    """
    yolo_p = Path(yolo_dir).resolve()
    mask_p = Path(pseudo_mask_dir).resolve()

    img_dir = yolo_p / "images" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Split images directory not found at: {img_dir}")

    pairs: List[Tuple[Path, Path, str]] = []

    for img_file in sorted(img_dir.glob("*.jpg")):
        stem = img_file.stem
        # Extract class prefix from stem (e.g. crazing_100 -> crazing)
        category = stem.rsplit("_", 1)[0]
        mask_file = mask_p / f"{stem}_mask.png"

        pairs.append((img_file, mask_file, category))

    logger.info(f"Discovered {len(pairs)} steel segmentation pairs for split '{split}'")
    return pairs
