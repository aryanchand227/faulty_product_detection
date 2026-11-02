"""
Segmentation Dataset Module for Industrial Product Defect Inspection.

Pairs product images with pixel-level ground-truth binary defect masks.
Includes:
- MVTec AD bottle defect image-mask pairing
- Support for good (non-defective) samples paired with all-zero black masks
- Deterministic stratified train/val/test splitting
- Integration with IndustrialDualAugmentation preserving nearest-neighbor mask discrete values
"""

import logging
from pathlib import Path
import random
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


class IndustrialSegmentationDataset(Dataset):
    """
    PyTorch Dataset for pixel-level semantic defect segmentation.

    Pairs RGB images with single-channel binary masks (1 = defect, 0 = normal surface).
    """

    def __init__(
        self,
        samples: List[Tuple[Path, Optional[Path], str]],
        target_size: Tuple[int, int] = (256, 256),
        is_training: bool = False,
        augmentation: Optional[IndustrialDualAugmentation] = None
    ) -> None:
        """
        Args:
            samples: List of tuples (image_path, mask_path_or_none, defect_category).
            target_size: (H, W) tuple for resizing.
            is_training: If True and augmentation provided, applies stochastic dual transforms.
            augmentation: IndustrialDualAugmentation instance.
        """
        self.samples = samples
        self.target_size = target_size
        self.is_training = is_training
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, int]]]:
        img_path, mask_path, category = self.samples[idx]

        # 1. Load image (RGB)
        img_np = load_image(img_path, target_size=self.target_size)

        # 2. Load or construct ground-truth mask
        if mask_path is not None and Path(mask_path).exists():
            mask_np = load_and_preprocess_mask(mask_path, target_size=self.target_size)
        else:
            # Defect-free / normal sample -> all-zero black mask
            mask_np = np.zeros(self.target_size, dtype=np.uint8)

        # 3. Apply Dual Augmentation (synchronized spatial transforms, isolated photometric transforms)
        if self.is_training and self.augmentation is not None:
            img_np, mask_np = self.augmentation(img_np, mask_np)

        # 4. Convert to normalized tensors
        img_tensor = to_tensor(img_np)
        img_tensor = normalize_tensor(img_tensor)

        # Mask tensor: shape (1, H, W), float {0.0, 1.0}
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        metadata = {
            "image_path": str(img_path),
            "category": category,
            "is_defective": int(mask_path is not None)
        }

        return img_tensor, mask_tensor, metadata


def build_mvtec_segmentation_splits(
    mvtec_bottle_root: Union[str, Path] = "datasets/raw/mvtec_ad/bottle",
    include_good_in_train: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Tuple[Path, Optional[Path], str]], List[Tuple[Path, Optional[Path], str]], List[Tuple[Path, Optional[Path], str]]]:
    """
    Discovers MVTec bottle samples and constructs strict train, validation, and test splits.

    Guarantees:
    - Balanced representation across defect categories (broken_large, broken_small, contamination).
    - Zero data leakage between partitions.
    """
    root = Path(mvtec_bottle_root).resolve()
    test_dir = root / "test"
    gt_dir = root / "ground_truth"
    train_dir = root / "train" / "good"

    if not test_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"MVTec AD bottle directory not found at: {root}")

    rng = random.Random(seed)

    defective_by_cat: Dict[str, List[Tuple[Path, Path, str]]] = {}
    defect_categories = ["broken_large", "broken_small", "contamination"]

    for cat in defect_categories:
        cat_img_dir = test_dir / cat
        cat_gt_dir = gt_dir / cat
        pairs: List[Tuple[Path, Path, str]] = []

        for img_p in sorted(cat_img_dir.glob("*.png")):
            # Mask filename format: 000_mask.png
            mask_p = cat_gt_dir / f"{img_p.stem}_mask.png"
            if mask_p.exists():
                pairs.append((img_p, mask_p, cat))
            else:
                logger.warning(f"Missing mask for image: {img_p}")

        rng.shuffle(pairs)
        defective_by_cat[cat] = pairs

    train_samples: List[Tuple[Path, Optional[Path], str]] = []
    val_samples: List[Tuple[Path, Optional[Path], str]] = []
    test_samples: List[Tuple[Path, Optional[Path], str]] = []

    # Stratified split per defect category
    for cat, pairs in defective_by_cat.items():
        n = len(pairs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_samples.extend(pairs[:n_train])
        val_samples.extend(pairs[n_train:n_train + n_val])
        test_samples.extend(pairs[n_train + n_val:])

    # Optionally include normal good samples to teach U-Net negative background suppression
    if include_good_in_train and train_dir.exists():
        good_imgs = sorted(list(train_dir.glob("*.png")))
        rng.shuffle(good_imgs)

        # Take a balanced fraction of good samples
        n_good_train = len(train_samples) // 2
        for g_img in good_imgs[:n_good_train]:
            train_samples.append((g_img, None, "good"))

    # Also include test/good samples in val and test
    test_good_dir = test_dir / "good"
    if test_good_dir.exists():
        test_good_imgs = sorted(list(test_good_dir.glob("*.png")))
        rng.shuffle(test_good_imgs)
        val_samples.extend([(p, None, "good") for p in test_good_imgs[:len(test_good_imgs) // 2]])
        test_samples.extend([(p, None, "good") for p in test_good_imgs[len(test_good_imgs) // 2:]])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    logger.info(
        f"Built Segmentation Splits -> Train: {len(train_samples)}, "
        f"Val: {len(val_samples)}, Test: {len(test_samples)}"
    )

    return train_samples, val_samples, test_samples
