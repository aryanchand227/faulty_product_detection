"""
Unit tests for Model B: NEU Steel Weakly-Supervised Segmentation Dataset.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.steel_segmentation_dataset import NEUSteelSegmentationDataset, build_neu_steel_segmentation_pairs


def test_steel_segmentation_dataset_getitem(tmp_path):
    import cv2

    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Create dummy steel image (200x200x3)
    img_path = img_dir / "crazing_1.jpg"
    cv2.imwrite(str(img_path), np.ones((200, 200, 3), dtype=np.uint8) * 150)

    # Create dummy pseudo-mask (200x200)
    mask_path = mask_dir / "crazing_1_mask.png"
    cv2.imwrite(str(mask_path), np.zeros((200, 200), dtype=np.uint8))

    pairs = [(img_path, mask_path, "crazing")]
    dataset = NEUSteelSegmentationDataset(pairs, target_size=(224, 224), is_training=False)

    assert len(dataset) == 1
    img_t, mask_t, meta = dataset[0]

    assert img_t.shape == (3, 224, 224)
    assert mask_t.shape == (1, 224, 224)
    assert meta["category"] == "crazing"
    assert meta["supervision_type"] == "weak_pseudo_label"
