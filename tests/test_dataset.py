"""
Unit Tests for MVTec Dataset Loader and PyTorch DataLoader integration.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.config import MVTEC_BOTTLE_DIR
from src.data.dataset import MVTecDataset

def test_dataset_train_split():
    """Verify train split loading: exactly 209 samples, all normal."""
    dataset = MVTecDataset(category_dir=MVTEC_BOTTLE_DIR, split="train", img_size=256)
    assert len(dataset) == 209, f"Expected 209 train samples, got {len(dataset)}"
    
    # Check first sample properties
    sample = dataset[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["label"] == 0
    assert sample["defect_type"] == "good"
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].sum().item() == 0.0  # Normal sample has empty mask

def test_dataset_test_split():
    """Verify test split loading: exactly 83 samples (20 good, 63 defective)."""
    dataset = MVTecDataset(category_dir=MVTEC_BOTTLE_DIR, split="test", img_size=256)
    assert len(dataset) == 83, f"Expected 83 test samples, got {len(dataset)}"
    
    # Count normal vs defective
    labels = [dataset[i]["label"] for i in range(len(dataset))]
    assert labels.count(0) == 20, "Expected 20 good samples in test set"
    assert labels.count(1) == 63, "Expected 63 anomalous samples in test set"

def test_defective_sample_mask():
    """Verify that defective sample returns a valid non-zero mask."""
    dataset = MVTecDataset(category_dir=MVTEC_BOTTLE_DIR, split="test", img_size=256)
    
    # Find an anomalous sample
    anom_idx = None
    for idx, s in enumerate(dataset.samples):
        if s["is_anomaly"] == 1:
            anom_idx = idx
            break
            
    assert anom_idx is not None
    sample = dataset[anom_idx]
    assert sample["label"] == 1
    assert sample["mask"].sum().item() > 0, "Defective sample must have non-empty mask"
    # Verify discrete values: only 0.0 or 1.0
    unique_vals = torch.unique(sample["mask"]).tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0), f"Mask contained non-binary value: {v}"

def test_dataloader_batching():
    """Verify that MVTecDataset integrates seamlessly with torch DataLoader."""
    dataset = MVTecDataset(category_dir=MVTEC_BOTTLE_DIR, split="test", img_size=128)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    batch = next(iter(loader))
    assert batch["image"].shape == (8, 3, 128, 128)
    assert batch["mask"].shape == (8, 1, 128, 128)
    assert batch["label"].shape == (8,)
    assert len(batch["defect_type"]) == 8

