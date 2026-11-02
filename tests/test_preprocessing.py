"""
Unit Tests for Preprocessing and Dual Augmentation Pipeline.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from src.config import MVTEC_BOTTLE_DIR
from src.data.preprocessing import (
    load_image,
    to_tensor,
    normalize_tensor,
    denormalize_tensor,
    tensor_to_numpy_image,
    load_and_preprocess_mask,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from src.data.augmentation import IndustrialDualAugmentation


def test_load_image_rgb_vs_bgr():
    """Verify that RGB and BGR loading produce inverted R and B channels."""
    sample_img_path = next(MVTEC_BOTTLE_DIR.glob("train/good/*.png"))
    
    img_rgb = load_image(sample_img_path, target_size=(100, 100), color_space="RGB")
    img_bgr = load_image(sample_img_path, target_size=(100, 100), color_space="BGR")
    
    assert img_rgb.shape == (100, 100, 3)
    assert img_bgr.shape == (100, 100, 3)
    # Channel 0 of RGB should equal channel 2 of BGR
    assert np.array_equal(img_rgb[:, :, 0], img_bgr[:, :, 2])
    assert np.array_equal(img_rgb[:, :, 2], img_bgr[:, :, 0])


def test_normalization_and_denormalization_roundtrip():
    """Verify that denormalize(normalize(x)) reconstructs original tensor within 1e-5."""
    torch.manual_seed(42)
    original_tensor = torch.rand((3, 64, 64), dtype=torch.float32)
    
    normalized = normalize_tensor(original_tensor, IMAGENET_MEAN, IMAGENET_STD)
    reconstructed = denormalize_tensor(normalized, IMAGENET_MEAN, IMAGENET_STD)
    
    assert torch.allclose(original_tensor, reconstructed, atol=1e-5), (
        "Round-trip normalization failed to recover original pixel values."
    )


def test_mask_nearest_neighbor_preservation():
    """Verify that discrete masks retain strictly {0.0, 1.0} without fractional blurring."""
    sample_mask_path = next(MVTEC_BOTTLE_DIR.glob("ground_truth/broken_large/*.png"))
    
    mask_tensor = load_and_preprocess_mask(sample_mask_path, target_size=(128, 128))
    assert mask_tensor.shape == (1, 128, 128)
    
    unique_vals = torch.unique(mask_tensor).tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0), f"Fractional boundary value {v} found in mask."


def test_dual_augmentation_synchronization():
    """Verify spatial transforms apply identically to paired image and mask."""
    # Create asymmetric dummy image and mask
    img = torch.zeros((3, 64, 64), dtype=torch.float32)
    img[:, :, :32] = 1.0  # Left half bright
    
    mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    mask[:, :, :32] = 1.0  # Left half positive
    
    # Force horizontal flip (prob=1.0) with zero rotation and no normalization
    aug = IndustrialDualAugmentation(
        target_size=(64, 64),
        mode="train",
        hflip_prob=1.0,
        max_rotation_degrees=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        normalize=False
    )
    
    aug_img, aug_mask = aug(img, mask)
    
    # Both should now be on the right half
    assert aug_img[:, :, 32:].sum() > 0
    assert aug_img[:, :, :32].sum() == 0
    assert aug_mask[:, :, 32:].sum() > 0
    assert aug_mask[:, :, :32].sum() == 0


def test_validation_mode_is_deterministic():
    """Verify that val/test mode performs zero random jitter and is deterministic."""
    img = torch.rand((3, 128, 128), dtype=torch.float32)
    mask = (torch.rand((1, 128, 128)) > 0.5).float()
    
    aug = IndustrialDualAugmentation(target_size=(128, 128), mode="val", normalize=True)
    
    out1_img, out1_mask = aug(img, mask)
    out2_img, out2_mask = aug(img, mask)
    
    assert torch.equal(out1_img, out2_img), "Validation mode produced non-deterministic images."
    assert torch.equal(out1_mask, out2_mask), "Validation mode produced non-deterministic masks."

