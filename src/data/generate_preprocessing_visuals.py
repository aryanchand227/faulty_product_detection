"""
Generate Visual Artifacts for Preprocessing and Dual Augmentation Pipeline.

Produces side-by-side before-and-after visual comparisons:
1. Normalization & Inverse Normalization roundtrip fidelity.
2. Dual Augmentation tracking: verifies that spatial transforms lock
   the ground-truth defect mask to the physical defect across all augmentations.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import MVTEC_BOTTLE_DIR, DOCS_DIR
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

PREPROCESSING_ASSETS_DIR = DOCS_DIR / "assets" / "preprocessing"
PREPROCESSING_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_normalization_visual():
    """Generates visual comparison showing normalization and exact denormalization."""
    sample_img_path = next(MVTEC_BOTTLE_DIR.glob("train/good/*.png"))
    img_rgb = load_image(sample_img_path, target_size=(256, 256), color_space="RGB")
    tensor_raw = to_tensor(img_rgb)
    
    # Normalize with ImageNet stats
    tensor_norm = normalize_tensor(tensor_raw, IMAGENET_MEAN, IMAGENET_STD)
    
    # Reconstruct via denormalization
    tensor_reconstructed = denormalize_tensor(tensor_norm, IMAGENET_MEAN, IMAGENET_STD)
    img_reconstructed = tensor_to_numpy_image(tensor_reconstructed)
    
    # Pixel absolute difference
    abs_diff = np.abs(img_rgb.astype(np.float32) - img_reconstructed.astype(np.float32))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original RGB [0, 255]", fontsize=10)
    axes[0].axis("off")

    # Visualizing normalized channel 0 (R) with coolwarm colormap
    norm_vis = tensor_norm[0].numpy()
    im1 = axes[1].imshow(norm_vis, cmap="coolwarm")
    axes[1].set_title(f"2. Normalized Tensor (Red Ch)\nRange: [{norm_vis.min():.2f}, {norm_vis.max():.2f}]", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_reconstructed)
    axes[2].set_title("3. Reconstructed RGB\nvia denormalize_tensor()", fontsize=10)
    axes[2].axis("off")

    im3 = axes[3].imshow(abs_diff.mean(axis=2), cmap="magma")
    axes[3].set_title(f"4. Absolute Difference Map\nMax Error: {abs_diff.max():.4f}", fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle("Normalization & Invertibility Roundtrip Verification", fontsize=12)
    plt.tight_layout()
    plt.savefig(PREPROCESSING_ASSETS_DIR / "normalization_roundtrip.png", dpi=200)
    plt.close()
    print("[OK] Saved normalization_roundtrip.png")


def generate_dual_augmentation_gallery():
    """Generates gallery showing synchronized dual augmentation across image and mask."""
    sample_img_path = next(MVTEC_BOTTLE_DIR.glob("test/broken_large/000.png"))
    sample_mask_path = MVTEC_BOTTLE_DIR / "ground_truth" / "broken_large" / "000_mask.png"

    img_rgb = load_image(sample_img_path, target_size=(384, 384), color_space="RGB")
    img_tensor = to_tensor(img_rgb)
    mask_tensor = load_and_preprocess_mask(sample_mask_path, target_size=(384, 384))

    aug_pipeline = IndustrialDualAugmentation(
        target_size=(384, 384),
        mode="train",
        hflip_prob=0.5,
        max_rotation_degrees=12.0,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        normalize=False  # Keep unnormalized for visual plotting
    )

    fig, axes = plt.subplots(4, 3, figsize=(11, 13))

    def make_overlay(img_t, mask_t):
        img_np = tensor_to_numpy_image(img_t)
        mask_np = (mask_t[0].numpy() > 0).astype(np.uint8)
        overlay = img_np.copy()
        overlay[mask_np > 0] = [255, 30, 30]
        blended = cv2.addWeighted(img_np, 0.65, overlay, 0.35, 0)
        return img_np, mask_np, blended

    # Row 0: Original Pair
    orig_img, orig_mask, orig_overlay = make_overlay(img_tensor, mask_tensor)
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title("Original Image (384x384)", fontsize=9)
    axes[0, 0].axis("off")
    axes[0, 1].imshow(orig_mask, cmap="gray")
    axes[0, 1].set_title("Original Defect Mask", fontsize=9)
    axes[0, 1].axis("off")
    axes[0, 2].imshow(orig_overlay)
    axes[0, 2].set_title("Original Synchronized Overlay", fontsize=9)
    axes[0, 2].axis("off")

    # Rows 1-3: Three stochastic augmented variants
    torch.manual_seed(101)
    for row_idx in range(1, 4):
        aug_img, aug_mask = aug_pipeline(img_tensor.clone(), mask_tensor.clone())
        a_img, a_mask, a_overlay = make_overlay(aug_img, aug_mask)

        axes[row_idx, 0].imshow(a_img)
        axes[row_idx, 0].set_title(f"Augmented Image #{row_idx}", fontsize=9)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(a_mask, cmap="gray")
        axes[row_idx, 1].set_title(f"Augmented Mask #{row_idx}", fontsize=9)
        axes[row_idx, 1].axis("off")
        axes[row_idx, 2].imshow(a_overlay)
        axes[row_idx, 2].set_title(f"Synchronized Overlay #{row_idx}", fontsize=9)
        axes[row_idx, 2].axis("off")

    plt.suptitle("Industrial Dual Augmentation: Spatial Synchronization of Image & Defect Mask", fontsize=12)
    plt.tight_layout()
    plt.savefig(PREPROCESSING_ASSETS_DIR / "dual_augmentation_samples.png", dpi=200)
    plt.close()
    print("[OK] Saved dual_augmentation_samples.png")


if __name__ == "__main__":
    generate_normalization_visual()
    generate_dual_augmentation_gallery()

