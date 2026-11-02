"""
Dual-Transform Industrial Augmentation Pipeline.

Ensures:
1. Identical spatial transformations (rotations, flips, affine shifts) are applied
   synchronously to both product image and segmentation mask.
2. Photometric augmentations (brightness, contrast, blur) are applied ONLY to the image,
   preserving discrete {0.0, 1.0} labels on the mask.
3. Domain-specific physical validity:
   - Horizontal flips are allowed (lateral symmetry of upright bottles).
   - Vertical flips are strictly disabled (bottles do not travel upside-down on a conveyor).
"""

from typing import Dict, Optional, Tuple, Union
import random
import torch
import torchvision.transforms.functional as TF
from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, normalize_tensor


class IndustrialDualAugmentation:
    """
    Synchronized augmentation pipeline for paired (image, mask) samples.

    Args:
        target_size: Target (width, height) tuple (e.g. (256, 256), (384, 384), (512, 512)).
        mode: 'train', 'val', or 'test'.
        hflip_prob: Probability of horizontal flip.
        max_rotation_degrees: Maximum angle for subtle rotational jitter (+/- deg).
        brightness_jitter: Maximum factor for brightness variation (+/-).
        contrast_jitter: Maximum factor for contrast variation (+/-).
        normalize: Whether to apply ImageNet channel normalization.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        mode: str = "train",
        hflip_prob: float = 0.5,
        max_rotation_degrees: float = 10.0,
        brightness_jitter: float = 0.15,
        contrast_jitter: float = 0.15,
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.mode = mode.lower()
        self.hflip_prob = hflip_prob
        self.max_rotation_degrees = max_rotation_degrees
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.normalize = normalize

        if self.mode not in ("train", "val", "test"):
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got '{self.mode}'")

    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Executes augmentation pipeline on image and optional mask.

        Args:
            image: Float32 tensor of shape (3, H, W) in range [0.0, 1.0].
            mask: Optional Float32 tensor of shape (1, H, W) with binary {0.0, 1.0}.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Augmented image and mask.
        """
        width, height = self.target_size

        # 1. Deterministic Resize
        # Image: Bilinear interpolation
        image = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        # Mask: Nearest-neighbor interpolation
        if mask is not None:
            mask = TF.resize(mask, [height, width], interpolation=TF.InterpolationMode.NEAREST)

        # In validation or testing mode, return resized and normalized without random jitter
        if self.mode != "train":
            if self.normalize:
                image = normalize_tensor(image, IMAGENET_MEAN, IMAGENET_STD)
            return image, mask

        # ====================================================================
        # TRAINING-ONLY AUGMENTATIONS
        # ====================================================================

        # A. Synchronized Horizontal Flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        # B. Synchronized Subtle Rotation (+/- max_rotation_degrees)
        if self.max_rotation_degrees > 0.0 and random.random() < 0.5:
            angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
            # Rotate image with Bilinear
            image = TF.rotate(image, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            # Rotate mask with Nearest-Neighbor
            if mask is not None:
                mask = TF.rotate(mask, angle=angle, interpolation=TF.InterpolationMode.NEAREST)

        # C. Photometric Jitter (APPLIED ONLY TO IMAGE, NEVER TO MASK)
        if self.brightness_jitter > 0.0:
            b_factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            image = TF.adjust_brightness(image, b_factor)

        if self.contrast_jitter > 0.0:
            c_factor = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
            image = TF.adjust_contrast(image, c_factor)

        # Clamp image to valid range [0.0, 1.0] after photometric transforms
        image = torch.clamp(image, 0.0, 1.0)

        # D. Optional ImageNet Normalization (must be final step before model input)
        if self.normalize:
            image = normalize_tensor(image, IMAGENET_MEAN, IMAGENET_STD)

        return image, mask

