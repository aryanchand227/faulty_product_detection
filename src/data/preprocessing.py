"""
Image and Mask Preprocessing Module.

Provides reusable, tested functions for:
- Reading images and converting color spaces (BGR to RGB)
- Standardizing pixel tensors using ImageNet or custom statistics
- Inverse normalization (denormalization) for visualization and debugging
- Discrete mask preprocessing with strict nearest-neighbor interpolation
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ImageNet statistics (standard for models pretrained on ImageNet-1k)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    color_space: str = "RGB"
) -> np.ndarray:
    """
    Loads an image from disk using OpenCV, validates readability, converts
    color space, and optionally resizes it with bilinear interpolation.

    Args:
        path: Filepath to the image.
        target_size: Optional (width, height) tuple for resizing.
        color_space: Target color format ('RGB' or 'BGR'). Default is 'RGB'.

    Returns:
        numpy.ndarray: Image array with shape (H, W, C) and dtype uint8.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV fails to decode the image file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {path}")

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Corrupted or unsupported image file: {path}")

    if color_space.upper() == "RGB":
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif color_space.upper() == "BGR":
        img = img_bgr
    else:
        raise ValueError(f"Unsupported color_space: {color_space}. Use 'RGB' or 'BGR'.")

    if target_size is not None:
        width, height = target_size
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Converts a uint8 (H, W, C) numpy array in range [0, 255]
    into a float32 (C, H, W) PyTorch tensor in range [0.0, 1.0].
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(img)}")

    if img.ndim == 2:  # Grayscale image (H, W) -> (1, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    elif img.ndim == 3:  # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
    else:
        raise ValueError(f"Invalid array dimensions: {img.ndim}. Expected 2 or 3.")

    return tensor


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> torch.Tensor:
    """
    Standardizes a float32 tensor of shape (C, H, W) or (B, C, H, W)
    using channel-wise z-score normalization:
        z = (x - mean) / std

    Args:
        tensor: Float tensor with values scaled to [0.0, 1.0].
        mean: Per-channel means.
        std: Per-channel standard deviations.

    Returns:
        torch.Tensor: Standardized tensor of identical shape.
    """
    if tensor.max() > 1.5:
        raise ValueError(
            f"Expected input tensor scaled to [0.0, 1.0], but found max value: {tensor.max().item():.2f}"
        )

    # Use PyTorch's native functional normalize
    return TF.normalize(tensor, mean=list(mean), std=list(std))


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> torch.Tensor:
    """
    Inverts channel-wise standardization to reconstruct a [0.0, 1.0] image:
        x = z * std + mean
    Clips output to [0.0, 1.0] to eliminate floating-point precision artifacts.
    """
    # Clone to prevent in-place mutation of the input tensor
    x = tensor.clone()
    
    if x.ndim == 3:  # (C, H, W)
        for c in range(3):
            x[c] = x[c] * std[c] + mean[c]
    elif x.ndim == 4:  # (B, C, H, W)
        for c in range(3):
            x[:, c] = x[:, c] * std[c] + mean[c]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape: {tensor.shape}")

    return torch.clamp(x, 0.0, 1.0)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a float32 (C, H, W) PyTorch tensor in range [0.0, 1.0]
    back into a uint8 (H, W, C) numpy array in range [0, 255] for OpenCV/Matplotlib.
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor (C, H, W), got shape: {tensor.shape}")

    img_np = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img_uint8 = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img_uint8


def load_and_preprocess_mask(
    mask_path: Optional[Union[str, Path]],
    target_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Loads and preprocesses a ground-truth segmentation mask.
    
    If mask_path is None or empty (normal product), returns a zero-filled tensor.
    If mask exists, resizes it with INTER_NEAREST to preserve discrete labels,
    and thresholds pixels to {0.0, 1.0}.

    Args:
        mask_path: Path to mask PNG file or None.
        target_size: (width, height) tuple.

    Returns:
        torch.Tensor: Shape (1, H, W), dtype float32, binary values {0.0, 1.0}.
    """
    width, height = target_size

    if mask_path is None:
        return torch.zeros((1, height, width), dtype=torch.float32)

    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    mask_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise ValueError(f"Failed to decode mask: {path}")

    # CRITICAL: Nearest-neighbor interpolation preserves discrete class boundaries
    mask_resized = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_NEAREST)

    # Binarize to exact 0.0 or 1.0
    binary_mask = (mask_resized > 0).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)

