"""
PyTorch Dataset Implementation for MVTec AD Industrial Quality Inspection.

This module provides the MVTecDataset class, which conforms to PyTorch's
torch.utils.data.Dataset protocol. It handles image loading, color-space
conversion, discrete mask parsing, and transformation pipelines.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

class MVTecDataset(Dataset):
    """
    Dataset class for MVTec Anomaly Detection benchmark.
    
    Attributes:
        category_dir: Path to category root folder (e.g., datasets/raw/mvtec_ad/bottle)
        split: 'train' or 'test'
        img_size: Target square image dimensions (H = W = img_size)
        transform: Optional callable transform applied to RGB PIL image or Tensor
        target_transform: Optional callable transform applied to binary mask
    """

    def __init__(
        self,
        category_dir: Union[str, Path],
        split: str = "train",
        img_size: int = 256,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.category_dir = Path(category_dir)
        self.split = split.lower()
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform

        if self.split not in ("train", "test"):
            raise ValueError(f"Split must be 'train' or 'test', got '{self.split}'")

        if not self.category_dir.exists():
            raise FileNotFoundError(f"Category directory does not exist: {self.category_dir}")

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """
        Scans directory structure and records image paths, labels, and mask paths.
        
        Directory layout:
          train/good/*.png                 -> normal images (no mask)
          test/good/*.png                  -> normal test images (no mask)
          test/<defect_type>/*.png         -> anomalous test images
          ground_truth/<defect_type>/*_mask.png -> defect pixel masks
        """
        split_dir = self.category_dir / self.split
        gt_dir = self.category_dir / "ground_truth"

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue

            defect_type = folder.name
            is_anomaly = 0 if defect_type == "good" else 1

            for img_path in sorted(folder.glob("*.png")):
                mask_path = None
                if is_anomaly:
                    expected_mask = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                    if expected_mask.exists():
                        mask_path = expected_mask
                    else:
                        raise FileNotFoundError(f"Ground truth mask missing for: {img_path}")

                self.samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "defect_type": defect_type,
                    "is_anomaly": is_anomaly,
                })

    def __len__(self) -> int:
        """
        Returns the total number of samples in this dataset split.
        
        PyTorch DataLoader uses this to compute batch iterations per epoch:
            steps_per_epoch = ceil(len(dataset) / batch_size)
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        """
        Loads and prepares a single data sample by index.
        
        Steps:
        1. Read image from disk. Convert BGR (OpenCV default) to RGB.
        2. Resize image using Bilinear interpolation.
        3. Convert image to PyTorch tensor with shape (3, H, W) and normalized to [0.0, 1.0].
        4. Load mask (if anomalous) or create zero-mask (if good).
        5. Resize mask using NEAREST-NEIGHBOR interpolation to protect discrete 0/1 values.
        6. Return standardized dictionary.
        
        Returns:
            Dict containing:
                - 'image': Tensor of shape (3, H, W), dtype float32, range [0.0, 1.0]
                - 'mask': Tensor of shape (1, H, W), dtype float32, binary {0.0, 1.0}
                - 'label': int (0 for normal, 1 for anomaly)
                - 'defect_type': str (e.g. 'good', 'broken_large')
                - 'image_path': str (source path on disk)
        """
        sample = self.samples[idx]
        img_path = str(sample["image_path"])

        # 1. Load image using OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image at: {img_path}")

        # 2. Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 3. Resize image (cv2.INTER_LINEAR = Bilinear interpolation)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # 4. Convert (H, W, C) uint8 [0, 255] to PyTorch tensor (C, H, W) float32 [0.0, 1.0]
        # Transpose: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float() / 255.0

        # Apply optional custom transform pipeline
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # 5. Handle ground-truth segmentation mask
        if sample["mask_path"] is not None:
            mask_gray = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                raise RuntimeError(f"Failed to read mask at: {sample['mask_path']}")
            
            # CRITICAL: Always use INTER_NEAREST for discrete class/defect masks
            mask_resized = cv2.resize(mask_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            # Threshold to binary {0.0, 1.0}
            mask_binary = (mask_resized > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # Shape: (1, H, W)
        else:
            # Good samples have no defect pixels -> return all-zero mask of shape (1, H, W)
            mask_tensor = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_tensor)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": sample["is_anomaly"],
            "defect_type": sample["defect_type"],
            "image_path": img_path,
        }

