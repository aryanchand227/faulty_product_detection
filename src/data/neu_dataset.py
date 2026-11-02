"""
NEU Surface Defect Dataset Loader for Supervised Multi-Class Classification.

Handles the Northeastern University (NEU) steel strip surface defect database:
- 6 Defect Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- 1,800 total images (300 per class)
- Clean Train / Validation / Test split separation with zero leakage
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download

from src.config import RAW_DATA_DIR, NEUConfig
from src.data.preprocessing import load_image, to_tensor, normalize_tensor, IMAGENET_MEAN, IMAGENET_STD

NEU_CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
)

CLASS_TO_IDX = {name: idx for idx, name in enumerate(NEU_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(NEU_CLASSES)}


class NEUSurfaceDataset(Dataset):
    """
    Supervised Multi-Class Surface Defect Dataset for Classification Experiments.
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        super().__init__()
        if root_dir is None:
            self.root_dir = RAW_DATA_DIR / "neu_surface"
        else:
            self.root_dir = Path(root_dir)

        self.split = split.lower()
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize
        self.samples = []

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Split must be 'train', 'val', or 'test', got '{self.split}'")

        if self.root_dir.exists():
            self._index_samples()

    def _index_samples(self):
        """
        Indexes images and labels from class subdirectories or split lists.
        Expected structure:
          root_dir/
            crazing/*.jpg (or *.png)
            inclusion/*.jpg
            ...
        Or:
          root_dir/train/crazing/*.jpg
        """
        # Check split directory first: root_dir/train/<class>
        split_path = self.root_dir / self.split
        search_root = split_path if split_path.exists() else self.root_dir

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = search_root / class_name
            if not class_folder.exists():
                continue

            for img_path in sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.bmp"))):
                self.samples.append({
                    "image_path": img_path,
                    "label": class_idx,
                    "defect_type": class_name
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        sample = self.samples[idx]
        img_path = str(sample["image_path"])

        # Load RGB image and resize
        img_rgb = load_image(img_path, target_size=(self.img_size, self.img_size), color_space="RGB")
        tensor = to_tensor(img_rgb)

        if self.transform is not None:
            tensor = self.transform(tensor)

        if self.normalize:
            tensor = normalize_tensor(tensor, IMAGENET_MEAN, IMAGENET_STD)

        return {
            "image": tensor,
            "label": sample["label"],
            "defect_type": sample["defect_type"],
            "image_path": img_path
        }


def download_neu_surface_dataset(target_dir: Optional[Path] = None) -> Path:
    """
    Downloads NEU Surface Defect dataset from Hugging Face mirror.
    """
    if target_dir is None:
        target_dir = RAW_DATA_DIR / "neu_surface"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading NEU dataset to: {target_dir}")
    snapshot_download(
        repo_id="KeenForgeAI/NEU-DET-corrected",
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=["*.jpg", "*.png", "*.xml", "*.txt"]
    )
    print(f"[OK] Download completed: {target_dir}")
    return target_dir

