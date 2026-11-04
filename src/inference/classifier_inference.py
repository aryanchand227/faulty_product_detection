"""
Defect Classification Inference Engine for Industrial Inspection.

Executes Tier 2 deep learning classification:
1. Categorizes identified anomalies into discrete defect morphology classes.
2. Computes predictive probability distributions and Shannon entropy for uncertainty estimation.
3. Provides production-grade error handling and fallback behaviors.
"""

from dataclasses import asdict, dataclass, field
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_image,
    normalize_tensor,
    to_tensor,
)
from src.models.classification.resnet_classifier import ResNetClassifier

logger = logging.getLogger(__name__)

# Standard Industrial Defect Class Catalog (NEU-DET Steel + General AOI)
DEFAULT_DEFECT_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Glassware Defect Catalog (MVTec Bottle)
MVTEC_BOTTLE_CLASSES: List[str] = [
    "good",
    "broken_large",
    "broken_small",
    "contamination"
]


@dataclass
class DefectClassificationResponse:
    """
    Structured response payload for Tier 2 defect classification.

    Attributes:
        predicted_class_id: Index of highest probability class
        predicted_class_name: Human-readable defect label
        confidence: Maximum softmax probability in [0.0, 1.0]
        is_defective: True if predicted class is not a nominal/good class
        probabilities: Full probability distribution across all classes
        entropy: Shannon entropy in bits (measures classifier uncertainty)
        inference_time_ms: Forward pass wall-clock latency in milliseconds
    """
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    is_defective: bool
    probabilities: Dict[str, float]
    entropy: float
    inference_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClassifierInferencer:
    """
    Industrial inference engine for defect category classification.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_instance: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        target_size: Tuple[int, int] = (224, 224),
        nominal_class_name: str = "good"
    ) -> None:
        """
        Initializes the classification inference engine.

        Args:
            model_path: Optional path to saved PyTorch state_dict or model checkpoint.
            model_instance: Optional pre-initialized PyTorch nn.Module.
            class_names: List of class labels in index order.
            device: Hardware target ('cpu', 'cuda').
            target_size: (width, height) expected by model.
            nominal_class_name: Name of class representing a non-defective product.
        """
        self.device = torch.device(device)
        self.target_size = target_size
        self.nominal_class_name = nominal_class_name.lower()
        self.class_names = class_names or DEFAULT_DEFECT_CLASSES

        if model_instance is not None:
            self.model = model_instance.to(self.device)
            self.model.eval()
        elif model_path is not None:
            model_p = Path(model_path)
            if model_p.exists():
                self.model = ResNetClassifier(num_classes=len(self.class_names), pretrained=False)
                checkpoint = torch.load(model_p, map_location=self.device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
            else:
                logger.warning(f"Classification checkpoint not found at: {model_p}. Using initialized backbone.")
                self.model = ResNetClassifier(num_classes=len(self.class_names), pretrained=False).to(self.device)
                self.model.eval()
        else:
            self.model = ResNetClassifier(num_classes=len(self.class_names), pretrained=False).to(self.device)
            self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path]
    ) -> DefectClassificationResponse:
        """
        Runs defect classification forward inference.

        Args:
            image_input: File path, PIL Image, or NumPy array (RGB/BGR).

        Returns:
            DefectClassificationResponse: Categorization results and uncertainty telemetry.
        """
        t_start = time.perf_counter()

        # Load and preprocess image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            img_np = load_image(img_path, target_size=self.target_size)
        elif isinstance(image_input, Image.Image):
            img_resized = image_input.convert("RGB").resize(self.target_size, Image.BILINEAR)
            img_np = np.array(img_resized)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 3:
                img_rgb = image_input
            else:
                img_rgb = image_input[:, :, :3]
            img_np = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        img_t = to_tensor(img_np)
        img_norm = normalize_tensor(img_t, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(self.device)

        logits = self.model(img_norm)
        probs_tensor = F.softmax(logits, dim=1).squeeze(0)
        probs_np = probs_tensor.cpu().numpy()

        best_idx = int(np.argmax(probs_np))
        best_conf = float(probs_np[best_idx])
        best_name = self.class_names[best_idx] if best_idx < len(self.class_names) else f"class_{best_idx}"

        # Calculate Shannon entropy: H(p) = - sum(p * log2(p + eps))
        eps = 1e-9
        entropy = float(-np.sum(probs_np * np.log2(probs_np + eps)))

        # Format probability distribution
        prob_dict: Dict[str, float] = {}
        for idx, name in enumerate(self.class_names):
            if idx < len(probs_np):
                prob_dict[name] = round(float(probs_np[idx]), 4)

        is_defective = best_name.lower() != self.nominal_class_name

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return DefectClassificationResponse(
            predicted_class_id=best_idx,
            predicted_class_name=best_name,
            confidence=round(best_conf, 4),
            is_defective=is_defective,
            probabilities=prob_dict,
            entropy=round(entropy, 4),
            inference_time_ms=round(elapsed_ms, 2)
        )
