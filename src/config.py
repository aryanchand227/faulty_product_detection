"""
Centralized project configuration.

This module contains all configurable parameters for the project.
Settings are organized by component to make them easy to find and modify.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Path Configuration
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
ANNOTATIONS_DIR = DATASETS_DIR / "annotations"

# MVTec AD paths
MVTEC_DIR = RAW_DATA_DIR / "mvtec_ad"
MVTEC_BOTTLE_DIR = MVTEC_DIR / "bottle"

# NEU Surface Defect paths
NEU_DIR = RAW_DATA_DIR / "neu_surface"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Output paths
DOCS_DIR = PROJECT_ROOT / "docs"


# ============================================================================
# Dataset Configuration
# ============================================================================

@dataclass
class MVTecConfig:
    """Configuration for MVTec AD dataset."""
    name: str = "mvtec_ad"
    category: str = "bottle"
    image_size: int = 256  # Resize target (square)
    
    # MVTec bottle defect types (from official test split)
    defect_types: tuple = ("broken_large", "broken_small", "contamination")
    
    # Normalization (ImageNet statistics, used with pretrained backbones)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class NEUConfig:
    """Configuration for NEU Surface Defect dataset."""
    name: str = "neu_surface"
    num_classes: int = 6
    class_names: tuple = (
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches"
    )
    image_size: int = 224  # Standard input resolution for classification
    total_images: int = 1797  # Empirically verified audit count (1797 images & XMLs)
    
    # Normalization (ImageNet statistics)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Default training hyperparameters.
    
    These are starting points. Actual values may be adjusted
    per experiment based on validation performance.
    """
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    num_workers: int = 2  # DataLoader workers
    pin_memory: bool = True  # Faster GPU transfer when True
    seed: int = 42
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    
    # Mixed precision (FP16) - reduces memory, may speed up training
    use_mixed_precision: bool = True


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ClassificationConfig:
    """Configuration for classification models."""
    num_classes: int = 2  # normal vs defective (binary)
    pretrained: bool = True
    freeze_backbone: bool = True  # Start frozen, unfreeze later
    dropout: float = 0.3


@dataclass
class YOLOConfig:
    """Configuration for YOLO detection model."""
    model_variant: str = "yolov8n"  # nano variant for efficiency
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 50


@dataclass
class UNetConfig:
    """Configuration for U-Net segmentation model."""
    in_channels: int = 3  # RGB input
    out_channels: int = 1  # Binary mask output
    features: tuple = (64, 128, 256, 512)  # Feature map sizes
    image_size: int = 256


@dataclass
class PatchCoreConfig:
    """Configuration for PatchCore anomaly detection."""
    backbone: str = "wide_resnet50_2"
    layers_to_extract: tuple = ("layer2", "layer3")
    coreset_sampling_ratio: float = 0.1  # Fraction of patches to keep
    num_neighbors: int = 9  # k in k-NN
    image_size: int = 256


# ============================================================================
# Inference Configuration
# ============================================================================

@dataclass  
class InferenceConfig:
    """Configuration for inference and deployment."""
    device: str = "cpu"  # "cpu" or "cuda"
    max_image_size_mb: int = 10
    confidence_threshold: float = 0.5
    
    # Severity thresholds (configurable, NOT physically validated)
    severity_thresholds: dict = field(default_factory=lambda: {
        "low": 0.01,      # < 1% of image area marked defective
        "medium": 0.05,    # 1-5%
        "high": 0.15,      # 5-15%
        "critical": 1.0    # > 15%
    })


# ============================================================================
# Application Configuration
# ============================================================================

@dataclass
class APIConfig:
    """Configuration for FastAPI backend."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    allowed_origins: tuple = ("http://localhost:3000",)
    max_upload_size_mb: int = 10
    
    # Supported image formats
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
