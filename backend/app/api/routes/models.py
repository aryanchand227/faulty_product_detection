"""
Model Registry Endpoints providing architecture, weights source, and benchmark metrics.
"""

from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.history import ModelRegistryItem, ModelRegistryResponse

router = APIRouter()


@router.get(
    "/registry",
    response_model=ModelRegistryResponse,
    summary="Get Model Registry & Pipeline Architecture"
)
def get_model_registry():
    """
    Returns architecture details, checkpoint provenance, and benchmark metrics
    for all active models in the multi-tier inspection pipeline.
    """
    models = [
        ModelRegistryItem(
            tier="Tier 1: Gatekeeper & Anomaly Localization",
            model_name="PatchCore Anomaly Detector",
            architecture="Wide-ResNet-50-2 (Layer 2 & 3 mid-level features)",
            checkpoint_source=f"HuggingFace Hub ({settings.HF_REPO_ID})",
            device=settings.DEVICE,
            is_loaded=True,
            primary_metrics={
                "image_auroc": 1.0000,
                "pixel_auroc": 0.9848,
                "aupro_at_fpr_0.30": 0.9376,
                "memory_bank_prototypes": 16386,
                "coreset_subsampling_ratio": 0.10,
                "calibrated_separation_threshold": 15.50,
            }
        ),
        ModelRegistryItem(
            tier="Tier 2: Defect Classification",
            model_name="Supervised ResNet-18 Classifier",
            architecture="ResNet-18 (Transfer Learning)",
            checkpoint_source="Local PyTorch Registry (models/classification/)",
            device=settings.DEVICE,
            is_loaded=True,
            primary_metrics={
                "accuracy": 1.0000,
                "num_classes": 6,
                "domain": "NEU Surface Steel Strip Defect"
            }
        ),
        ModelRegistryItem(
            tier="Tier 3A: Defect Localization",
            model_name="YOLOv8 Nano Detector",
            architecture="Ultralytics YOLOv8n (BBox Detection)",
            checkpoint_source="Local PyTorch Weights (yolov8n.pt)",
            device=settings.DEVICE,
            is_loaded=True,
            primary_metrics={
                "box_mAP_50": 0.965,
                "target": "Surface Defect Bounding Boxes"
            }
        ),
        ModelRegistryItem(
            tier="Tier 3B: Defect Segmentation",
            model_name="U-Net Pixel Segmenter",
            architecture="ResNet-34 Encoder + ConvTranspose2d Decoder",
            checkpoint_source="Local PyTorch Registry (models/segmentation/)",
            device=settings.DEVICE,
            is_loaded=True,
            primary_metrics={
                "mean_iou": 0.882,
                "dice_coefficient": 0.937,
                "target": "Pixel-Level Defect Masks"
            }
        ),
    ]

    return ModelRegistryResponse(
        service=settings.PROJECT_NAME,
        device=settings.DEVICE,
        models=models
    )

