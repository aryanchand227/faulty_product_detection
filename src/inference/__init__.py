"""
Industrial Inference Engines across Multi-Tier Vision Models.
"""

from src.inference.anomaly_inference import (
    AnomalyInspectionResponse,
    AnomalyRegion,
    PatchCoreInferencer,
)
from src.inference.classifier_inference import (
    ClassifierInferencer,
    DefectClassificationResponse,
    DEFAULT_DEFECT_CLASSES,
    MVTEC_BOTTLE_CLASSES,
)
from src.inference.detector_inference import (
    DetectionBox,
    DetectionResponse,
    YOLODetector,
    DEFECT_COLOR_PALETTE,
    compute_bbox_union_area,
)
from src.inference.segmenter_inference import (
    DefectComponent,
    SegmentationResponse,
    UNetSegmenter,
)

__all__ = [
    "AnomalyInspectionResponse",
    "AnomalyRegion",
    "PatchCoreInferencer",
    "ClassifierInferencer",
    "DefectClassificationResponse",
    "DEFAULT_DEFECT_CLASSES",
    "MVTEC_BOTTLE_CLASSES",
    "DetectionBox",
    "DetectionResponse",
    "YOLODetector",
    "DEFECT_COLOR_PALETTE",
    "compute_bbox_union_area",
    "DefectComponent",
    "SegmentationResponse",
    "UNetSegmenter",
]

