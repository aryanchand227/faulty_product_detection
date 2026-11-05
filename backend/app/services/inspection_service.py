"""
Inspection Service managing the UnifiedQualityInspector instance,
dynamic calibration adjustments, and REST API payload synthesis.
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import DecisionEngine, QualityThresholds
from src.inspection.unified_pipeline import ProductDomain, UnifiedInspectionResult, UnifiedQualityInspector

logger = logging.getLogger(__name__)


class InspectionService:
    """
    Singleton service maintaining the active UnifiedQualityInspector instance for the API.
    """

    _instance: Optional["InspectionService"] = None

    def __init__(self, inspector: Optional[UnifiedQualityInspector] = None):
        self.device = "cpu"
        self.calibrator = CameraCalibrator(mm_per_pixel=0.15)
        self.thresholds = QualityThresholds(
            anomaly_threshold=15.50,
            anomaly_margin=0.10,
            critical_area_mm2=25.0,
            critical_classes=["broken_large", "crazing"]
        )
        self.decision_engine = DecisionEngine(thresholds=self.thresholds)

        if inspector is not None:
            self.inspector = inspector
        else:
            # Initialize with default/fallback inspector
            self.inspector = UnifiedQualityInspector(
                domain=ProductDomain.GLASS_BOTTLE,
                calibrator=self.calibrator,
                decision_engine=self.decision_engine,
                early_exit=True,
                device=self.device
            )

    @classmethod
    def get_instance(cls) -> "InspectionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, service: "InspectionService") -> None:
        cls._instance = service

    def update_thresholds(self, updates: Dict[str, Any]) -> QualityThresholds:
        """Dynamically modifies operational quality thresholds."""
        current_dict = self.thresholds.to_dict()
        for k, v in updates.items():
            if v is not None and k in current_dict:
                current_dict[k] = v

        self.thresholds = QualityThresholds(**current_dict)
        self.decision_engine.thresholds = self.thresholds
        self.inspector.decision_engine = self.decision_engine
        return self.thresholds

    def update_calibration(self, mm_per_pixel: float) -> CameraCalibrator:
        """Dynamically updates metric camera scaling factor."""
        self.calibrator = CameraCalibrator(mm_per_pixel=mm_per_pixel)
        self.inspector.calibrator = self.calibrator
        return self.calibrator

    def inspect_image(
        self,
        image_bytes: bytes,
        domain: ProductDomain = ProductDomain.GLASS_BOTTLE,
        render_visuals: bool = True,
        force_full_pipeline: bool = False,
        extract_morphology: bool = True,
        generate_certificate: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inspection on uploaded image bytes and returns structured dictionary.
        """
        # Decode image from raw bytes
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from provided byte stream.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Set domain for inspector
        self.inspector.domain = domain

        result: UnifiedInspectionResult = self.inspector.inspect(
            image_input=img_rgb,
            render_visuals=render_visuals,
            force_full_pipeline=force_full_pipeline,
            extract_morphology=extract_morphology,
            generate_certificate=generate_certificate
        )

        res_dict = result.to_dict()

        # Encode annotated visual overlay to Base64 PNG if rendered
        if result.annotated_image is not None and render_visuals:
            annotated_bgr = cv2.cvtColor(result.annotated_image, cv2.COLOR_RGB2BGR)
            success, encoded_img = cv2.imencode(".png", annotated_bgr)
            if success:
                b64_str = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
                res_dict["annotated_image_base64"] = f"data:image/png;base64,{b64_str}"
            else:
                res_dict["annotated_image_base64"] = None
        else:
            res_dict["annotated_image_base64"] = None

        return res_dict
