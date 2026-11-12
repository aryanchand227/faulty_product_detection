"""
Inspection Service managing the UnifiedQualityInspector instance,
thread-safe configuration snapshots, audit logging, and REST API payload synthesis.
"""

import base64
from datetime import datetime, timezone
import io
import logging
import threading
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import DecisionEngine, QualityThresholds
from src.inspection.unified_pipeline import ProductDomain, UnifiedInspectionResult, UnifiedQualityInspector

logger = logging.getLogger(__name__)


class InspectionService:
    """
    Singleton service maintaining the active UnifiedQualityInspector instance,
    thread-safe configuration snapshots, and mutation audit logs for the API.
    """

    _instance: Optional["InspectionService"] = None
    _lock = threading.Lock()

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

        # Configuration versioning and audit trail
        self.config_version_seq = 0
        self.audit_log: List[Dict[str, Any]] = []

        if inspector is not None:
            self.inspector = inspector
        else:
            t1_inferencer = self._load_tier1_inferencer()
            self.inspector = UnifiedQualityInspector(
                domain=ProductDomain.GLASS_BOTTLE,
                tier1_inferencer=t1_inferencer,
                calibrator=self.calibrator,
                decision_engine=self.decision_engine,
                early_exit=True,
                device=self.device
            )

    def _load_tier1_inferencer(self):
        """Loads PatchCore with Wide-ResNet-50-2 memory bank from Hugging Face Hub or local cache."""
        try:
            import os
            import torch
            from huggingface_hub import hf_hub_download
            from src.models.anomaly_detection.patchcore import PatchCore
            from src.inference.anomaly_inference import PatchCoreInferencer
            from backend.app.core.config import settings

            hf_token = os.getenv("HF_TOKEN")
            logger.info(f"Loading Tier 1 PatchCore weights from Hugging Face Hub ({settings.HF_REPO_ID})...")
            ckpt_path = hf_hub_download(
                repo_id=settings.HF_REPO_ID,
                filename="checkpoints/patchcore_bottle_wide_resnet50_2.pt",
                token=hf_token
            )
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model = PatchCore(backbone_name="wide_resnet50_2", coreset_ratio=0.10, device=self.device)
            model.memory_bank = ckpt["memory_bank"].to(self.device)
            model.threshold = float(ckpt.get("threshold_oracle", 15.50))
            model.backbone.eval()
            logger.info(f"Successfully loaded PatchCore memory bank {model.memory_bank.shape} from {ckpt_path}")
            return PatchCoreInferencer(patchcore_instance=model, device=self.device)
        except Exception as e:
            logger.warning(f"Could not load Hugging Face PatchCore weights: {e}. Falling back to default.")
            from src.inference.anomaly_inference import PatchCoreInferencer
            return PatchCoreInferencer(device=self.device)

    @property
    def config_version(self) -> str:
        return f"CFG-v1.0.{self.config_version_seq}"

    @classmethod
    def get_instance(cls) -> "InspectionService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def set_instance(cls, service: "InspectionService") -> None:
        with cls._lock:
            cls._instance = service

    def update_thresholds(
        self,
        updates: Dict[str, Any],
        actor: str = "authorized_operator"
    ) -> QualityThresholds:
        """
        Dynamically modifies operational quality thresholds, appends to the audit trail,
        and increments the configuration version.
        """
        with self._lock:
            current_dict = self.thresholds.to_dict()
            changed = {}
            for k, v in updates.items():
                if v is not None and k in current_dict:
                    if current_dict[k] != v:
                        changed[k] = {"old": current_dict[k], "new": v}
                        current_dict[k] = v

            self.thresholds = QualityThresholds(**current_dict)
            self.decision_engine.thresholds = self.thresholds
            self.inspector.decision_engine = self.decision_engine

            self.config_version_seq += 1
            now_iso = datetime.now(timezone.utc).isoformat()
            self.audit_log.append({
                "timestamp_utc": now_iso,
                "actor": actor,
                "action": "UPDATE_THRESHOLDS",
                "changed_parameters": changed,
                "config_version": self.config_version
            })

            logger.info(f"Thresholds updated to {self.config_version} by {actor}: {changed}")
            return self.thresholds

    def update_calibration(
        self,
        mm_per_pixel: float,
        actor: str = "authorized_operator"
    ) -> CameraCalibrator:
        """
        Dynamically updates metric camera scaling factor and logs audit record.
        """
        with self._lock:
            old_scale = self.calibrator.mm_per_pixel
            self.calibrator = CameraCalibrator(mm_per_pixel=mm_per_pixel)
            self.inspector.calibrator = self.calibrator

            self.config_version_seq += 1
            now_iso = datetime.now(timezone.utc).isoformat()
            self.audit_log.append({
                "timestamp_utc": now_iso,
                "actor": actor,
                "action": "UPDATE_CALIBRATION",
                "changed_parameters": {"mm_per_pixel": {"old": old_scale, "new": mm_per_pixel}},
                "config_version": self.config_version
            })

            logger.info(f"Calibration updated to {self.config_version} by {actor}: {old_scale} -> {mm_per_pixel}")
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
        Executes end-to-end inspection on uploaded image bytes with configuration snapshotting.
        """
        # Decode image from raw bytes
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from provided byte stream. File may be corrupted or malformed.")

        img_h, img_w = img_bgr.shape[:2]
        if img_h < 16 or img_w < 16:
            raise ValueError(f"Image dimensions ({img_w}x{img_h}) are below minimum inspection resolution (16x16).")
        if img_h > 4096 or img_w > 4096:
            raise ValueError(f"Image dimensions ({img_w}x{img_h}) exceed maximum allowable resolution (4096x4096).")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Thread-safe snapshot: Capture active configuration version for this inspection execution
        with self._lock:
            active_cfg_version = self.config_version
            # Set domain for active run
            self.inspector.domain = domain

        result: UnifiedInspectionResult = self.inspector.inspect(
            image_input=img_rgb,
            render_visuals=render_visuals,
            force_full_pipeline=force_full_pipeline,
            extract_morphology=extract_morphology,
            generate_certificate=generate_certificate
        )

        res_dict = result.to_dict()

        # If a certificate was generated, ensure config_version is stamped
        if res_dict.get("certificate"):
            res_dict["certificate"]["config_version"] = active_cfg_version

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
