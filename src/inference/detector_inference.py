"""
YOLO Object Detection Inference Engine for Industrial Inspection.

Provides a production-grade inference interface for loading trained YOLO models,
running defect localization, computing pixel metrics, rendering high-contrast
visual overlays, and exporting structured inspection payloads.
"""

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Standard color palette (BGR) for distinct defect visual identification
DEFECT_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "crazing": (34, 34, 220),         # Crimson Red
    "inclusion": (0, 140, 255),       # Vivid Orange
    "patches": (180, 0, 180),         # Magenta / Purple
    "pitted_surface": (42, 80, 165),  # Earth Brown / Bronze
    "rolled-in_scale": (200, 200, 0), # Cyan / Sky Blue
    "scratches": (0, 215, 255),       # Gold / Amber
    "unknown": (128, 128, 128)        # Neutral Gray
}


@dataclass
class DetectionBox:
    """Represents a single localized industrial surface defect bounding box."""
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[float, float, float, float]       # (xmin, ymin, xmax, ymax) in pixels
    box_normalized: Tuple[float, float, float, float] # (xcenter, ycenter, width, height) in [0, 1]
    area_pixels: float
    area_ratio: float                                 # Defect area relative to total image area


@dataclass
class DetectionResponse:
    """Structured inspection payload for image object detection."""
    image_width: int
    image_height: int
    num_detections: int
    is_defective: bool
    inference_time_ms: float
    detections: List[DetectionBox] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts response dataclass to standard Python dictionary."""
        return asdict(self)


class YOLODetector:
    """
    Industrial YOLO Object Detection and Defect Localization Engine.

    Encapsulates model weight loading, hardware target assignment, inference execution,
    bounding box extraction, and visual annotation rendering.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ) -> None:
        """
        Initializes the YOLO Detector.

        Args:
            model_path: Path to .pt model weights or Ultralytics model identifier.
            conf_threshold: Minimum confidence score to accept candidate bounding box.
            iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
            device: Hardware device target ('cpu', 'cuda', '0').
        """
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = str(device)

        logger.info(f"Loading YOLO model from: {self.model_path} on device: {self.device}")
        self.model = YOLO(str(self.model_path))

        # Extract class mapping from model metadata
        self.names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> DetectionResponse:
        """
        Executes defect detection on a single image.

        Args:
            image_input: File path, PIL Image, or NumPy array (RGB/BGR).
            conf_threshold: Optional confidence threshold override.
            iou_threshold: Optional IoU suppression threshold override.

        Returns:
            DetectionResponse: Structured payload containing localized defects.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        # Load or normalize image array to extract geometric dimensions
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Inspection image not found at: {img_path}")
            np_img = cv2.imread(str(img_path))
            if np_img is None:
                raise ValueError(f"Failed to read image at: {img_path}")
            img_h, img_w = np_img.shape[:2]
            inference_input = str(img_path)
        elif isinstance(image_input, Image.Image):
            img_w, img_h = image_input.size
            inference_input = image_input
        elif isinstance(image_input, np.ndarray):
            img_h, img_w = image_input.shape[:2]
            inference_input = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        total_pixels = float(img_w * img_h)

        start_time = time.perf_counter()

        # Run Ultralytics inference
        results = self.model.predict(
            source=inference_input,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[DetectionBox] = []

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    box = xyxy[i]
                    c_id = int(classes[i])
                    c_name = self.names.get(c_id, f"defect_{c_id}")
                    score = float(confs[i])

                    xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    bw = max(0.0, xmax - xmin)
                    bh = max(0.0, ymax - ymin)
                    area_px = bw * bh
                    area_ratio = area_px / total_pixels if total_pixels > 0 else 0.0

                    # Compute normalized (x_center, y_center, width, height)
                    xc_norm = ((xmin + xmax) / 2.0) / img_w
                    yc_norm = ((ymin + ymax) / 2.0) / img_h
                    w_norm = bw / img_w
                    h_norm = bh / img_h

                    detections.append(DetectionBox(
                        class_id=c_id,
                        class_name=c_name,
                        confidence=round(score, 4),
                        box_xyxy=(round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)),
                        box_normalized=(round(xc_norm, 4), round(yc_norm, 4), round(w_norm, 4), round(h_norm, 4)),
                        area_pixels=round(area_px, 1),
                        area_ratio=round(area_ratio, 6)
                    ))

        return DetectionResponse(
            image_width=img_w,
            image_height=img_h,
            num_detections=len(detections),
            is_defective=len(detections) > 0,
            inference_time_ms=round(elapsed_ms, 2),
            detections=detections
        )

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[DetectionBox],
        show_confidence: bool = True,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Renders bounding boxes and defect badges onto an image.

        Args:
            image: BGR NumPy array.
            detections: List of DetectionBox objects to draw.
            show_confidence: Whether to append confidence percentage to the label.
            thickness: Bounding box outline stroke thickness.

        Returns:
            np.ndarray: Annotated BGR image.
        """
        annotated = image.copy()

        for det in detections:
            xmin, ymin, xmax, ymax = [int(v) for v in det.box_xyxy]
            color = DEFECT_COLOR_PALETTE.get(det.class_name.lower(), DEFECT_COLOR_PALETTE["unknown"])

            # 1. Draw defect bounding box rectangle
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, thickness)

            # 2. Build text label string
            if show_confidence:
                label = f"{det.class_name}: {det.confidence * 100:.1f}%"
            else:
                label = det.class_name

            # 3. Compute text size for badge background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Determine badge placement (above box if space permits, else inside top)
            badge_ymin = max(0, ymin - text_h - baseline - 4)
            badge_ymax = badge_ymin + text_h + baseline + 4
            badge_xmax = min(annotated.shape[1], xmin + text_w + 6)

            # Draw badge background rectangle
            cv2.rectangle(annotated, (xmin, badge_ymin), (badge_xmax, badge_ymax), color, -1)

            # Draw high-contrast text (white text on colored badge)
            text_pos = (xmin + 3, badge_ymax - baseline - 2)
            cv2.putText(annotated, label, text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return annotated

    def predict_and_visualize(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        conf_threshold: Optional[float] = None
    ) -> Tuple[DetectionResponse, np.ndarray]:
        """
        High-level helper to detect defects and return both the structured result
        and the visually annotated image.

        Args:
            image_path: Path to product image.
            output_path: Optional destination to save annotated image.
            conf_threshold: Optional confidence threshold override.

        Returns:
            Tuple[DetectionResponse, np.ndarray]: Inspection response and annotated BGR image.
        """
        img_p = Path(image_path)
        img = cv2.imread(str(img_p))
        if img is None:
            raise ValueError(f"Failed to load image from {img_p}")

        response = self.predict(img, conf_threshold=conf_threshold)
        annotated = self.draw_detections(img, response.detections)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), annotated)
            logger.info(f"Saved annotated inspection image to {out_p}")

        return response, annotated

    def export_onnx(self, output_path: Optional[Union[str, Path]] = None, imgsz: int = 640) -> Path:
        """
        Exports the model to ONNX format for accelerated runtime deployment.

        Args:
            output_path: Optional desired destination path.
            imgsz: Input resolution.

        Returns:
            Path: Path to exported .onnx file.
        """
        logger.info(f"Exporting YOLO model to ONNX with imgsz={imgsz}...")
        exported = self.model.export(format="onnx", imgsz=imgsz)
        return Path(exported)
