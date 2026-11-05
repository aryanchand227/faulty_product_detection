"""
Camera Calibration and Metric Dimension Transformation for Industrial AOI.

Translates discrete pixel-plane geometries into physical real-world engineering dimensions
(e.g. millimeters and square millimeters) for automated defect severity evaluation.

Features:
1. FOV-based scaling: Computes mm_per_pixel from known physical field of view and sensor width.
2. Checkerboard calibration: Computes optical intrinsics and radial distortion coefficients
   using OpenCV standard checkerboard corner detection.
3. Planar metric transformation:
   - 1D linear distances: length_mm = length_px * mm_per_pixel
   - 2D surface areas: area_mm2 = area_px * (mm_per_pixel)^2
   - Bounding box dimensioning: (width_mm, height_mm, aspect_ratio)
4. Config persistence: Serialization to and from JSON.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibrationConfig:
    """Configuration parameters for optical camera calibration."""
    mm_per_pixel: float = 0.10  # Default: 1 pixel = 0.1 mm (10 px = 1 mm)
    sensor_width_px: int = 1920
    sensor_height_px: int = 1080
    fov_width_mm: Optional[float] = None
    fov_height_mm: Optional[float] = None
    focal_length_mm: Optional[float] = None
    working_distance_mm: Optional[float] = None
    camera_matrix: Optional[List[List[float]]] = None  # 3x3 intrinsic matrix
    dist_coeffs: Optional[List[float]] = None          # [k1, k2, p1, p2, k3]
    calibration_date: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraCalibrationConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CameraCalibrator:
    """
    Industrial camera calibration manager for optical dimensional inspection.
    """

    def __init__(
        self,
        config: Optional[CameraCalibrationConfig] = None,
        mm_per_pixel: Optional[float] = None,
        config_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initializes the Camera Calibrator.

        Args:
            config: CameraCalibrationConfig instance.
            mm_per_pixel: Direct override for linear scale factor (mm/pixel).
            config_path: File path to load JSON configuration from.
        """
        if config_path is not None:
            self.config = self.load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = CameraCalibrationConfig()

        if mm_per_pixel is not None:
            self.config.mm_per_pixel = float(mm_per_pixel)

        if self.config.mm_per_pixel <= 0.0:
            raise ValueError(f"mm_per_pixel must be positive, got {self.config.mm_per_pixel}")

    @property
    def mm_per_pixel(self) -> float:
        """Linear scale factor: physical millimeters represented by one pixel."""
        return self.config.mm_per_pixel

    @property
    def mm2_per_pixel2(self) -> float:
        """Areal scale factor: physical square millimeters per pixel squared."""
        return self.config.mm_per_pixel ** 2

    def pixels_to_mm(self, pixels: float) -> float:
        """
        Converts pixel distance to physical millimeters.

        Args:
            pixels: 1D distance or length in pixels.

        Returns:
            float: Length in millimeters.
        """
        return float(pixels) * self.mm_per_pixel

    def mm_to_pixels(self, mm: float) -> float:
        """
        Converts physical millimeters to pixel distance.

        Args:
            mm: 1D distance or length in millimeters.

        Returns:
            float: Distance in pixels.
        """
        return float(mm) / self.mm_per_pixel

    def pixel_area_to_mm2(self, pixel_area: float, is_planar_surface: bool = True) -> float:
        """
        Converts discrete pixel area to physical area in square millimeters (mm^2).

        Geometric Validity & Optical Constraints:
        -----------------------------------------
        1. Flat Planar Surfaces (is_planar_surface=True, e.g. NEU-DET rolled steel strips):
           The single-scalar relationship Area_mm2 = Area_px * (mm_per_pixel)^2 is physically
           rigorous, assuming the surface is perpendicular to the optical axis and lens
           distortion is corrected.
        2. 3D Curved Surfaces (is_planar_surface=False, e.g. MVTec AD glass bottles):
           A single planar scalar represents an ESTIMATED 2D PROJECTED DEFECT AREA on the camera
           sensor plane, NOT the true 3D surface area. Due to cylindrical foreshortening
           (dA_proj = cos(theta) * dA_true), apparent area shrinks toward the bottle silhouette.
           Accurate 3D surface measurement requires cylindrical unrolling, multi-camera views,
           or optical laser triangulation.

        Args:
            pixel_area: Area in pixels^2 (e.g. from mask or bbox).
            is_planar_surface: Flag indicating whether inspected surface is flat.

        Returns:
            float: Area in square millimeters (mm^2).
        """
        return float(pixel_area) * self.mm2_per_pixel2

    def estimate_projected_area_mm2(self, pixel_area: float) -> float:
        """
        Calculates estimated 2D projected defect area on the sensor plane (mm^2).
        Explicitly designated for 3D curved objects where true surface area is unrolled.
        """
        return self.pixel_area_to_mm2(pixel_area, is_planar_surface=False)

    def mm2_to_pixel_area(self, area_mm2: float) -> float:
        """
        Converts physical surface area (mm^2) to pixel count.

        Args:
            area_mm2: Area in mm^2.

        Returns:
            float: Area in pixels.
        """
        return float(area_mm2) / self.mm2_per_pixel2

    def compute_box_dimensions_mm(
        self,
        box_xywh: Tuple[float, float, float, float]
    ) -> Dict[str, float]:
        """
        Computes physical dimensions for a bounding box (x, y, w, h).

        Args:
            box_xywh: (x, y, width, height) in pixel coordinates.

        Returns:
            Dict containing width_mm, height_mm, area_mm2, and aspect_ratio.
        """
        _, _, w_px, h_px = box_xywh
        w_mm = self.pixels_to_mm(w_px)
        h_mm = self.pixels_to_mm(h_px)
        area_mm2 = self.pixel_area_to_mm2(w_px * h_px)
        aspect_ratio = float(w_px / h_px) if h_px > 0 else 0.0

        return {
            "width_mm": round(w_mm, 3),
            "height_mm": round(h_mm, 3),
            "area_mm2": round(area_mm2, 3),
            "aspect_ratio": round(aspect_ratio, 3)
        }

    def calibrate_from_fov(
        self,
        fov_width_mm: float,
        image_width_px: int,
        fov_height_mm: Optional[float] = None,
        image_height_px: Optional[int] = None
    ) -> float:
        """
        Calculates mm_per_pixel from known physical field of view and sensor resolution.

        Args:
            fov_width_mm: Conveyor inspection field of view width in mm.
            image_width_px: Image sensor resolution width in pixels.
            fov_height_mm: Optional FOV height in mm.
            image_height_px: Optional image height in pixels.

        Returns:
            float: Calibrated mm_per_pixel.
        """
        if fov_width_mm <= 0 or image_width_px <= 0:
            raise ValueError("FOV dimensions and image resolutions must be positive.")

        scale_x = fov_width_mm / float(image_width_px)
        self.config.mm_per_pixel = float(scale_x)
        self.config.fov_width_mm = float(fov_width_mm)
        self.config.sensor_width_px = int(image_width_px)

        if fov_height_mm is not None and image_height_px is not None:
            self.config.fov_height_mm = float(fov_height_mm)
            self.config.sensor_height_px = int(image_height_px)

        logger.info(f"Calibrated scale from FOV: {self.config.mm_per_pixel:.5f} mm/pixel")
        return self.config.mm_per_pixel

    def calibrate_from_checkerboard(
        self,
        image: np.ndarray,
        pattern_size: Tuple[int, int] = (9, 6),
        square_size_mm: float = 10.0
    ) -> Tuple[bool, float]:
        """
        Calibrates mm_per_pixel from a captured image of a planar checkerboard target.

        Args:
            image: Grayscale or RGB image containing checkerboard target.
            pattern_size: Internal corner counts (columns, rows), e.g. (9, 6).
            square_size_mm: Physical dimension of one checker square in mm.

        Returns:
            Tuple[bool, float]: (success_flag, calculated_mm_per_pixel)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not ret or corners is None:
            logger.warning("Checkerboard corners not detected in calibration target image.")
            return False, self.config.mm_per_pixel

        # Refine corners with sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        pts = corners_sub.reshape(-1, 2)

        # Compute average Euclidean distance between adjacent corners
        cols, rows = pattern_size
        distances_px = []

        # Horizontal distances
        for r in range(rows):
            for c in range(cols - 1):
                idx1 = r * cols + c
                idx2 = r * cols + (c + 1)
                pt1 = pts[idx1]
                pt2 = pts[idx2]
                dist = float(np.linalg.norm(pt1 - pt2))
                distances_px.append(dist)

        # Vertical distances
        for r in range(rows - 1):
            for c in range(cols):
                idx1 = r * cols + c
                idx2 = (r + 1) * cols + c
                pt1 = pts[idx1]
                pt2 = pts[idx2]
                dist = float(np.linalg.norm(pt1 - pt2))
                distances_px.append(dist)

        mean_px_per_square = float(np.mean(distances_px))
        if mean_px_per_square <= 0:
            return False, self.config.mm_per_pixel

        calibrated_scale = float(square_size_mm / mean_px_per_square)
        self.config.mm_per_pixel = calibrated_scale
        logger.info(
            f"Checkerboard calibration success: mean square width = {mean_px_per_square:.2f} px, "
            f"calibrated mm_per_pixel = {calibrated_scale:.5f}"
        )
        return True, calibrated_scale

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Removes radial lens distortion using camera matrix and distortion coefficients if present.

        Args:
            image: Raw camera image array.

        Returns:
            np.ndarray: Undistorted image array.
        """
        if self.config.camera_matrix is None or self.config.dist_coeffs is None:
            return image

        k = np.array(self.config.camera_matrix, dtype=np.float32)
        d = np.array(self.config.dist_coeffs, dtype=np.float32)
        return cv2.undistort(image, k, d)

    def save_config(self, save_path: Union[str, Path]) -> None:
        """Serializes calibration configuration to JSON file."""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved calibration config to: {p}")

    @classmethod
    def load_config(cls, config_path: Union[str, Path]) -> CameraCalibrationConfig:
        """Loads calibration configuration from JSON file."""
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Calibration config not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CameraCalibrationConfig.from_dict(data)
