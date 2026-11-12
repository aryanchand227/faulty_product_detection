"""
Defect Morphology Analysis & Structured Inspection Reporting Module.

Performs granular geometric characterization on segmented defect regions:
- Contour extraction and moment analysis (perimeter, area, equivalent diameter)
- Maximum linear extent / crack length (major-axis length via rotated rect & Feret diameter)
- Geometric shape descriptors:
    * Circularity (Isoperimetric Quotient): 4 * pi * Area / Perimeter^2
    * Aspect Ratio: Major Axis / Minor Axis via minimum area rotated bounding box
    * Solidity: Area / Convex Hull Area
    * Compactness: Perimeter^2 / Area
    * Extent: Area / Axis-Aligned Bounding Box Area
    * Discrete Grid Reliability Flag: Guards against rasterization artifacts on tiny (<25 px) flaws
- Geometric Morphology Typing (Elongated Slit vs Compact Blob vs Diffuse Cluster)
    * NOTE: Morphology labels are 2D geometric shape descriptors, NOT confirmed physical defect types.
- Product-Specific Functional Multi-Zone Overlap (Highest-risk touched rule)
- Stress-Concentration Inspired Heuristic Risk Weighting:
    * NOTE: The 2.2x multiplier for elongated flaws is an engineering heuristic inspired by the
      classical Inglis (1913) notch stress-concentration equation (Kt = 1 + 2 * sqrt(a / rho)).
      It is not a direct physical Griffith energy-balance calculation or 3D stress-intensity factor (KI).
- Authenticated HMAC-SHA256 Inspection Certificates with area reconciliation.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import InspectionVerdict, InspectionDecision, SeverityLevel, RoutingAction
from src.inspection.unified_pipeline import ProductDomain, UnifiedInspectionResult

logger = logging.getLogger(__name__)

# Default HMAC secret key for inspection certificate authentication (overrideable via env var)
DEFAULT_HMAC_SECRET = os.getenv("AOI_HMAC_SECRET_KEY", "aoi-production-secret-key-2026-compliance").encode("utf-8")


class MorphologyType(str, Enum):
    """
    2D geometric shape characterization derived from contour analysis.
    NOTE: These are purely geometric descriptors of the segmented 2D mask,
    not verified physical or metallurgical defect identities (e.g. pit vs fracture).
    """
    GEOMETRIC_ELONGATED_SLIT = "GEOMETRIC_ELONGATED_SLIT"    # High aspect ratio / linear flaw (e.g. crack, scratch)
    GEOMETRIC_COMPACT_BLOB = "GEOMETRIC_COMPACT_BLOB"        # High circularity & solidity (e.g. rounded cavity, blister)
    GEOMETRIC_DIFFUSE_CLUSTER = "GEOMETRIC_DIFFUSE_CLUSTER"  # Low solidity / dispersed non-convex boundary
    GEOMETRIC_IRREGULAR = "GEOMETRIC_IRREGULAR"              # Intermediate irregular morphology

    # Backward compatibility aliases
    FILAMENTARY_CRACK = "GEOMETRIC_ELONGATED_SLIT"
    COMPACT_PIT_OR_INCLUSION = "GEOMETRIC_COMPACT_BLOB"
    COMPACT_PIT = "GEOMETRIC_COMPACT_BLOB"
    DIFFUSE_CONTAMINATION = "GEOMETRIC_DIFFUSE_CLUSTER"
    IRREGULAR_BLEMISH = "GEOMETRIC_IRREGULAR"


class SpatialZone(str, Enum):
    """Product-specific functional inspection zone."""
    # Glass Container Zones (Vertical profile)
    SEALING_RIM = "SEALING_RIM"          # Finish / top sealing surface (extreme leak & safety risk)
    NECK_SHOULDER = "NECK_SHOULDER"      # Transition curvature zone
    BODY_SIDEWALL = "BODY_SIDEWALL"      # Cylindrical container body
    BASE_HEEL = "BASE_HEEL"              # Bottom contact and push-up surface (impact zone)

    # Steel Sheet Zones (Lateral profile)
    EDGE_SHEAR_ZONE = "EDGE_SHEAR_ZONE"  # Outer strip edges (crack initiation risk)
    MAIN_SURFACE = "MAIN_SURFACE"        # Central strip surface

    # Generic
    CENTRAL_ZONE = "CENTRAL_ZONE"
    PERIPHERY_ZONE = "PERIPHERY_ZONE"
    UNKNOWN_ZONE = "UNKNOWN_ZONE"


# Functional zone risk weighting policy parameters
ZONE_RISK_MULTIPLIERS: Dict[SpatialZone, float] = {
    # Glassware policy
    SpatialZone.SEALING_RIM: 2.5,        # Leaks, cap sealing failures, glass ingestion risk
    SpatialZone.NECK_SHOULDER: 1.5,
    SpatialZone.BODY_SIDEWALL: 1.0,
    SpatialZone.BASE_HEEL: 2.0,          # Thermal shock & mechanical conveyor impact
    # Flat Steel policy
    SpatialZone.EDGE_SHEAR_ZONE: 2.0,    # Edge tear propagation in rolling mill
    SpatialZone.MAIN_SURFACE: 1.0,
    # Generic
    SpatialZone.CENTRAL_ZONE: 1.0,
    SpatialZone.PERIPHERY_ZONE: 1.5,
    SpatialZone.UNKNOWN_ZONE: 1.0,
}

# Heuristic risk multipliers inspired by Inglis (1913) notch concentration concepts
# Elongated slits receive higher priority due to notch sensitivity under tensile load.
MORPHOLOGY_RISK_MULTIPLIERS: Dict[MorphologyType, float] = {
    MorphologyType.GEOMETRIC_ELONGATED_SLIT: 2.2,
    MorphologyType.GEOMETRIC_IRREGULAR: 1.4,
    MorphologyType.GEOMETRIC_COMPACT_BLOB: 1.0,
    MorphologyType.GEOMETRIC_DIFFUSE_CLUSTER: 0.7,
}


@dataclass
class MorphologicalFeatures:
    """Granular geometric shape features for an individual localized defect component."""
    contour_id: int
    area_pixels: int
    physical_area_mm2: float
    perimeter_pixels: float
    physical_perimeter_mm: float
    bounding_box_xywh: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    aspect_ratio: float                # Major axis / Minor axis (>= 1.0)
    circularity: float                 # 4 * pi * Area / P^2 in [0.0, 1.0]
    is_shape_reliable: bool            # True if area >= 25 px (guards against discrete rasterization bias)
    solidity: float                    # Area / Convex Hull Area in [0.0, 1.0]
    compactness: float                 # P^2 / Area (>= 4*pi)
    extent: float                      # Area / Bounding Box Area in [0.0, 1.0]
    equivalent_diameter_mm: float      # 2 * sqrt(Area_mm2 / pi)
    major_axis_length_mm: float        # True maximum span / crack length via rotated rect
    max_feret_diameter_mm: float       # Caliper maximum point-to-point Euclidean distance
    morphology_type: MorphologyType
    spatial_zone: SpatialZone          # Primary assigned zone (highest risk touched)
    zone_overlaps: Dict[str, float]    # Percentage distribution across all intersected zones
    zone_risk_weight: float
    morphology_risk_weight: float
    heuristic_severity_score: float    # Flaw prioritization ranking heuristic

    # Backward compatibility alias
    @property
    def adjusted_severity_score(self) -> float:
        return self.heuristic_severity_score

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["morphology_type"] = self.morphology_type.value
        d["spatial_zone"] = self.spatial_zone.value
        d["adjusted_severity_score"] = self.heuristic_severity_score
        return d


@dataclass
class InspectionCertificate:
    """Authenticated inspection certificate with cryptographic HMAC and area reconciliation."""
    certificate_id: str
    timestamp_utc: str
    product_domain: str
    config_version: str
    source_image_hash: str
    decision: str
    severity: str
    routing_action: str
    anomaly_score: float
    primary_defect_type: str
    dominant_morphology: str
    total_defect_area_mm2: float        # Total positive mask area
    component_defect_area_mm2: float    # Sum of analyzed component areas
    filtered_noise_area_mm2: float      # Sub-threshold speckle noise area
    is_area_reconciled: bool            # True if total == component + noise
    max_single_defect_area_mm2: float
    max_defect_length_mm: float         # Maximum major-axis / crack length observed
    measurement_mode: str
    num_defect_components: int
    zone_summary: Dict[str, Dict[str, Any]]
    defect_components: List[Dict[str, Any]]
    engineering_rationale: str
    calibration_scale_mm_per_pixel: float
    sha256_integrity_hash: str          # Content digest of canonical fields
    hmac_sha256_signature: str          # Authenticated signature generated with secret key
    key_id: str = "KEY-2026-V1"
    model_artifact_id: str = "TAC13/industrial-damage-patchcore-bottle@main"
    software_version: str = "1.0.0"
    calibration_profile_id: str = "CALIB-BOTTLE-0.15MM-V1"
    decision_policy_version: str = "POL-v1.0.0"

    # Backward compatibility property
    @property
    def sha256_audit_signature(self) -> str:
        return self.hmac_sha256_signature

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sha256_audit_signature"] = self.hmac_sha256_signature
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Renders inspection certificate as a clean industrial audit document."""
        md = []
        md.append(f"# INDUSTRIAL QUALITY INSPECTION CERTIFICATE")
        md.append(f"**Certificate ID:** `{self.certificate_id}`  ")
        md.append(f"**Timestamp (UTC):** {self.timestamp_utc}  ")
        md.append(f"**Configuration Version:** `{self.config_version}`  ")
        md.append(f"**Product Domain:** `{self.product_domain}`  ")
        md.append(f"**Image SHA-256:** `{self.source_image_hash[:16]}...`  ")
        md.append(f"**HMAC Key Identifier:** `{self.key_id}`  ")
        md.append(f"**Model Artifact ID:** `{self.model_artifact_id}`  ")
        md.append(f"**Software / Policy Version:** `{self.software_version}` / `{self.decision_policy_version}`  ")
        md.append(f"**HMAC-SHA256 Auth Signature:** `{self.hmac_sha256_signature[:24]}...`  \n")
        md.append("---")
        md.append(f"## 1. Quality Triage Verdict")
        md.append(f"- **Final Decision:** **`{self.decision}`**")
        md.append(f"- **Severity Level:** **`{self.severity}`**")
        md.append(f"- **Conveyor Routing Action:** **`{self.routing_action}`**")
        md.append(f"- **Anomaly Score:** `{self.anomaly_score:.2f}`")
        md.append(f"- **Primary Classification:** `{self.primary_defect_type}`")
        md.append(f"- **Dominant Morphology:** `{self.dominant_morphology}` (2D Geometric Descriptor)")
        md.append(f"- **Total Projected Damage Area:** `{self.total_defect_area_mm2:.2f} mm²` ({self.measurement_mode})")
        md.append(f"- **Max Linear Defect Length:** `{self.max_defect_length_mm:.2f} mm`")
        md.append(f"- **Area Reconciliation:** Components: `{self.component_defect_area_mm2:.2f} mm²` | Filtered Noise: `{self.filtered_noise_area_mm2:.2f} mm²` (Reconciled: `{self.is_area_reconciled}`)\n")
        md.append(f"> **Engineering Rationale:** {self.engineering_rationale}\n")
        md.append("---")
        md.append("## 2. Spatial Zone Risk Distribution")
        md.append("| Functional Zone | Defect Count | Total Area (mm²) | Zone Risk Policy |")
        md.append("| :--- | :---: | :---: | :---: |")
        for zone_name, info in self.zone_summary.items():
            md.append(f"| `{zone_name}` | {info['count']} | {info['total_area_mm2']:.2f} mm² | {info['risk_weight']}x |")
        md.append("")
        if self.defect_components:
            md.append("---")
            md.append("## 3. Localized Defect Components (Top 5 by Area)")
            md.append("| # | Morphology | Zone (Overlap) | Area (mm²) | Max Length (mm) | Circ. | Solidity | Heuristic Risk |")
            md.append("| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
            for comp in self.defect_components[:5]:
                md.append(
                    f"| {comp['contour_id']} | `{comp['morphology_type']}` | `{comp['spatial_zone']}` | "
                    f"{comp['physical_area_mm2']:.2f} | {comp['major_axis_length_mm']:.2f} | "
                    f"{comp['circularity']:.2f} | {comp['solidity']:.2f} | **{comp['heuristic_severity_score']:.1f}** |"
                )
        md.append("")
        md.append("---")
        md.append(f"*Verified by Automated Optical Inspection Engine. Camera Scale: {self.calibration_scale_mm_per_pixel:.4f} mm/pixel.*")
        return "\n".join(md)


class DefectMorphologyAnalyzer:
    """
    Extracts advanced morphological descriptors and spatial zone intersections from binary defect masks.
    """

    def __init__(self, min_contour_area_px: int = 10, min_reliable_area_px: int = 25):
        """
        Args:
            min_contour_area_px: Minimum pixel area to consider as a valid defect component
                                  (rejects optical noise/speckle).
            min_reliable_area_px: Threshold above which circularity and boundary descriptors
                                   are free from discrete pixel grid quantization bias.
        """
        self.min_contour_area_px = min_contour_area_px
        self.min_reliable_area_px = min_reliable_area_px

    def analyze_mask(
        self,
        binary_mask: np.ndarray,
        mm_per_pixel: float = 0.15,
        domain: ProductDomain = ProductDomain.GLASS_BOTTLE
    ) -> List[MorphologicalFeatures]:
        """
        Analyzes a 2D binary defect mask and computes morphological descriptors for each contour.
        Implements multi-zone overlap analysis to ensure defects touching high-risk zones inherit
        appropriate safety weighting.
        """
        if binary_mask is None or binary_mask.ndim != 2:
            return []

        # Ensure binary uint8 [0, 255]
        mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255
        img_h, img_w = mask_uint8.shape[:2]

        if np.count_nonzero(mask_uint8) == 0:
            return []

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features_list: List[MorphologicalFeatures] = []
        scale_area = float(mm_per_pixel) ** 2

        contour_id = 0
        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))
            if area_px < self.min_contour_area_px:
                continue

            contour_id += 1
            perimeter_px = float(cv2.arcLength(cnt, closed=True))
            if perimeter_px <= 0.0:
                perimeter_px = 1.0

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = max(float(w * h), 1.0)

            # Centroid via spatial moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)

            # Minimum Area Rotated Rect for true aspect ratio and major axis length
            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                rect_w, rect_h = rect[1]
                major_axis_px = max(rect_w, rect_h)
                minor_axis_px = max(min(rect_w, rect_h), 1.0)
                aspect_ratio = float(major_axis_px / minor_axis_px)
            else:
                major_axis_px = float(max(w, h))
                minor_axis_px = float(max(min(w, h), 1))
                aspect_ratio = float(major_axis_px / minor_axis_px)

            # Maximum Feret diameter (caliper distance)
            hull = cv2.convexHull(cnt)
            hull_pts = hull.reshape(-1, 2)
            if len(hull_pts) >= 2:
                # Pairwise distance maximum on convex hull
                diff = hull_pts[:, np.newaxis, :] - hull_pts[np.newaxis, :, :]
                dist_sq = np.sum(diff ** 2, axis=-1)
                max_feret_px = float(np.sqrt(np.max(dist_sq)))
            else:
                max_feret_px = major_axis_px

            # Shape Descriptors
            # 1. Circularity (Isoperimetric Quotient): 4 * pi * A / P^2
            circularity = float((4.0 * math.pi * area_px) / (perimeter_px ** 2))
            circularity = min(max(circularity, 0.0), 1.0)

            # 2. Convex Hull and Solidity: A / A_hull
            hull_area = max(float(cv2.contourArea(hull)), area_px, 1.0)
            solidity = float(area_px / hull_area)
            solidity = min(max(solidity, 0.0), 1.0)

            # 3. Compactness: P^2 / A
            compactness = float((perimeter_px ** 2) / max(area_px, 1.0))

            # 4. Extent: A / BBox_Area
            extent = float(area_px / bbox_area)
            extent = min(max(extent, 0.0), 1.0)

            # Metric linear and area conversions
            phys_area_mm2 = area_px * scale_area
            phys_perim_mm = perimeter_px * mm_per_pixel
            eq_diam_mm = 2.0 * math.sqrt(phys_area_mm2 / math.pi)
            major_axis_len_mm = major_axis_px * mm_per_pixel
            max_feret_mm = max_feret_px * mm_per_pixel

            # Flag reliability of shape descriptors
            is_reliable = (area_px >= self.min_reliable_area_px)

            # Classify geometric morphology
            morphology = self._classify_morphology(aspect_ratio, circularity, solidity)

            # Multi-zone overlap analysis (Highest-risk touched rule)
            primary_zone, zone_overlaps, zone_weight = self._analyze_zone_overlap(
                cnt, img_w, img_h, domain
            )

            morph_weight = MORPHOLOGY_RISK_MULTIPLIERS.get(morphology, 1.0)

            # Heuristic severity score combines physical damage area with stress multipliers
            heuristic_severity = round(phys_area_mm2 * morph_weight * zone_weight, 2)

            features = MorphologicalFeatures(
                contour_id=contour_id,
                area_pixels=int(area_px),
                physical_area_mm2=round(phys_area_mm2, 3),
                perimeter_pixels=round(perimeter_px, 1),
                physical_perimeter_mm=round(phys_perim_mm, 2),
                bounding_box_xywh=(int(x), int(y), int(w), int(h)),
                centroid_xy=(round(cx, 1), round(cy, 1)),
                aspect_ratio=round(aspect_ratio, 2),
                circularity=round(circularity, 3),
                is_shape_reliable=is_reliable,
                solidity=round(solidity, 3),
                compactness=round(compactness, 2),
                extent=round(extent, 3),
                equivalent_diameter_mm=round(eq_diam_mm, 2),
                major_axis_length_mm=round(major_axis_len_mm, 2),
                max_feret_diameter_mm=round(max_feret_mm, 2),
                morphology_type=morphology,
                spatial_zone=primary_zone,
                zone_overlaps=zone_overlaps,
                zone_risk_weight=zone_weight,
                morphology_risk_weight=morph_weight,
                heuristic_severity_score=heuristic_severity
            )
            features_list.append(features)

        # Sort descending by physical damage area
        features_list.sort(key=lambda f: f.physical_area_mm2, reverse=True)
        return features_list

    @staticmethod
    def _classify_morphology(
        aspect_ratio: float,
        circularity: float,
        solidity: float
    ) -> MorphologyType:
        """
        Classifies 2D geometric shape morphology using deterministic geometric rules.
        """
        # 1. Elongated slit: High aspect ratio or very low circularity with elongation
        if aspect_ratio >= 3.0 or (circularity <= 0.25 and aspect_ratio >= 2.0):
            return MorphologyType.GEOMETRIC_ELONGATED_SLIT

        # 2. Compact blob: High circularity and high solidity
        if circularity >= 0.55 and solidity >= 0.80 and aspect_ratio <= 1.8:
            return MorphologyType.GEOMETRIC_COMPACT_BLOB

        # 3. Diffuse cluster: Low solidity (dispersed/concave) or high spread
        if solidity <= 0.65 and aspect_ratio < 3.0:
            return MorphologyType.GEOMETRIC_DIFFUSE_CLUSTER

        # 4. Irregular morphology
        return MorphologyType.GEOMETRIC_IRREGULAR

    @classmethod
    def _analyze_zone_overlap(
        cls,
        cnt: np.ndarray,
        img_w: int,
        img_h: int,
        domain: ProductDomain
    ) -> Tuple[SpatialZone, Dict[str, float], float]:
        """
        Performs multi-zone overlap analysis for a defect contour.
        Assigns the primary zone as the HIGHEST RISK zone touched by any portion
        of the defect, preventing multi-zone defects from evading critical classification.
        """
        x, y, w, h = cv2.boundingRect(cnt)

        # Draw component raster in local bounding patch
        patch_mask = np.zeros((h, w), dtype=np.uint8)
        cnt_shifted = cnt - np.array([x, y])
        cv2.drawContours(patch_mask, [cnt_shifted], -1, 255, -1)

        total_pts = np.count_nonzero(patch_mask)
        if total_pts == 0:
            return SpatialZone.UNKNOWN_ZONE, {}, 1.0

        py_indices, px_indices = np.where(patch_mask == 255)
        abs_y = py_indices + y
        abs_x = px_indices + x

        norm_y = abs_y / max(img_h, 1)
        norm_x = abs_x / max(img_w, 1)

        zone_counts: Dict[SpatialZone, int] = {}

        if domain == ProductDomain.GLASS_BOTTLE:
            # Vertical stratification
            for ny in norm_y:
                if ny < 0.16:
                    z = SpatialZone.SEALING_RIM
                elif ny < 0.38:
                    z = SpatialZone.NECK_SHOULDER
                elif ny < 0.84:
                    z = SpatialZone.BODY_SIDEWALL
                else:
                    z = SpatialZone.BASE_HEEL
                zone_counts[z] = zone_counts.get(z, 0) + 1

        elif domain == ProductDomain.STEEL_SHEET:
            # Lateral stratification
            for nx in norm_x:
                if nx < 0.10 or nx > 0.90:
                    z = SpatialZone.EDGE_SHEAR_ZONE
                else:
                    z = SpatialZone.MAIN_SURFACE
                zone_counts[z] = zone_counts.get(z, 0) + 1

        else:
            # Radial partition
            for nx, ny in zip(norm_x, norm_y):
                dist = math.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
                z = SpatialZone.PERIPHERY_ZONE if dist > 0.35 else SpatialZone.CENTRAL_ZONE
                zone_counts[z] = zone_counts.get(z, 0) + 1

        # Percentage overlaps
        zone_overlaps = {
            z.value: round(cnt / total_pts, 3) for z, cnt in zone_counts.items()
        }

        # Highest-risk touched rule: assign the zone with highest risk multiplier
        primary_zone = max(
            zone_counts.keys(),
            key=lambda z: ZONE_RISK_MULTIPLIERS.get(z, 1.0)
        )
        max_risk_weight = ZONE_RISK_MULTIPLIERS.get(primary_zone, 1.0)

        return primary_zone, zone_overlaps, max_risk_weight


class InspectionReportGenerator:
    """
    Generates authenticated, cryptographically signed compliance inspection certificates.
    """

    @staticmethod
    def generate_certificate(
        inspection_result: UnifiedInspectionResult,
        morphological_features: Optional[List[MorphologicalFeatures]] = None,
        source_image_bytes: Optional[bytes] = None,
        config_version: str = "CFG-v1.0.0",
        certificate_prefix: str = "CERT",
        hmac_secret: bytes = DEFAULT_HMAC_SECRET,
        key_id: str = "KEY-2026-V1",
        model_artifact_id: str = "TAC13/industrial-damage-patchcore-bottle@main",
        software_version: str = "1.0.0",
        calibration_profile_id: str = "CALIB-BOTTLE-0.15MM-V1",
        decision_policy_version: str = "POL-v1.0.0",
    ) -> InspectionCertificate:
        """
        Synthesizes an end-to-end inspection certificate with HMAC-SHA256 authentication
        and area reconciliation.
        """
        now = datetime.now(timezone.utc)
        time_str = now.strftime("%Y%m%d_%H%M%S")
        timestamp_iso = now.isoformat()

        # Compute source image hash
        if source_image_bytes is not None:
            img_hash = hashlib.sha256(source_image_bytes).hexdigest()
        else:
            raw_id = f"{inspection_result.product_domain}_{inspection_result.total_latency_ms}_{inspection_result.verdict.anomaly_score}"
            img_hash = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

        cert_id = f"{certificate_prefix}-{time_str}-{img_hash[:8].upper()}"

        features = morphological_features or []
        components_dict = [f.to_dict() for f in features]

        # Damage breakdown by spatial zone
        zone_summary: Dict[str, Dict[str, Any]] = {}
        for f in features:
            z_key = f.spatial_zone.value
            if z_key not in zone_summary:
                zone_summary[z_key] = {
                    "count": 0,
                    "total_area_mm2": 0.0,
                    "risk_weight": f.zone_risk_weight
                }
            zone_summary[z_key]["count"] += 1
            zone_summary[z_key]["total_area_mm2"] = round(zone_summary[z_key]["total_area_mm2"] + f.physical_area_mm2, 3)

        # Dominant morphology & maximum crack length
        if features:
            morph_counts: Dict[str, int] = {}
            for f in features:
                morph_counts[f.morphology_type.value] = morph_counts.get(f.morphology_type.value, 0) + 1
            dominant_morph = max(morph_counts.items(), key=lambda kv: kv[1])[0]
            max_single_area = features[0].physical_area_mm2
            max_defect_length = max(f.major_axis_length_mm for f in features)
            sum_component_area = round(sum(f.physical_area_mm2 for f in features), 3)
        else:
            dominant_morph = "NONE" if inspection_result.verdict.decision == InspectionDecision.PASS else "UNIFORM"
            max_single_area = 0.0
            max_defect_length = 0.0
            sum_component_area = 0.0

        total_area = round(float(inspection_result.verdict.physical_defect_area_mm2), 3)

        # Explicit Area Reconciliation:
        # total_defect_area == component_defect_area + filtered_noise_area
        filtered_noise_area = round(max(0.0, total_area - sum_component_area), 3)
        is_reconciled = abs(total_area - (sum_component_area + filtered_noise_area)) < 0.002

        # Canonical data dictionary for hashing (EXCLUDING signature and hash fields)
        canonical_payload = {
            "certificate_id": cert_id,
            "timestamp_utc": timestamp_iso,
            "product_domain": str(inspection_result.product_domain),
            "config_version": config_version,
            "source_image_hash": img_hash,
            "decision": inspection_result.verdict.decision.value,
            "severity": inspection_result.verdict.severity.value,
            "routing_action": inspection_result.verdict.routing.value,
            "anomaly_score": round(inspection_result.verdict.anomaly_score, 4),
            "primary_defect_type": inspection_result.verdict.primary_defect_type,
            "dominant_morphology": dominant_morph,
            "total_defect_area_mm2": total_area,
            "component_defect_area_mm2": sum_component_area,
            "filtered_noise_area_mm2": filtered_noise_area,
            "is_area_reconciled": is_reconciled,
            "max_single_defect_area_mm2": round(max_single_area, 3),
            "max_defect_length_mm": round(max_defect_length, 3),
            "measurement_mode": inspection_result.physical_metrics.get("measurement_mode", "estimated_projected"),
            "num_defect_components": len(features),
            "zone_summary": zone_summary,
            "defect_components": components_dict,
            "engineering_rationale": inspection_result.verdict.engineering_rationale,
            "calibration_scale_mm_per_pixel": round(float(inspection_result.physical_metrics.get("scale_mm_per_pixel", 0.15)), 5),
            "key_id": key_id,
            "model_artifact_id": model_artifact_id,
            "software_version": software_version,
            "calibration_profile_id": calibration_profile_id,
            "decision_policy_version": decision_policy_version,
        }

        # 1. SHA-256 Content Integrity Hash
        canonical_bytes = json.dumps(canonical_payload, sort_keys=True).encode("utf-8")
        integrity_hash = hashlib.sha256(canonical_bytes).hexdigest()

        # 2. Cryptographic HMAC-SHA256 Authentication Signature
        auth_sig = hmac.new(hmac_secret, canonical_bytes, hashlib.sha256).hexdigest()

        canonical_payload["sha256_integrity_hash"] = integrity_hash
        canonical_payload["hmac_sha256_signature"] = auth_sig

        return InspectionCertificate(**canonical_payload)

    @staticmethod
    def verify_certificate(
        certificate_data: Dict[str, Any],
        hmac_secret: Optional[bytes] = None,
        key_registry: Optional[Dict[str, bytes]] = None,
    ) -> bool:
        """
        Verifies the cryptographic HMAC authenticity and SHA-256 integrity of an inspection certificate.
        Supports key rotation via key_registry lookup based on certificate key_id.
        Returns True if the certificate is authentic and untampered.
        """
        cert_copy = dict(certificate_data)
        reported_sig = cert_copy.pop("hmac_sha256_signature", None)
        reported_hash = cert_copy.pop("sha256_integrity_hash", None)
        cert_copy.pop("sha256_audit_signature", None)  # Remove alias if present

        if not reported_sig or not reported_hash:
            return False

        # Resolve secret key
        resolved_key = hmac_secret
        if resolved_key is None:
            key_id = cert_copy.get("key_id", "KEY-2026-V1")
            if key_registry and key_id in key_registry:
                resolved_key = key_registry[key_id]
            else:
                resolved_key = DEFAULT_HMAC_SECRET

        canonical_bytes = json.dumps(cert_copy, sort_keys=True).encode("utf-8")
        computed_hash = hashlib.sha256(canonical_bytes).hexdigest()
        computed_sig = hmac.new(resolved_key, canonical_bytes, hashlib.sha256).hexdigest()

        is_hash_valid = hmac.compare_digest(reported_hash, computed_hash)
        is_sig_valid = hmac.compare_digest(reported_sig, computed_sig)

        return is_hash_valid and is_sig_valid
