"""
Defect Morphology Analysis & Structured Inspection Reporting Module.

Performs granular geometric characterization on segmented defect regions:
- Contour extraction and moment analysis (perimeter, area, equivalent diameter)
- Geometric shape descriptors:
    * Circularity (Isoperimetric Quotient): 4 * pi * Area / Perimeter^2
    * Aspect Ratio: Major Axis / Minor Axis via minimum area rotated bounding box
    * Solidity: Area / Convex Hull Area
    * Compactness: Perimeter^2 / Area
    * Extent: Area / Axis-Aligned Bounding Box Area
- Defect Morphological Typing (Filamentary Crack vs Compact Pit vs Diffuse Contamination)
- Product-Specific Spatial Zoning (e.g. Bottle Sealing Rim vs Body vs Base; Steel Edges)
- Griffith Fracture Mechanics Risk Weighting (Stress concentration multipliers)
- Cryptographically verifiable (SHA-256) structured inspection certificates for industrial audit trails.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.inspection.camera_calibration import CameraCalibrator
from src.inspection.decision_engine import InspectionVerdict, InspectionDecision, SeverityLevel, RoutingAction
from src.inspection.unified_pipeline import ProductDomain, UnifiedInspectionResult

logger = logging.getLogger(__name__)


class MorphologyType(str, Enum):
    """Morphological classification of localized material defect."""
    FILAMENTARY_CRACK = "FILAMENTARY_CRACK"        # Elongated, high stress concentration (crack, scratch)
    COMPACT_PIT_OR_INCLUSION = "COMPACT_PIT"       # Circular/compact cavity, low notch factor (pit, pinhole, bubble)
    DIFFUSE_CONTAMINATION = "DIFFUSE_CONTAMINATION" # Low solidity, dispersed surface blemish (liquid, dirt, smirch)
    IRREGULAR_BLEMISH = "IRREGULAR_BLEMISH"        # Complex non-convex morphology


class SpatialZone(str, Enum):
    """Product-specific functional inspection zone."""
    # Glass Container Zones
    SEALING_RIM = "SEALING_RIM"          # Finish / top sealing surface (extreme leak & safety risk)
    NECK_SHOULDER = "NECK_SHOULDER"      # Transition curvature zone
    BODY_SIDEWALL = "BODY_SIDEWALL"      # Cylindrical container body
    BASE_HEEL = "BASE_HEEL"              # Bottom contact and push-up surface (impact zone)

    # Steel Sheet Zones
    EDGE_SHEAR_ZONE = "EDGE_SHEAR_ZONE"  # Outer strip edges (crack initiation risk)
    MAIN_SURFACE = "MAIN_SURFACE"        # Central strip surface

    # Generic
    CENTRAL_ZONE = "CENTRAL_ZONE"
    PERIPHERY_ZONE = "PERIPHERY_ZONE"
    UNKNOWN_ZONE = "UNKNOWN_ZONE"


# Zone risk weighting multipliers
ZONE_RISK_MULTIPLIERS: Dict[SpatialZone, float] = {
    # Glassware
    SpatialZone.SEALING_RIM: 2.5,        # Leaks, cap sealing failures, glass ingestion risk
    SpatialZone.NECK_SHOULDER: 1.5,
    SpatialZone.BODY_SIDEWALL: 1.0,
    SpatialZone.BASE_HEEL: 2.0,          # Thermal shock & mechanical conveyor impact
    # Flat Steel
    SpatialZone.EDGE_SHEAR_ZONE: 2.0,    # Edge tear propagation in rolling mill
    SpatialZone.MAIN_SURFACE: 1.0,
    # Generic
    SpatialZone.CENTRAL_ZONE: 1.0,
    SpatialZone.PERIPHERY_ZONE: 1.5,
    SpatialZone.UNKNOWN_ZONE: 1.0,
}

# Morphology risk weighting multipliers based on stress concentration (Griffith theory)
MORPHOLOGY_RISK_MULTIPLIERS: Dict[MorphologyType, float] = {
    MorphologyType.FILAMENTARY_CRACK: 2.2,       # High stress concentration factor Kt
    MorphologyType.IRREGULAR_BLEMISH: 1.4,
    MorphologyType.COMPACT_PIT_OR_INCLUSION: 1.0,
    MorphologyType.DIFFUSE_CONTAMINATION: 0.7,   # Non-structural cosmetic flaw
}


@dataclass
class MorphologicalFeatures:
    """Granular geometric shape features for a single localized defect component."""
    contour_id: int
    area_pixels: int
    physical_area_mm2: float
    perimeter_pixels: float
    physical_perimeter_mm: float
    bounding_box_xywh: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    aspect_ratio: float                # Major axis / Minor axis (>= 1.0)
    circularity: float                 # 4 * pi * Area / P^2 in [0.0, 1.0]
    solidity: float                    # Area / Convex Hull Area in [0.0, 1.0]
    compactness: float                 # P^2 / Area (>= 4*pi)
    extent: float                      # Area / Bounding Box Area in [0.0, 1.0]
    equivalent_diameter_mm: float      # 2 * sqrt(Area_mm2 / pi)
    morphology_type: MorphologyType
    spatial_zone: SpatialZone
    zone_risk_weight: float
    morphology_risk_weight: float
    adjusted_severity_score: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["morphology_type"] = self.morphology_type.value
        d["spatial_zone"] = self.spatial_zone.value
        return d


@dataclass
class InspectionCertificate:
    """Cryptographically verifiable inspection certificate for industrial compliance."""
    certificate_id: str
    timestamp_utc: str
    product_domain: str
    source_image_hash: str
    decision: str
    severity: str
    routing_action: str
    anomaly_score: float
    primary_defect_type: str
    dominant_morphology: str
    total_defect_area_mm2: float
    max_single_defect_area_mm2: float
    measurement_mode: str
    num_defect_components: int
    zone_summary: Dict[str, Dict[str, Any]]
    defect_components: List[Dict[str, Any]]
    engineering_rationale: str
    calibration_scale_mm_per_pixel: float
    sha256_audit_signature: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Renders inspection certificate as a clean industrial audit document."""
        md = []
        md.append(f"# INDUSTRIAL QUALITY INSPECTION CERTIFICATE")
        md.append(f"**Certificate ID:** `{self.certificate_id}`  ")
        md.append(f"**Timestamp (UTC):** {self.timestamp_utc}  ")
        md.append(f"**Product Domain:** `{self.product_domain}`  ")
        md.append(f"**Image SHA-256:** `{self.source_image_hash[:16]}...`  ")
        md.append(f"**Audit Signature:** `{self.sha256_audit_signature[:24]}...`  \n")
        md.append("---")
        md.append(f"## 1. Quality Triage Verdict")
        md.append(f"- **Final Decision:** **`{self.decision}`**")
        md.append(f"- **Severity Level:** **`{self.severity}`**")
        md.append(f"- **Conveyor Routing Action:** **`{self.routing_action}`**")
        md.append(f"- **Anomaly Score:** `{self.anomaly_score:.2f}`")
        md.append(f"- **Primary Classification:** `{self.primary_defect_type}`")
        md.append(f"- **Dominant Morphology:** `{self.dominant_morphology}`")
        md.append(f"- **Total Projected Damage Area:** `{self.total_defect_area_mm2:.2f} mm²` ({self.measurement_mode})\n")
        md.append(f"> **Engineering Rationale:** {self.engineering_rationale}\n")
        md.append("---")
        md.append("## 2. Spatial Zone Risk Distribution")
        md.append("| Functional Zone | Defect Count | Total Area (mm²) | Zone Risk Weight |")
        md.append("| :--- | :---: | :---: | :---: |")
        for zone_name, info in self.zone_summary.items():
            md.append(f"| `{zone_name}` | {info['count']} | {info['total_area_mm2']:.2f} mm² | {info['risk_weight']}x |")
        md.append("")
        if self.defect_components:
            md.append("---")
            md.append("## 3. Localized Defect Components (Top 5 by Area)")
            md.append("| # | Morphology | Zone | Area (mm²) | Circ. | Solidity | Aspect Ratio | Adjusted Risk |")
            md.append("| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
            for comp in self.defect_components[:5]:
                md.append(
                    f"| {comp['contour_id']} | `{comp['morphology_type']}` | `{comp['spatial_zone']}` | "
                    f"{comp['physical_area_mm2']:.2f} | {comp['circularity']:.2f} | {comp['solidity']:.2f} | "
                    f"{comp['aspect_ratio']:.1f}:1 | **{comp['adjusted_severity_score']:.1f}** |"
                )
        md.append("")
        md.append("---")
        md.append(f"*Verified by Automated Optical Inspection Engine. Camera Scale: {self.calibration_scale_mm_per_pixel:.4f} mm/pixel.*")
        return "\n".join(md)


class DefectMorphologyAnalyzer:
    """
    Extracts advanced morphological and spatial features from binary defect segmentation masks.
    """

    def __init__(self, min_contour_area_px: int = 10):
        """
        Args:
            min_contour_area_px: Minimum pixel area to consider as a valid defect component
                                  (rejects sub-pixel optical salt-and-pepper noise).
        """
        self.min_contour_area_px = min_contour_area_px

    def analyze_mask(
        self,
        binary_mask: np.ndarray,
        mm_per_pixel: float = 0.15,
        domain: ProductDomain = ProductDomain.GLASS_BOTTLE
    ) -> List[MorphologicalFeatures]:
        """
        Analyzes a 2D binary defect mask and computes morphological descriptors for each contour.

        Args:
            binary_mask: 2D uint8 NumPy array where defect pixels are > 0.
            mm_per_pixel: Spatial calibration scale factor in mm per pixel.
            domain: Product domain governing spatial zoning logic.

        Returns:
            List of MorphologicalFeatures dataclasses sorted in descending order of physical area.
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

            # Bounding boxes
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

            # Minimum Area Rotated Rect for true aspect ratio
            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                rect_w, rect_h = rect[1]
                major_axis = max(rect_w, rect_h)
                minor_axis = max(min(rect_w, rect_h), 1.0)
                aspect_ratio = float(major_axis / minor_axis)
            else:
                aspect_ratio = float(max(w, h) / max(min(w, h), 1))

            # Shape Descriptors
            # 1. Circularity (Isoperimetric Quotient): 4 * pi * A / P^2
            circularity = float((4.0 * math.pi * area_px) / (perimeter_px ** 2))
            circularity = min(max(circularity, 0.0), 1.0)

            # 2. Convex Hull and Solidity: A / A_hull
            hull = cv2.convexHull(cnt)
            hull_area = max(float(cv2.contourArea(hull)), area_px, 1.0)
            solidity = float(area_px / hull_area)
            solidity = min(max(solidity, 0.0), 1.0)

            # 3. Compactness: P^2 / A
            compactness = float((perimeter_px ** 2) / max(area_px, 1.0))

            # 4. Extent: A / BBox_Area
            extent = float(area_px / bbox_area)
            extent = min(max(extent, 0.0), 1.0)

            # Metric conversions
            phys_area_mm2 = area_px * scale_area
            phys_perim_mm = perimeter_px * mm_per_pixel
            eq_diam_mm = 2.0 * math.sqrt(phys_area_mm2 / math.pi)

            # Classify morphology
            morphology = self._classify_morphology(aspect_ratio, circularity, solidity)

            # Determine spatial zone
            spatial_zone = self._determine_spatial_zone(cx, cy, img_w, img_h, domain)

            # Compute risk weights
            zone_weight = ZONE_RISK_MULTIPLIERS.get(spatial_zone, 1.0)
            morph_weight = MORPHOLOGY_RISK_MULTIPLIERS.get(morphology, 1.0)

            # Adjusted severity score combines physical damage area with stress multipliers
            adjusted_severity = round(phys_area_mm2 * morph_weight * zone_weight, 2)

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
                solidity=round(solidity, 3),
                compactness=round(compactness, 2),
                extent=round(extent, 3),
                equivalent_diameter_mm=round(eq_diam_mm, 2),
                morphology_type=morphology,
                spatial_zone=spatial_zone,
                zone_risk_weight=zone_weight,
                morphology_risk_weight=morph_weight,
                adjusted_severity_score=adjusted_severity
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
        Classifies defect morphology using geometric rules.
        """
        # 1. Filamentary Crack / Scratch: High aspect ratio or very low circularity with elongation
        if aspect_ratio >= 3.0 or (circularity <= 0.25 and aspect_ratio >= 2.0):
            return MorphologyType.FILAMENTARY_CRACK

        # 2. Compact Pit / Inclusion: High circularity and high solidity
        if circularity >= 0.55 and solidity >= 0.80 and aspect_ratio <= 1.8:
            return MorphologyType.COMPACT_PIT_OR_INCLUSION

        # 3. Diffuse Contamination: Low solidity (dispersed/concave) or high spread
        if solidity <= 0.65 and aspect_ratio < 3.0:
            return MorphologyType.DIFFUSE_CONTAMINATION

        # 4. Irregular blemish
        return MorphologyType.IRREGULAR_BLEMISH

    @staticmethod
    def _determine_spatial_zone(
        cx: float,
        cy: float,
        img_w: int,
        img_h: int,
        domain: ProductDomain
    ) -> SpatialZone:
        """
        Determines the functional product zone containing the defect centroid.
        """
        norm_y = cy / max(img_h, 1)
        norm_x = cx / max(img_w, 1)

        if domain == ProductDomain.GLASS_BOTTLE:
            # Vertical stratification for cylindrical bottle
            if norm_y < 0.16:
                return SpatialZone.SEALING_RIM     # Top finish & sealing lip
            elif norm_y < 0.38:
                return SpatialZone.NECK_SHOULDER   # Neck curve and shoulder
            elif norm_y < 0.84:
                return SpatialZone.BODY_SIDEWALL   # Main cylindrical body
            else:
                return SpatialZone.BASE_HEEL       # Bottom push-up & conveyor contact heel

        elif domain == ProductDomain.STEEL_SHEET:
            # Lateral stratification for rolling steel strip
            if norm_x < 0.10 or norm_x > 0.90:
                return SpatialZone.EDGE_SHEAR_ZONE # Strip edge trim
            else:
                return SpatialZone.MAIN_SURFACE

        else:
            # Generic radial partition
            dist_from_center = math.sqrt((norm_x - 0.5) ** 2 + (norm_y - 0.5) ** 2)
            if dist_from_center > 0.35:
                return SpatialZone.PERIPHERY_ZONE
            return SpatialZone.CENTRAL_ZONE


class InspectionReportGenerator:
    """
    Generates structured, cryptographically signed compliance inspection certificates.
    """

    @staticmethod
    def generate_certificate(
        inspection_result: UnifiedInspectionResult,
        morphological_features: Optional[List[MorphologicalFeatures]] = None,
        source_image_bytes: Optional[bytes] = None,
        certificate_prefix: str = "CERT"
    ) -> InspectionCertificate:
        """
        Synthesizes an end-to-end inspection certificate with SHA-256 integrity hash.
        """
        now = datetime.now(timezone.utc)
        time_str = now.strftime("%Y%m%d_%H%M%S")
        timestamp_iso = now.isoformat()

        # Compute source image hash or generate fallback
        if source_image_bytes is not None:
            img_hash = hashlib.sha256(source_image_bytes).hexdigest()
        else:
            raw_id = f"{inspection_result.product_domain}_{inspection_result.total_latency_ms}_{inspection_result.verdict.anomaly_score}"
            img_hash = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

        cert_id = f"{certificate_prefix}-{time_str}-{img_hash[:8].upper()}"

        features = morphological_features or []
        components_dict = [f.to_dict() for f in features]

        # Summarize damage by spatial zone
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

        # Determine dominant morphology
        if features:
            morph_counts: Dict[str, int] = {}
            for f in features:
                morph_counts[f.morphology_type.value] = morph_counts.get(f.morphology_type.value, 0) + 1
            dominant_morph = max(morph_counts.items(), key=lambda kv: kv[1])[0]
            max_single_area = features[0].physical_area_mm2
        else:
            dominant_morph = "NONE" if inspection_result.verdict.decision == InspectionDecision.PASS else "UNIFORM"
            max_single_area = 0.0

        # Create pre-signature dictionary for canonical hashing
        payload = {
            "certificate_id": cert_id,
            "timestamp_utc": timestamp_iso,
            "product_domain": str(inspection_result.product_domain),
            "source_image_hash": img_hash,
            "decision": inspection_result.verdict.decision.value,
            "severity": inspection_result.verdict.severity.value,
            "routing_action": inspection_result.verdict.routing.value,
            "anomaly_score": round(inspection_result.verdict.anomaly_score, 4),
            "primary_defect_type": inspection_result.verdict.primary_defect_type,
            "dominant_morphology": dominant_morph,
            "total_defect_area_mm2": round(inspection_result.verdict.physical_defect_area_mm2, 3),
            "max_single_defect_area_mm2": round(max_single_area, 3),
            "measurement_mode": inspection_result.physical_metrics.get("measurement_mode", "estimated_projected"),
            "num_defect_components": len(features),
            "zone_summary": zone_summary,
            "defect_components": components_dict,
            "engineering_rationale": inspection_result.verdict.engineering_rationale,
            "calibration_scale_mm_per_pixel": round(float(inspection_result.physical_metrics.get("scale_mm_per_pixel", 0.15)), 5),
        }

        # Compute SHA-256 audit digest of canonical JSON payload
        canonical_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        audit_sig = hashlib.sha256(canonical_bytes).hexdigest()
        payload["sha256_audit_signature"] = audit_sig

        return InspectionCertificate(**payload)
