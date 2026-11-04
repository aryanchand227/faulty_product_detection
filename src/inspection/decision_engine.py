"""
Industrial Quality Decision Engine for Multi-Tier Inspection.

Synthesizes findings across Tier 1 (PatchCore anomaly score), Tier 2 (defect classification),
and Tier 3 (YOLO bounding boxes + U-Net pixel segmentation) with calibrated physical metrics
into deterministic operational triage verdicts.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class InspectionDecision(str, Enum):
    """Primary operational triage decision for manufactured product."""
    PASS = "PASS"        # Nominal surface meeting quality standards
    REVIEW = "REVIEW"    # Marginal anomaly or uncertain defect requiring operator confirmation
    FAIL = "FAIL"        # Confirmed defect violating quality specifications; rejected


class SeverityLevel(str, Enum):
    """Categorical severity rating for identified product damage."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    CRITICAL = "CRITICAL"


class RoutingAction(str, Enum):
    """Physical conveyor routing destination in automated factory line."""
    LINE_CONTINUE = "LINE_CONTINUE"                     # Packaging and shipping
    MANUAL_INSPECTION_BAY = "MANUAL_INSPECTION_BAY"     # Operator visual inspection station
    AUTOMATIC_REJECT_BIN = "AUTOMATIC_REJECT_BIN"       # Pneumatic ejection to scrap/recycle chute


@dataclass
class QualityThresholds:
    """Configurable industrial tolerance thresholds and rejection limits."""
    anomaly_threshold: float = 15.5                # Operational PatchCore cutoff (tau)
    anomaly_margin: float = 0.15                   # Uncertainty margin delta (+/- 15%)
    max_acceptable_area_mm2: float = 0.50          # Flaws below this may be tolerated or reviewed
    critical_area_mm2: float = 5.00                # Flaws exceeding this trigger automatic rejection
    critical_classes: List[str] = field(default_factory=lambda: [
        "crazing", "broken_large", "deep_crack", "puncture"
    ])
    min_classifier_confidence: float = 0.60        # Lower confidence triggers REVIEW
    max_entropy_threshold: float = 2.0             # Shannon entropy ceiling for classification certainty

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InspectionVerdict:
    """
    Comprehensive industrial triage verdict synthesizing all vision tiers.
    """
    decision: InspectionDecision
    severity: SeverityLevel
    routing: RoutingAction
    primary_defect_type: str
    confidence: float
    physical_defect_area_mm2: float
    anomaly_score: float
    is_anomalous: bool
    rejection_reasons: List[str] = field(default_factory=list)
    engineering_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        data["severity"] = self.severity.value
        data["routing"] = self.routing.value
        return data


class DecisionEngine:
    """
    Industrial rule engine evaluating multi-tier optical inspection telemetry.
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None) -> None:
        self.thresholds = thresholds or QualityThresholds()
        self._critical_classes_set: Set[str] = {c.lower() for c in self.thresholds.critical_classes}

    def evaluate(
        self,
        anomaly_score: float,
        is_anomalous: bool,
        defect_class: Optional[str] = None,
        class_confidence: float = 1.0,
        class_entropy: float = 0.0,
        pixel_defect_area_mm2: float = 0.0,
        bbox_union_area_mm2: float = 0.0,
        num_defect_regions: int = 0
    ) -> InspectionVerdict:
        """
        Synthesizes multi-tier telemetry into an industrial inspection verdict.

        Args:
            anomaly_score: Continuous score from Tier 1 PatchCore.
            is_anomalous: Flag indicating whether score >= anomaly_threshold.
            defect_class: Predicted defect morphology from Tier 2 classifier.
            class_confidence: Classifier softmax confidence [0.0, 1.0].
            class_entropy: Shannon entropy in bits from Tier 2.
            pixel_defect_area_mm2: Calibrated physical damaged area from U-Net segmentation.
            bbox_union_area_mm2: Calibrated physical area of YOLO bounding box union.
            num_defect_regions: Number of distinct localized defect regions.

        Returns:
            InspectionVerdict: Structured industrial verdict and routing directive.
        """
        tau = self.thresholds.anomaly_threshold
        delta = self.thresholds.anomaly_margin
        tau_lower = tau * (1.0 - delta)
        tau_upper = tau * (1.0 + delta)

        effective_area_mm2 = max(pixel_defect_area_mm2, 0.0)
        # If segmentation area is 0 but bbox detected substantial defect, use conservative portion of bbox
        if effective_area_mm2 == 0.0 and bbox_union_area_mm2 > 0:
            effective_area_mm2 = bbox_union_area_mm2 * 0.35  # Approximate defect fill factor

        defect_name = (defect_class or "nominal").lower()
        is_nominal_class = defect_name in {"good", "nominal", "none", "normal"}

        reasons: List[str] = []

        # -----------------------------------------------------------------
        # Tier 1 Gatekeeper Early Check: Pure Nominal Part
        # -----------------------------------------------------------------
        if anomaly_score < tau_lower and num_defect_regions == 0 and effective_area_mm2 == 0.0:
            return InspectionVerdict(
                decision=InspectionDecision.PASS,
                severity=SeverityLevel.NONE,
                routing=RoutingAction.LINE_CONTINUE,
                primary_defect_type="nominal",
                confidence=round(1.0 - (anomaly_score / tau), 4) if tau > 0 else 1.0,
                physical_defect_area_mm2=0.0,
                anomaly_score=round(anomaly_score, 4),
                is_anomalous=False,
                rejection_reasons=[],
                engineering_rationale=(
                    f"Tier 1 Gatekeeper PASSED: Anomaly score {anomaly_score:.2f} is well below "
                    f"threshold {tau:.2f} (safety margin {tau_lower:.2f}). No defect regions detected."
                )
            )

        # -----------------------------------------------------------------
        # Critical Defect Condition Checks (Automatic FAIL)
        # -----------------------------------------------------------------
        # 1. Critical defect class detected with sufficient confidence
        if not is_nominal_class and defect_name in self._critical_classes_set:
            if class_confidence >= self.thresholds.min_classifier_confidence:
                reasons.append(
                    f"Critical defect category '{defect_name}' identified with {class_confidence:.1%} confidence."
                )

        # 2. Severe physical damage surface area
        if effective_area_mm2 >= self.thresholds.critical_area_mm2:
            reasons.append(
                f"Calibrated physical damaged area ({effective_area_mm2:.2f} mm²) exceeds "
                f"critical limit ({self.thresholds.critical_area_mm2:.2f} mm²)."
            )

        # 3. High anomaly score exceeding upper confidence margin
        if anomaly_score >= tau_upper:
            reasons.append(
                f"Severe anomaly score ({anomaly_score:.2f}) exceeds upper margin ({tau_upper:.2f})."
            )

        if reasons:
            return InspectionVerdict(
                decision=InspectionDecision.FAIL,
                severity=SeverityLevel.CRITICAL,
                routing=RoutingAction.AUTOMATIC_REJECT_BIN,
                primary_defect_type=defect_name,
                confidence=round(class_confidence, 4),
                physical_defect_area_mm2=round(effective_area_mm2, 3),
                anomaly_score=round(anomaly_score, 4),
                is_anomalous=True,
                rejection_reasons=reasons,
                engineering_rationale=(
                    f"Product REJECTED: {'; '.join(reasons)} "
                    f"Conveyor pneumatic diverter triggered to scrap bin."
                )
            )

        # -----------------------------------------------------------------
        # Borderline / Ambiguous Condition Checks (REVIEW)
        # -----------------------------------------------------------------
        review_reasons: List[str] = []

        # Score in uncertainty margin band [tau_lower, tau_upper]
        if tau_lower <= anomaly_score < tau_upper:
            review_reasons.append(
                f"Anomaly score ({anomaly_score:.2f}) lies in borderline uncertainty band "
                f"[{tau_lower:.2f}, {tau_upper:.2f}]."
            )

        # Non-nominal class identified, but area is moderate
        if not is_nominal_class and effective_area_mm2 > self.thresholds.max_acceptable_area_mm2:
            review_reasons.append(
                f"Detected defect '{defect_name}' with area {effective_area_mm2:.2f} mm² "
                f"(tolerance: {self.thresholds.max_acceptable_area_mm2:.2f} mm²)."
            )

        # High classifier uncertainty (low confidence or high entropy)
        if class_confidence < self.thresholds.min_classifier_confidence and num_defect_regions > 0:
            review_reasons.append(
                f"Classifier confidence ({class_confidence:.1%}) below reliable minimum "
                f"({self.thresholds.min_classifier_confidence:.1%})."
            )

        if review_reasons:
            severity = SeverityLevel.MEDIUM if effective_area_mm2 > 1.0 else SeverityLevel.LOW
            return InspectionVerdict(
                decision=InspectionDecision.REVIEW,
                severity=severity,
                routing=RoutingAction.MANUAL_INSPECTION_BAY,
                primary_defect_type=defect_name,
                confidence=round(class_confidence, 4),
                physical_defect_area_mm2=round(effective_area_mm2, 3),
                anomaly_score=round(anomaly_score, 4),
                is_anomalous=is_anomalous,
                rejection_reasons=review_reasons,
                engineering_rationale=(
                    f"Diverted for OPERATOR REVIEW: {'; '.join(review_reasons)} "
                    f"Routed to manual QA station."
                )
            )

        # -----------------------------------------------------------------
        # Sub-threshold Flaws Meeting Tolerance (PASS with optional note)
        # -----------------------------------------------------------------
        return InspectionVerdict(
            decision=InspectionDecision.PASS,
            severity=SeverityLevel.LOW if effective_area_mm2 > 0 else SeverityLevel.NONE,
            routing=RoutingAction.LINE_CONTINUE,
            primary_defect_type=defect_name if effective_area_mm2 > 0 else "nominal",
            confidence=round(class_confidence, 4),
            physical_defect_area_mm2=round(effective_area_mm2, 3),
            anomaly_score=round(anomaly_score, 4),
            is_anomalous=False,
            rejection_reasons=[],
            engineering_rationale=(
                f"Product PASSED: Anomaly score {anomaly_score:.2f} < {tau:.2f}. "
                f"Detected micro-flaw area ({effective_area_mm2:.2f} mm²) within allowable tolerance "
                f"({self.thresholds.max_acceptable_area_mm2:.2f} mm²)."
            )
        )
