"""
Unified Multi-Tier Visual Inspection Dashboard & Industrial HUD Overlay.

Renders production-grade annotated visual outputs combining:
- Tier 1 PatchCore anomaly heatmaps
- Tier 2 Classification tags and certainty telemetry
- Tier 3 YOLO bounding boxes and PyTorch U-Net segmentation contours
- Calibrated physical dimensions (mm and mm²)
- Industrial conveyor status HUD banner (PASS / REVIEW / FAIL)
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.inference.detector_inference import DEFECT_COLOR_PALETTE, DetectionBox
from src.inspection.decision_engine import InspectionDecision, InspectionVerdict

# Status Banner Color Scheme (RGB)
STATUS_COLORS_RGB: Dict[InspectionDecision, Tuple[int, int, int]] = {
    InspectionDecision.PASS: (34, 177, 76),     # Industrial Green
    InspectionDecision.REVIEW: (255, 165, 0),   # Caution Amber / Orange
    InspectionDecision.FAIL: (220, 30, 30)      # Rejection Red
}


class InspectionVisualizer:
    """
    Renders unified multi-tier inspection overlays and comparative dashboards.
    """

    @staticmethod
    def render_hud_overlay(
        image_rgb: np.ndarray,
        verdict: InspectionVerdict,
        anomaly_heatmap: Optional[np.ndarray] = None,
        boxes: Optional[List[DetectionBox]] = None,
        segmentation_mask: Optional[np.ndarray] = None,
        latency_ms: float = 0.0,
        heatmap_alpha: float = 0.30,
        mask_alpha: float = 0.40
    ) -> np.ndarray:
        """
        Draws comprehensive multi-tier annotations and an industrial HUD header.

        Args:
            image_rgb: Base image array (H, W, 3) in uint8 RGB format.
            verdict: InspectionVerdict from DecisionEngine.
            anomaly_heatmap: Continuous anomaly map (H, W) or (224, 224) in [0, max].
            boxes: Optional localized defect bounding boxes from Tier 3 YOLO.
            segmentation_mask: Optional binary defect segmentation mask (H, W).
            latency_ms: Total pipeline processing latency in milliseconds.
            heatmap_alpha: Blending weight for anomaly heatmap.
            mask_alpha: Blending weight for defect segmentation mask.

        Returns:
            np.ndarray: Annotated RGB image with HUD banner.
        """
        canvas = image_rgb.copy()
        img_h, img_w = canvas.shape[:2]

        # -------------------------------------------------------------
        # 1. Blend PatchCore Anomaly Heatmap (if part is anomalous/review)
        # -------------------------------------------------------------
        if anomaly_heatmap is not None and verdict.decision != InspectionDecision.PASS:
            h_resized = cv2.resize(anomaly_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 255]
            h_min, h_max = float(np.min(h_resized)), float(np.max(h_resized))
            if h_max > h_min:
                h_norm = ((h_resized - h_min) / (h_max - h_min) * 255.0).astype(np.uint8)
            else:
                h_norm = np.zeros((img_h, img_w), dtype=np.uint8)

            color_map = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
            color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

            # Apply only to regions above background threshold to prevent washed out look
            thresh_val = int(255.0 * 0.40)
            heat_mask = (h_norm >= thresh_val)[:, :, None]
            canvas = np.where(
                heat_mask,
                cv2.addWeighted(canvas, 1.0 - heatmap_alpha, color_map_rgb, heatmap_alpha, 0),
                canvas
            )

        # -------------------------------------------------------------
        # 2. Blend U-Net Semantic Defect Mask & Contours
        # -------------------------------------------------------------
        if segmentation_mask is not None:
            m_resized = cv2.resize(
                (segmentation_mask > 0).astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )
            if np.count_nonzero(m_resized) > 0:
                # Magenta overlay for pixel-accurate damaged surface
                mask_color = np.array([255, 0, 128], dtype=np.uint8)
                colored_mask = np.zeros_like(canvas)
                colored_mask[m_resized > 0] = mask_color

                canvas = np.where(
                    m_resized[:, :, None] > 0,
                    cv2.addWeighted(canvas, 1.0 - mask_alpha, colored_mask, mask_alpha, 0),
                    canvas
                )

                # Crisp boundary contour
                contours, _ = cv2.findContours(m_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)  # Cyan contour

        # -------------------------------------------------------------
        # 3. Draw YOLO Bounding Boxes
        # -------------------------------------------------------------
        if boxes:
            for b in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in b.box_xyxy]
                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w, x2))
                y2 = max(0, min(img_h, y2))

                # Color lookup
                color_bgr = DEFECT_COLOR_PALETTE.get(b.class_name.lower(), (0, 255, 255))
                box_color = (color_bgr[2], color_bgr[1], color_bgr[0])  # Convert BGR to RGB

                # Box rectangle
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

                # Label tag
                tag = f"{b.class_name} {b.confidence:.0%}"
                (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                tag_y1 = max(0, y1 - th - baseline - 4)
                tag_y2 = y1
                cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 6, tag_y2), box_color, -1)
                cv2.putText(
                    canvas,
                    tag,
                    (x1 + 3, tag_y2 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # -------------------------------------------------------------
        # 4. Top Industrial Status HUD Banner
        # -------------------------------------------------------------
        banner_height = 56
        hud_canvas = np.zeros((img_h + banner_height, img_w, 3), dtype=np.uint8)

        # Draw dark industrial banner background
        hud_canvas[:banner_height, :] = (24, 26, 32)
        hud_canvas[banner_height:, :] = canvas

        status_color = STATUS_COLORS_RGB.get(verdict.decision, (180, 180, 180))

        # 1. Left status badge
        badge_text = f" {verdict.decision.value} "
        badge_routing = f"[{verdict.routing.value}]"
        cv2.rectangle(hud_canvas, (10, 8), (140, 48), status_color, -1)
        cv2.putText(
            hud_canvas,
            badge_text,
            (14, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 2. Defect & Severity Telemetry
        telemetry_line1 = (
            f"Defect: {verdict.primary_defect_type.upper()} "
            f"| Severity: {verdict.severity.value} "
            f"| Conf: {verdict.confidence:.1%}"
        )
        telemetry_line2 = (
            f"Damaged Area: {verdict.physical_defect_area_mm2:.2f} mm2 "
            f"| Anomaly: {verdict.anomaly_score:.2f} "
            f"| Latency: {latency_ms:.1f} ms"
        )

        cv2.putText(
            hud_canvas,
            telemetry_line1,
            (152, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            hud_canvas,
            telemetry_line2,
            (152, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 190, 205),
            1,
            cv2.LINE_AA
        )

        # Divider line
        cv2.line(hud_canvas, (0, banner_height - 1), (img_w, banner_height - 1), (50, 55, 65), 1)

        return hud_canvas

    @staticmethod
    def create_quad_panel_dashboard(
        image_rgb: np.ndarray,
        verdict: InspectionVerdict,
        anomaly_heatmap: Optional[np.ndarray] = None,
        boxes: Optional[List[DetectionBox]] = None,
        segmentation_mask: Optional[np.ndarray] = None,
        latency_ms: float = 0.0,
        target_size: Tuple[int, int] = (300, 300)
    ) -> np.ndarray:
        """
        Creates a 4-pane comparative industrial quality dashboard:
        - Pane 1: Raw Product Feed
        - Pane 2: Tier 1 Anomaly Localization Heatmap
        - Pane 3: Tier 3 Defect Segmentation & Bounding Boxes
        - Pane 4: Final Unified Triage Verdict HUD

        Returns:
            np.ndarray: Combined (2 * H, 2 * W, 3) dashboard array.
        """
        tw, th = target_size
        p1 = cv2.resize(image_rgb, (tw, th))

        # Pane 2: Heatmap
        if anomaly_heatmap is not None:
            h_resized = cv2.resize(anomaly_heatmap, (tw, th))
            h_min, h_max = float(np.min(h_resized)), float(np.max(h_resized))
            if h_max > h_min:
                h_norm = ((h_resized - h_min) / (h_max - h_min) * 255.0).astype(np.uint8)
            else:
                h_norm = np.zeros((th, tw), dtype=np.uint8)
            p2 = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
            p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2RGB)
        else:
            p2 = p1.copy()

        # Pane 3: Spatial Detection & Segmentation
        p3 = p1.copy()
        if segmentation_mask is not None:
            m_res = cv2.resize((segmentation_mask > 0).astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
            p3[m_res > 0] = [255, 0, 128]
        if boxes:
            scale_x = tw / float(image_rgb.shape[1])
            scale_y = th / float(image_rgb.shape[0])
            for b in boxes:
                x1 = int(round(b.box_xyxy[0] * scale_x))
                y1 = int(round(b.box_xyxy[1] * scale_y))
                x2 = int(round(b.box_xyxy[2] * scale_x))
                y2 = int(round(b.box_xyxy[3] * scale_y))
                cv2.rectangle(p3, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Pane 4: Annotated HUD
        p4 = InspectionVisualizer.render_hud_overlay(
            image_rgb=image_rgb,
            verdict=verdict,
            anomaly_heatmap=anomaly_heatmap,
            boxes=boxes,
            segmentation_mask=segmentation_mask,
            latency_ms=latency_ms
        )
        p4_resized = cv2.resize(p4, (tw, th))

        # Add Titles
        def add_title(img: np.ndarray, text: str) -> np.ndarray:
            res = img.copy()
            cv2.rectangle(res, (0, 0), (tw, 24), (20, 20, 20), -1)
            cv2.putText(res, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            return res

        pane1 = add_title(p1, "1. Raw Product Surface")
        pane2 = add_title(p2, f"2. Tier 1 Heatmap (Score: {verdict.anomaly_score:.2f})")
        pane3 = add_title(p3, f"3. Tier 3 Localization ({verdict.physical_defect_area_mm2:.2f} mm2)")
        pane4 = add_title(p4_resized, f"4. Unified Triage: {verdict.decision.value}")

        top_row = np.hstack([pane1, pane2])
        bottom_row = np.hstack([pane3, pane4])
        dashboard = np.vstack([top_row, bottom_row])

        return dashboard
