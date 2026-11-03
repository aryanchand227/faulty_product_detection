"""
Weakly Supervised Pseudo-Mask Generator for NEU Surface Defects.

Addresses the dataset limitation where NEU-DET provides only bounding box XMLs,
not human-annotated pixel-level segmentation masks.

Methods:
1. Rectangular Box Mask: Fills bounding boxes with solid positive label (1).
2. Refined Intensity Pseudo-Mask: Restricts positive labels inside bounding boxes
   to anomalous pixels detected via local Otsu / adaptive thresholding against
   the local steel background.

IMPORTANT ACADEMIC & ENGINEERING NOTICE:
----------------------------------------
These generated masks are WEAK PSEUDO-LABELS. They allow exploring weakly-supervised
segmentation algorithms, but must NOT be represented as human ground truth in university
evaluations or production safety audits. For true pixel-level segmentation:
- Use MVTec AD Bottle (which includes authentic pixel-level ground truth masks)
- Or integrate the Severstal Steel Defect dataset (which provides human RLE pixel masks).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_refined_pseudo_mask(
    image: np.ndarray,
    voc_boxes: List[Tuple[float, float, float, float]],
    use_intensity_refinement: bool = True
) -> np.ndarray:
    """
    Generates a single-channel binary pseudo-mask (uint8 {0, 255}) from bounding boxes.

    Args:
        image: Grayscale or BGR NumPy array of the product image.
        voc_boxes: List of bounding boxes [(xmin, ymin, xmax, ymax), ...] in pixel coordinates.
        use_intensity_refinement: If True, applies local Otsu thresholding within each box
                                  to isolate the defect pixels from background steel.
                                  If False, fills the entire rectangular box.

    Returns:
        np.ndarray: Binary mask of shape (H, W) with values {0, 255}.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in voc_boxes:
        xmin = max(0, min(w, int(round(box[0]))))
        ymin = max(0, min(h, int(round(box[1]))))
        xmax = max(0, min(w, int(round(box[2]))))
        ymax = max(0, min(h, int(round(box[3]))))

        if xmax <= xmin or ymax <= ymin:
            continue

        box_crop = gray[ymin:ymax, xmin:xmax]

        if use_intensity_refinement and box_crop.size > 16:
            # Steel defects are either significantly darker (scratches, inclusions)
            # or lighter (rolled-in scale, patches) than the local mean.
            # Apply Otsu's thresholding to separate foreground defect from background steel.
            blur = cv2.GaussianBlur(box_crop, (3, 3), 0)
            _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # If Otsu selected >80% or <5% of the box, fall back to solid box to avoid degenerate masks
            defect_ratio = np.count_nonzero(otsu_thresh) / float(otsu_thresh.size)
            if 0.05 <= defect_ratio <= 0.85:
                mask[ymin:ymax, xmin:xmax] = np.maximum(mask[ymin:ymax, xmin:xmax], otsu_thresh)
            else:
                mask[ymin:ymax, xmin:xmax] = 255
        else:
            mask[ymin:ymax, xmin:xmax] = 255

    return mask


def batch_generate_neu_pseudo_masks(
    raw_neu_dir: Union[str, Path] = "datasets/raw/neu_surface",
    output_dir: Union[str, Path] = "datasets/processed/neu_pseudo_masks",
    use_intensity_refinement: bool = True
) -> int:
    """
    Generates pseudo-masks for all NEU surface images and saves them to disk.

    Returns:
        int: Number of generated mask files.
    """
    neu_p = Path(raw_neu_dir).resolve()
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    img_dir = neu_p / "images"
    ann_dir = neu_p / "annotations"

    if not ann_dir.exists():
        raise FileNotFoundError(f"NEU annotations not found at: {ann_dir}")

    # Index images
    img_map = {
        p.stem: p for p in img_dir.rglob("*.*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".bmp", ".png")
    }

    count = 0
    xml_files = sorted(list(ann_dir.glob("*.xml")))

    for xml_p in xml_files:
        stem = xml_p.stem
        if stem not in img_map:
            continue

        img_p = img_map[stem]
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        # Parse VOC boxes
        tree = ET.parse(xml_p)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is not None:
                # Account for 1-based VOC indexing
                xmin = float(bnd.find("xmin").text) - 1.0
                ymin = float(bnd.find("ymin").text) - 1.0
                xmax = float(bnd.find("xmax").text)
                ymax = float(bnd.find("ymax").text)
                boxes.append((xmin, ymin, xmax, ymax))

        mask = generate_refined_pseudo_mask(img, boxes, use_intensity_refinement=use_intensity_refinement)
        cv2.imwrite(str(out_p / f"{stem}_mask.png"), mask)
        count += 1

    logger.info(f"Generated {count} weakly-supervised pseudo-masks in: {out_p}")
    return count
