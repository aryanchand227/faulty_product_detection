"""
Annotation Converter Module: Pascal VOC XML to Normalized YOLO Format.

Handles:
- Invertible coordinate transformations: VOC [xmin, ymin, xmax, ymax] <-> YOLO [x_center, y_center, w, h]
- Parsing Pascal VOC XML files into normalized YOLO text files
- Preparing YOLO-standard dataset directory hierarchy (images/ and labels/ per split)
- Generating data.yaml dataset configuration
- Visual overlay verification to confirm bounding box alignment
"""

import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, ANNOTATIONS_DIR, DOCS_DIR
from src.data.neu_dataset import NEU_CLASSES, CLASS_TO_IDX

YOLO_DIR = ANNOTATIONS_DIR / "yolo"


def voc_to_yolo_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_width: float,
    img_height: float
) -> Tuple[float, float, float, float]:
    """
    Converts Pascal VOC absolute pixel coordinates [xmin, ymin, xmax, ymax]
    into normalized YOLO coordinates [x_center, y_center, width, height] in [0.0, 1.0].

    Formulas:
        x_center = (xmin + xmax) / (2 * img_width)
        y_center = (ymin + ymax) / (2 * img_height)
        width    = (xmax - xmin) / img_width
        height   = (ymax - ymin) / img_height

    Raises:
        ValueError: If bounding box coordinates are degenerate or outside image bounds.
    """
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Degenerate bounding box: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")

    # Clamp values to image boundaries
    xmin = max(0.0, float(xmin))
    ymin = max(0.0, float(ymin))
    xmax = min(float(img_width), float(xmax))
    ymax = min(float(img_height), float(ymax))

    box_w = xmax - xmin
    box_h = ymax - ymin

    x_center = (xmin + (box_w / 2.0)) / img_width
    y_center = (ymin + (box_h / 2.0)) / img_height
    norm_w = box_w / img_width
    norm_h = box_h / img_height

    # Validate range
    for val, name in [(x_center, "x_center"), (y_center, "y_center"), (norm_w, "width"), (norm_h, "height")]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Normalized {name} out of bounds [0, 1]: {val}")

    return round(x_center, 6), round(y_center, 6), round(norm_w, 6), round(norm_h, 6)


def yolo_to_voc_bbox(
    x_center: float,
    y_center: float,
    norm_w: float,
    norm_h: float,
    img_width: float,
    img_height: float
) -> Tuple[int, int, int, int]:
    """
    Inverts normalized YOLO coordinates back to absolute pixel coordinates [xmin, ymin, xmax, ymax].
    """
    box_w = norm_w * img_width
    box_h = norm_h * img_height

    xmin = int(round((x_center * img_width) - (box_w / 2.0)))
    ymin = int(round((y_center * img_height) - (box_h / 2.0)))
    xmax = int(round(xmin + box_w))
    ymax = int(round(ymin + box_h))

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(int(img_width), xmax)
    ymax = min(int(img_height), ymax)

    return xmin, ymin, xmax, ymax


def convert_single_xml(
    xml_path: Union[str, Path],
    class_mapping: Dict[str, int] = CLASS_TO_IDX
) -> List[Tuple[int, float, float, float, float]]:
    """
    Parses a single VOC XML file and extracts all bounding boxes in YOLO format.

    Returns:
        List of tuples: [(class_idx, x_center, y_center, w, h), ...]
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        img_w = float(size_elem.find("width").text)
        img_h = float(size_elem.find("height").text)
    else:
        img_w, img_h = 200.0, 200.0

    yolo_boxes = []

    for obj in root.findall("object"):
        raw_name = obj.find("name").text.strip().lower()
        normalized_name = raw_name.replace(" ", "_").replace("-", "_")

        matched_idx = None
        for c_name, c_idx in class_mapping.items():
            if c_name.replace("-", "_") == normalized_name:
                matched_idx = c_idx
                break

        if matched_idx is None:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        yolo_box = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        yolo_boxes.append((matched_idx, *yolo_box))

    return yolo_boxes


def build_yolo_dataset_structure(
    raw_neu_dir: Path,
    output_yolo_dir: Path = YOLO_DIR
) -> Path:
    """
    Builds the standard Ultralytics YOLO folder structure:
      yolo/
        images/train/, images/val/, images/test/
        labels/train/, labels/val/, labels/test/
        data.yaml
    """
    imagesets_dir = raw_neu_dir / "ImageSets"
    annotations_dir = raw_neu_dir / "annotations"
    images_dir = raw_neu_dir / "images"

    if not images_dir.exists():
        # Fallback to scanning raw directory
        images_dir = raw_neu_dir

    # Create directory tree
    for split in ("train", "val", "test"):
        (output_yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"[*] Building YOLO dataset structure in: {output_yolo_dir}")

    split_counts = {"train": 0, "val": 0, "test": 0}

    for split in ("train", "val", "test"):
        txt_file = imagesets_dir / f"{split}.txt"
        if not txt_file.exists():
            continue

        sample_ids = [line.strip() for line in txt_file.read_text().splitlines() if line.strip()]

        for sample_id in sample_ids:
            # Find image file (search recursively in images directory)
            possible_imgs = list(images_dir.rglob(f"{sample_id}.jpg")) + list(images_dir.rglob(f"{sample_id}.bmp")) + list(images_dir.rglob(f"{sample_id}.png"))
            if not possible_imgs:
                continue
            src_img = possible_imgs[0]

            # Find xml file
            xml_path = annotations_dir / f"{sample_id}.xml"
            if not xml_path.exists():
                continue

            # Convert XML to YOLO lines using our mathematical converter
            yolo_boxes = convert_single_xml(xml_path)

            # Copy image to YOLO directory
            dest_img = output_yolo_dir / "images" / split / src_img.name
            if not dest_img.exists():
                shutil.copy2(src_img, dest_img)

            # Write label file (.txt)
            dest_txt = output_yolo_dir / "labels" / split / f"{sample_id}.txt"
            with open(dest_txt, "w") as f:
                for box in yolo_boxes:
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

            split_counts[split] += 1

    print(f"[OK] Converted and partitioned samples:")
    print(f"  - Train: {split_counts['train']} images + labels")
    print(f"  - Val:   {split_counts['val']} images + labels")
    print(f"  - Test:  {split_counts['test']} images + labels")

    # Generate data.yaml
    yaml_content = {
        "path": str(output_yolo_dir.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(NEU_CLASSES)}
    }

    yaml_path = output_yolo_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"[OK] Generated YOLO configuration: {yaml_path}")
    return output_yolo_dir


def verify_visual_alignment(
    output_yolo_dir: Path = YOLO_DIR,
    num_samples: int = 3
):
    """
    Draws converted YOLO bounding boxes back onto images and saves verification figures.
    """
    assets_dir = DOCS_DIR / "assets" / "detection"
    assets_dir.mkdir(parents=True, exist_ok=True)

    test_imgs = sorted(list((output_yolo_dir / "images" / "test").glob("*.jpg")))[:num_samples]
    if not test_imgs:
        return

    fig, axes = plt.subplots(1, len(test_imgs), figsize=(5 * len(test_imgs), 5))
    if len(test_imgs) == 1:
        axes = [axes]

    for idx, img_path in enumerate(test_imgs):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        label_path = output_yolo_dir / "labels" / "test" / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    xmin, ymin, xmax, ymax = yolo_to_voc_bbox(xc, yc, bw, bh, w, h)

                    # Draw rectangle and label
                    cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (255, 50, 50), 2)
                    cls_name = NEU_CLASSES[cls_id]
                    cv2.putText(
                        img_rgb, cls_name, (xmin, max(15, ymin - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 50), 1
                    )

        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"Sample: {img_path.name}")
        axes[idx].axis("off")

    plt.suptitle("YOLO Annotation Verification: Ground-Truth Bounding Box Alignment", fontsize=12)
    plt.tight_layout()
    plt.savefig(assets_dir / "yolo_box_alignment_verification.png", dpi=200)
    plt.close()
    print(f"[OK] Visual bounding box alignment saved to: {assets_dir / 'yolo_box_alignment_verification.png'}")


if __name__ == "__main__":
    neu_raw_dir = RAW_DATA_DIR / "neu_surface"
    build_yolo_dataset_structure(neu_raw_dir)
    verify_visual_alignment()
