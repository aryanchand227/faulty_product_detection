"""
Unit Tests for Pascal VOC to YOLO Annotation Converter and Dataset Builder.
"""

import pytest
import yaml
from pathlib import Path
from src.config import ANNOTATIONS_DIR
from src.data.annotation_converter import (
    voc_to_yolo_bbox,
    yolo_to_voc_bbox,
    convert_single_xml
)
from src.data.neu_dataset import NEU_CLASSES

YOLO_DIR = ANNOTATIONS_DIR / "yolo"


def test_voc_to_yolo_conversion_one_based_math():
    """Verify 1-based Pascal VOC boundary coordinates conversion."""
    img_w, img_h = 200.0, 200.0
    
    # 1. Full image coverage: pixel 1 to pixel 200
    xmin, ymin, xmax, ymax = 1.0, 1.0, 200.0, 200.0
    xc, yc, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h, is_one_indexed=True)
    assert xc == 0.5
    assert yc == 0.5
    assert w == 1.0
    assert h == 1.0

    # 2. Box starting at left-most pixel (pixel 1) with width 50 pixels
    # Spans pixel 1 to pixel 50 -> continuous [0.0, 50.0]
    xmin, ymin, xmax, ymax = 1.0, 1.0, 50.0, 100.0
    xc, yc, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h, is_one_indexed=True)
    assert pytest.approx(w, rel=1e-5) == 50.0 / 200.0  # 0.25
    assert pytest.approx(xc, rel=1e-5) == 25.0 / 200.0 # 0.125
    assert pytest.approx(h, rel=1e-5) == 100.0 / 200.0 # 0.5
    assert pytest.approx(yc, rel=1e-5) == 50.0 / 200.0  # 0.25


def test_yolo_to_voc_invertibility_one_based():
    """Verify that converting VOC -> YOLO -> VOC recovers original 1-based coordinates."""
    img_w, img_h = 200.0, 200.0
    orig_box = (1, 10, 108, 121)
    
    yolo_box = voc_to_yolo_bbox(*orig_box, img_w, img_h, is_one_indexed=True)
    recovered_box = yolo_to_voc_bbox(*yolo_box, img_w, img_h, is_one_indexed=True)
    
    assert orig_box == recovered_box, f"Expected {orig_box}, got {recovered_box}"


def test_yolo_to_voc_invertibility_zero_based():
    """Verify that converting VOC -> YOLO -> VOC recovers original 0-based coordinates."""
    img_w, img_h = 200.0, 200.0
    orig_box = (20, 35, 140, 185)
    
    yolo_box = voc_to_yolo_bbox(*orig_box, img_w, img_h, is_one_indexed=False)
    recovered_box = yolo_to_voc_bbox(*yolo_box, img_w, img_h, is_one_indexed=False)
    
    assert orig_box == recovered_box, f"Expected {orig_box}, got {recovered_box}"


def test_degenerate_box_rejection():
    """Verify ValueError on inverted or degenerate boxes."""
    with pytest.raises(ValueError, match="Degenerate bounding box"):
        voc_to_yolo_bbox(100, 100, 50, 150, 200, 200)

    with pytest.raises(ValueError, match="Degenerate bounding box"):
        voc_to_yolo_bbox(50, 150, 100, 100, 200, 200)


def test_yolo_data_yaml_structure():
    """Verify data.yaml exists and contains correct splits and 6 class names."""
    yaml_path = YOLO_DIR / "data.yaml"
    assert yaml_path.exists(), f"data.yaml missing at {yaml_path}"

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    assert "train" in config
    assert "val" in config
    assert "test" in config
    assert len(config["names"]) == 6
    for idx, name in enumerate(NEU_CLASSES):
        assert config["names"][idx] == name


def test_yolo_dataset_partition_counts():
    """Verify exact sample counts across images and label text files (1,797 total)."""
    assert len(list((YOLO_DIR / "images" / "train").glob("*.jpg"))) == 1437
    assert len(list((YOLO_DIR / "images" / "val").glob("*.jpg"))) == 180
    assert len(list((YOLO_DIR / "images" / "test").glob("*.jpg"))) == 180

    assert len(list((YOLO_DIR / "labels" / "train").glob("*.txt"))) == 1437
    assert len(list((YOLO_DIR / "labels" / "val").glob("*.txt"))) == 180
    assert len(list((YOLO_DIR / "labels" / "test").glob("*.txt"))) == 180
