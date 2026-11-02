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


def test_voc_to_yolo_conversion_math():
    """Verify known box conversion calculations."""
    img_w, img_h = 200.0, 200.0
    xmin, ymin, xmax, ymax = 50.0, 50.0, 150.0, 150.0
    
    xc, yc, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
    assert xc == 0.5
    assert yc == 0.5
    assert w == 0.5
    assert h == 0.5


def test_yolo_to_voc_invertibility():
    """Verify that converting VOC -> YOLO -> VOC recovers original coordinates."""
    img_w, img_h = 200.0, 200.0
    orig_box = (20, 35, 140, 185)
    
    yolo_box = voc_to_yolo_bbox(*orig_box, img_w, img_h)
    recovered_box = yolo_to_voc_bbox(*yolo_box, img_w, img_h)
    
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
