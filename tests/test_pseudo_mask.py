"""
Unit tests for Weakly-Supervised Pseudo-Mask Generator.
"""

import numpy as np
import pytest

from src.data.pseudo_mask_generator import generate_refined_pseudo_mask


def test_pseudo_mask_solid_box():
    img = np.ones((100, 100), dtype=np.uint8) * 128
    boxes = [(10.0, 10.0, 40.0, 50.0)]

    mask = generate_refined_pseudo_mask(img, boxes, use_intensity_refinement=False)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8

    # Area outside box must be 0
    assert np.all(mask[0:10, :] == 0)
    assert np.all(mask[:, 0:10] == 0)

    # Area inside box must be 255
    assert np.all(mask[10:50, 10:40] == 255)


def test_pseudo_mask_empty_boxes():
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = generate_refined_pseudo_mask(img, [])
    assert mask.shape == (50, 50)
    assert np.count_nonzero(mask) == 0


def test_pseudo_mask_intensity_refinement():
    img = np.ones((100, 100), dtype=np.uint8) * 200
    # Simulate dark defect scratch inside box [20, 20] to [80, 80]
    img[40:60, 40:60] = 30
    boxes = [(20.0, 20.0, 80.0, 80.0)]

    mask = generate_refined_pseudo_mask(img, boxes, use_intensity_refinement=True)
    assert mask.shape == (100, 100)
    # The dark defect area should have positive mask
    assert np.any(mask[40:60, 40:60] == 255)

