"""
Unit tests for U-Net architecture, segmentation metrics, loss functions, and inference engine.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.segmentation.unet import UNet, DoubleConv
from src.evaluation.segmentation_metrics import (
    compute_iou,
    compute_dice_coefficient,
    compute_pixel_accuracy,
    compute_segmentation_metrics,
    SoftDiceLoss,
    CombinedBCEDiceLoss,
)
from src.inference.segmenter_inference import (
    UNetSegmenter,
    SegmentationResponse,
    DefectComponent,
)
from src.data.segmentation_dataset import IndustrialSegmentationDataset


# =====================================================================
# 1. U-Net Architecture Tests
# =====================================================================

def test_unet_output_shape():
    model = UNet(in_channels=3, num_classes=1, base_channels=16, bilinear=True)
    x = torch.randn(2, 3, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 1, 128, 128)


def test_unet_odd_input_dimension_padding():
    # Test handling of odd image dimensions (e.g. 127x127)
    model = UNet(in_channels=3, num_classes=1, base_channels=16, bilinear=True)
    x = torch.randn(1, 3, 127, 127)
    logits = model(x)
    assert logits.shape == (1, 1, 127, 127)


def test_unet_backward_gradient_flow():
    model = UNet(in_channels=3, num_classes=1, base_channels=16, bilinear=True)
    x = torch.randn(2, 3, 64, 64)
    target = torch.ones(2, 1, 64, 64)

    criterion = nn.BCEWithLogitsLoss()
    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()

    # Verify gradients exist and are not NaN
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"


def test_unet_predict_mask_binary_values():
    model = UNet(in_channels=3, num_classes=1, base_channels=16, bilinear=True)
    x = torch.randn(1, 3, 64, 64)
    mask = model.predict_mask(x, threshold=0.5)

    assert mask.shape == (1, 1, 64, 64)
    unique_vals = torch.unique(mask).tolist()
    for val in unique_vals:
        assert val in [0.0, 1.0]


# =====================================================================
# 2. Segmentation Metrics Tests
# =====================================================================

def test_iou_and_dice_perfect_overlap():
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    iou = compute_iou(mask, mask)
    dice = compute_dice_coefficient(mask, mask)
    acc = compute_pixel_accuracy(mask, mask)

    assert pytest.approx(iou, rel=1e-5) == 1.0
    assert pytest.approx(dice, rel=1e-5) == 1.0
    assert pytest.approx(acc, rel=1e-5) == 1.0


def test_iou_and_dice_disjoint():
    pred = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [0.0, 1.0]])

    iou = compute_iou(pred, target)
    dice = compute_dice_coefficient(pred, target)

    assert pytest.approx(iou, abs=1e-4) == 0.0
    assert pytest.approx(dice, abs=1e-4) == 0.0


def test_iou_and_dice_both_empty():
    pred = torch.zeros(4, 4)
    target = torch.zeros(4, 4)

    iou = compute_iou(pred, target)
    dice = compute_dice_coefficient(pred, target)

    assert iou == 1.0
    assert dice == 1.0


# =====================================================================
# 3. Loss Functions Tests
# =====================================================================

def test_soft_dice_loss_backward():
    dice_loss = SoftDiceLoss()
    probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = dice_loss(probs.unsqueeze(0).unsqueeze(0), targets.unsqueeze(0).unsqueeze(0))
    loss.backward()

    assert probs.grad is not None
    assert not torch.isnan(probs.grad).any()


def test_combined_bce_dice_loss():
    combo_loss = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.zeros(2, 1, 32, 32)
    targets[:, :, 10:20, 10:20] = 1.0

    loss = combo_loss(logits, targets)
    loss.backward()

    assert loss.item() > 0
    assert logits.grad is not None


# =====================================================================
# 4. Inference Engine Tests
# =====================================================================

def test_unet_segmenter_dummy_inference():
    segmenter = UNetSegmenter(model_path=None, device="cpu", base_channels=16)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    response, bin_mask, prob_map = segmenter.predict(dummy_img)

    assert isinstance(response, SegmentationResponse)
    assert response.image_width == 100
    assert response.image_height == 100
    assert bin_mask.shape == (100, 100)
    assert prob_map.shape == (100, 100)
    assert isinstance(response.is_defective, bool)
    assert response.inference_time_ms > 0


def test_unet_segmenter_draw_overlay():
    segmenter = UNetSegmenter(model_path=None, device="cpu", base_channels=16)
    dummy_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    dummy_mask = np.zeros((50, 50), dtype=np.uint8)
    dummy_mask[10:30, 10:30] = 255  # Defect blob

    annotated = segmenter.draw_segmentation_overlay(dummy_img, dummy_mask)
    assert annotated.shape == dummy_img.shape
    assert annotated.dtype == dummy_img.dtype
    # Pixels inside the defect region should have changed color
    assert not np.array_equal(annotated[15, 15], dummy_img[15, 15])

