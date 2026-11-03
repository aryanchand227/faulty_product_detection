"""
Unit and Integration Tests for Phase 7: PatchCore Anomaly Detection.

Verifies:
1. Backbone intermediate feature extraction (Layers 2 & 3).
2. Neighborhood feature aggregation and pooling.
3. Greedy k-Center coreset subsampling math and coverage.
4. Anomaly distance computation and Gaussian heatmap smoothing.
5. Image-level AUROC, Pixel-level AUROC, and Per-Region Overlap (AUPRO).
6. Triage decision logic (PASS / REVIEW / FAIL) and camera calibration.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.anomaly_detection.patchcore import (
    PatchCore,
    PatchCoreBackbone,
    GreedyCoresetSampler,
)
from src.evaluation.anomaly_metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_aupro,
    find_optimal_threshold,
    evaluate_anomaly_detection,
)
from src.inference.anomaly_inference import PatchCoreInferencer, AnomalyInspectionResponse


def test_patchcore_backbone_feature_extraction():
    """Verify backbone extracts Layer 2 and Layer 3 features with frozen weights."""
    backbone = PatchCoreBackbone(backbone_name="resnet18", pretrained=False, device="cpu")
    dummy_input = torch.randn(2, 3, 224, 224)

    # Check parameter freezing
    for param in backbone.parameters():
        assert not param.requires_grad

    feat2, feat3 = backbone(dummy_input)

    # ResNet-18: Layer 2 -> 128 channels, 28x28; Layer 3 -> 256 channels, 14x14
    assert feat2.shape == (2, 128, 28, 28)
    assert feat3.shape == (2, 256, 14, 14)


def test_patchcore_patch_feature_aggregation():
    """Verify Layer 3 bilinear alignment and 3x3 neighborhood pooling."""
    model = PatchCore(backbone_name="resnet18", patch_size=3, device="cpu")
    dummy_input = torch.randn(2, 3, 224, 224)

    patches, spatial_hw = model.extract_patch_features(dummy_input)

    assert spatial_hw == (28, 28)
    # Total patches per image = 28 * 28 = 784
    # Feature dimension = 128 + 256 = 384
    assert patches.shape == (2, 784, 384)


def test_greedy_coreset_sampler():
    """Verify greedy k-center coreset subsampling reduces size while keeping unique centers."""
    sampler = GreedyCoresetSampler(coreset_ratio=0.20, projection_dim=16, seed=42)

    # 100 vectors of dimension 64
    features = torch.randn(100, 64)
    coreset, indices = sampler.sample(features)

    assert coreset.shape == (20, 64)
    assert len(indices) == 20
    assert len(set(indices.tolist())) == 20
    assert torch.all(indices >= 0) and torch.all(indices < 100)


def test_patchcore_fit_and_predict_pipeline():
    """Verify complete fit and nearest-neighbor distance prediction flow."""
    model = PatchCore(backbone_name="resnet18", coreset_ratio=0.50, device="cpu")

    # Synthetic dataset of 3 normal images
    dummy_images = torch.randn(3, 3, 224, 224)
    dataset = TensorDataset(dummy_images)
    loader = DataLoader(dataset, batch_size=2)

    prototypes = model.fit(loader, verbose=False)
    assert model.memory_bank is not None
    assert prototypes > 0
    assert model.memory_bank.shape[1] == 384

    # Test prediction
    test_batch = torch.randn(2, 3, 224, 224)
    patch_dists, img_scores, spatial_hw = model.predict_patches(test_batch)

    assert patch_dists.shape == (2, 784)
    assert img_scores.shape == (2,)
    assert not torch.isnan(img_scores).any()

    # Test Gaussian-smoothed heatmap generation
    heatmaps = model.compute_anomaly_maps(
        patch_dists, spatial_hw, output_hw=(100, 100), sigma=2.0
    )
    assert heatmaps.shape == (2, 100, 100)
    assert heatmaps.dtype == np.float32
    assert np.all(heatmaps >= 0.0)


def test_image_and_pixel_auroc_metrics():
    """Verify AUROC correctly assigns 1.0 to perfectly separated distributions."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    auroc = compute_image_auroc(y_true, y_scores)
    assert auroc == 1.0

    # Inverted scores should yield 0.0
    auroc_inverted = compute_image_auroc(y_true, [0.9, 0.8, 0.2, 0.1])
    assert auroc_inverted == 0.0

    # Pixel AUROC test
    gt_mask = np.zeros((10, 10), dtype=np.uint8)
    gt_mask[3:7, 3:7] = 1

    a_map = np.zeros((10, 10), dtype=np.float32)
    a_map[3:7, 3:7] = 5.0  # Perfect prediction

    pix_auroc = compute_pixel_auroc([gt_mask], [a_map])
    assert pix_auroc == 1.0


def test_compute_aupro_synthetic():
    """Verify Per-Region Overlap (AUPRO) calculates bounded component recall."""
    gt_normal = np.zeros((20, 20), dtype=np.uint8)
    gt_defect = np.zeros((20, 20), dtype=np.uint8)
    gt_defect[5:10, 5:10] = 1  # 5x5 component

    # Score maps: clean image gets 0, defect image gets high score at defect
    score_normal = np.zeros((20, 20), dtype=np.float32)
    score_defect = np.zeros((20, 20), dtype=np.float32)
    score_defect[5:10, 5:10] = 10.0

    aupro = compute_aupro([gt_normal, gt_defect], [score_normal, score_defect], max_fpr=0.30)
    assert 0.0 <= aupro <= 1.0
    assert aupro > 0.90


def test_find_optimal_threshold_separable():
    """Verify F1-optimal threshold selection on separable scores."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]

    best_thresh, f1, prec, rec, cm = find_optimal_threshold(y_true, y_scores)
    assert f1 == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert cm == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_anomaly_inferencer_triage_decision_and_overlay():
    """Verify PatchCoreInferencer produces valid triage decisions and visual overlays."""
    model = PatchCore(backbone_name="resnet18", device="cpu")
    # Populate dummy memory bank with 5 random prototypes
    model.memory_bank = torch.randn(5, 384)
    model.threshold = 1.0

    inferencer = PatchCoreInferencer(patchcore_instance=model, device="cpu", default_threshold=1.0)
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 128

    response, bin_mask, a_map = inferencer.predict(dummy_img, threshold=1.0, mm_per_pixel=0.1)

    assert isinstance(response, AnomalyInspectionResponse)
    assert response.decision in ("PASS", "REVIEW", "FAIL")
    assert bin_mask.shape == (100, 100)
    assert a_map.shape == (100, 100)
    assert response.inference_time_ms > 0

    if response.defect_pixel_count > 0:
        expected_phys = round(response.defect_pixel_count * (0.1 ** 2), 2)
        assert response.physical_defect_area_mm2 == expected_phys

    # Visual overlay test
    overlay = inferencer.draw_anomaly_overlay(dummy_img, a_map, threshold=1.0)
    assert overlay.shape == dummy_img.shape
    assert overlay.dtype == np.uint8
