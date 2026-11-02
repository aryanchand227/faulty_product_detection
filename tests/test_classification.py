"""
Unit Tests for Classification Architectures, Metrics, and Training Loop.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.models.classification.simple_cnn import SimpleCNN
from src.models.classification.resnet_classifier import ResNetClassifier
from src.models.classification.efficientnet_classifier import EfficientNetClassifier
from src.evaluation.classification_metrics import compute_classification_metrics
from src.training.train_classifier import ClassificationTrainer


class DummyClassificationDataset(Dataset):
    """Synthetic dataset for fast training sanity check."""
    def __init__(self, size: int = 16, num_classes: int = 4):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 3-channel 64x64 synthetic image
        img = torch.randn(3, 64, 64)
        label = idx % self.num_classes
        return {"image": img, "label": label}


def test_simple_cnn_forward_and_backward():
    """Verify SimpleCNN forward pass shape and backprop gradient flow."""
    model = SimpleCNN(num_classes=4, in_channels=3, dropout=0.2)
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    assert out.shape == (4, 4), f"Expected shape (4, 4), got {out.shape}"
    
    loss = out.sum()
    loss.backward()
    
    # Check that first conv layer received non-zero gradients
    first_conv_grad = model.features[0].block[0].weight.grad
    assert first_conv_grad is not None
    assert torch.any(first_conv_grad != 0)


def test_resnet_freeze_and_unfreeze():
    """Verify ResNet transfer-learning layer freezing controls."""
    model = ResNetClassifier(num_classes=6, pretrained=False)
    
    # Freeze
    model.freeze_backbone()
    frozen_params = model.get_num_parameters()
    assert frozen_params["frozen_parameters"] > 0
    assert frozen_params["trainable_parameters"] > 0
    # Only the fc head should require grad
    for name, param in model.backbone.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False
            
    # Unfreeze
    model.unfreeze_backbone()
    unfrozen_params = model.get_num_parameters()
    assert unfrozen_params["frozen_parameters"] == 0
    assert unfrozen_params["trainable_parameters"] == unfrozen_params["total_parameters"]


def test_efficientnet_forward():
    """Verify EfficientNet-B0 forward output shape."""
    model = EfficientNetClassifier(num_classes=6, pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 6), f"Expected shape (2, 6), got {out.shape}"


def test_classification_metrics_accuracy_and_f1():
    """Verify metric calculations against ground truth."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 0, 2, 2]  # 5 out of 6 correct, 1 error
    
    metrics = compute_classification_metrics(y_true, y_pred, class_names=["A", "B", "C"])
    assert metrics["accuracy"] == pytest.approx(5.0 / 6.0, rel=1e-3)
    assert metrics["f1_macro"] > 0.7
    assert metrics["confusion_matrix"].shape == (3, 3)
    assert metrics["per_class_summary"].shape[0] == 3


def test_trainer_smoke_run(tmp_path):
    """Verify trainer runs 1 train epoch and 1 validation step without error."""
    dataset = DummyClassificationDataset(size=16, num_classes=4)
    loader = DataLoader(dataset, batch_size=8)
    
    model = SimpleCNN(num_classes=4, in_channels=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    
    trainer = ClassificationTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=False,
        checkpoint_dir=tmp_path,
        model_name="test_smoke_model"
    )
    
    history = trainer.fit(num_epochs=1, early_stopping_patience=1)
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert (tmp_path / "best_test_smoke_model.pt").exists()

