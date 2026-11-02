"""
ResNet-18 Transfer Learning Classifier.

Utilizes ImageNet-pretrained ResNet-18 backbone with residual skip connections:
    y = F(x, {W_i}) + x
Solves the vanishing gradient problem in deep convolutional networks.
Provides layer-freezing utilities for two-stage fine-tuning.
"""

from typing import Dict, List
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetClassifier(nn.Module):
    """
    ResNet-18 Classifier with configurable head and transfer-learning freeze controls.
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load ResNet-18 with ImageNet weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # ResNet-18 penultimate feature dimension is 512
        in_features = self.backbone.fc.in_features

        # Replace standard ImageNet 1000-class head with our defect classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18."""
        return self.backbone(x)

    def freeze_backbone(self):
        """
        Freezes all convolutional layers in the backbone.
        Only the classification head (self.backbone.fc) remains trainable.
        Stage 1 of transfer learning: learn task-specific classifier weights
        without disrupting pretrained low/mid-level feature extractors.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self, unfreeze_layer4_only: bool = False):
        """
        Unfreezes backbone parameters for fine-tuning.
        
        Args:
            unfreeze_layer4_only: If True, only unfreezes the deepest residual block (layer4)
                                  and the classification head, keeping low-level edge/texture
                                  layers (conv1, layer1, layer2, layer3) frozen.
        """
        if unfreeze_layer4_only:
            # Keep early layers frozen
            for name, param in self.backbone.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # Unfreeze everything
            for param in self.backbone.parameters():
                param.requires_grad = True

    def get_num_parameters(self) -> Dict[str, int]:
        """Calculates total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable
        }

