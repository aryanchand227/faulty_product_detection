"""
EfficientNet-B0 Transfer Learning Classifier.

Utilizes ImageNet-pretrained EfficientNet-B0 backbone.
Demonstrates:
- Depthwise Separable Convolutions (reducing FLOPs by 8-9x vs standard convs)
- Compound Scaling (balanced scaling of network depth, width, and resolution)
- High parameter efficiency (~5.3M parameters vs ResNet-18's ~11.7M)
"""

from typing import Dict
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 Classifier for Industrial Product Inspection.
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load EfficientNet-B0 with ImageNet weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # In EfficientNet-B0, penultimate feature dimension is 1280
        # self.backbone.classifier is: Sequential(Dropout(p=0.2), Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features

        # Replace classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EfficientNet-B0."""
        return self.backbone(x)

    def freeze_backbone(self):
        """
        Freezes the feature extraction backbone.
        Only the classification head remains trainable.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_num_parameters(self) -> Dict[str, int]:
        """Calculates parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable
        }

