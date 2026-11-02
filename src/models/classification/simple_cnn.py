"""
Custom Baseline Convolutional Neural Network (SimpleCNN).

Architecture designed for industrial surface defect classification:
- 4 Convolutional feature extraction blocks
- Batch Normalization for internal activation scale stabilization
- ReLU non-linear activations
- Max Pooling (2x2) for spatial downsampling and translation invariance
- Global Average Pooling (GAP) to eliminate parameter-heavy dense layers
- Dropout (0.3) for regularization
- Linear classification head
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard Convolutional Building Block:
    Conv2d(3x3, stride=1, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2, stride=2)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    4-Stage Custom Convolutional Neural Network for Defect Classification.

    Input shape:  (B, 3, H, W) e.g., (B, 3, 224, 224)
    Output shape: (B, num_classes) raw logits

    Tensor Shape Progression (for input 224x224):
      Stage 0 (Input):   (B,   3, 224, 224)
      Stage 1 (Block 1): (B,  32, 112, 112)
      Stage 2 (Block 2): (B,  64,  56,  56)
      Stage 3 (Block 3): (B, 128,  28,  28)
      Stage 4 (Block 4): (B, 256,  14,  14)
      Stage 5 (GAP):     (B, 256,   1,   1) -> Flatten -> (B, 256)
      Stage 6 (Head):    (B, num_classes)
    """

    def __init__(self, num_classes: int = 6, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Feature Extractor Backbone
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # Output: 32 channels
            ConvBlock(32, 64),            # Output: 64 channels
            ConvBlock(64, 128),           # Output: 128 channels
            ConvBlock(128, 256)           # Output: 256 channels
        )

        # Global Average Pooling collapses (B, 256, H', W') -> (B, 256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Unnormalized class logits of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def get_num_parameters(self) -> Dict[str, int]:
        """Calculates total, trainable, and non-trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable
        }

