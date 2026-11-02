"""
U-Net Architecture for Industrial Defect Semantic Segmentation.

Implements the classic Encoder-Decoder U-Net with skip connections:
1. Double convolution blocks (Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU)
2. Encoder downsampling via MaxPool2D
3. Bottleneck latent representation
4. Decoder upsampling (Bilinear interpolation or Transposed Conv2D)
5. Skip connection channel concatenation with adaptive spatial padding
6. 1x1 convolution output projection producing raw class logits
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive [Conv2D -> BatchNorm -> ReLU] operations.
    Maintains spatial dimensions via padding=1 for 3x3 kernels.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling module: MaxPool2D(2) followed by DoubleConv.
    Halves spatial dimensions (H/2, W/2) and doubles channel depth.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling module: Upsampling followed by skip-connection concatenation and DoubleConv.
    
    Supports:
    - Bilinear interpolation (avoids transposed convolution checkerboard artifacts)
    - ConvTranspose2D (learnable upsampling filter)
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Low-resolution feature map from previous decoder stage.
            x2: High-resolution skip-connection feature map from corresponding encoder stage.
        """
        x1 = self.up(x1)

        # Handle potential odd dimension rounding mismatch via reflection/zero padding
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dimension (B, C1 + C2, H, W)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution projecting feature maps to target number of classes/masks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for Pixel-Level Defect Segmentation.

    Args:
        in_channels: Input image channels (3 for RGB, 1 for Grayscale).
        num_classes: Number of segmentation masks (1 for binary defect vs background).
        base_channels: Channel width of the first layer (default 32 for efficient edge deployment,
                       or 64 for standard full capacity).
        bilinear: If True, uses bilinear interpolation upsampling; otherwise ConvTranspose2D.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        bilinear: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        c = base_channels
        factor = 2 if bilinear else 1

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, (c * 16) // factor)

        # Decoder (Expanding Path with Skip Connections)
        self.up1 = Up(c * 16, (c * 8) // factor, bilinear)
        self.up2 = Up(c * 8, (c * 4) // factor, bilinear)
        self.up3 = Up(c * 4, (c * 2) // factor, bilinear)
        self.up4 = Up(c * 2, c, bilinear)

        # Output Head
        self.outc = OutConv(c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized class logits.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with Skip Connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Generates discrete binary segmentation mask from input image tensor.

        Args:
            x: Input tensor (B, C, H, W).
            threshold: Probability threshold for positive defect prediction.

        Returns:
            torch.Tensor: Binary mask tensor of shape (B, 1, H, W) with values {0, 1}.
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()

    def count_parameters(self) -> int:
        """Returns total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
