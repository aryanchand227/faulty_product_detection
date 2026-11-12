"""
PatchCore Anomaly Detection Architecture.

Based on Roth et al. (CVPR 2022):
"Towards Total Recall in Industrial Anomaly Detection"

Core Components:
1. Pretrained Backbone: Frozen ResNet/WideResNet feature hierarchy (Layers 2 & 3).
2. Neighborhood Feature Aggregation: Bilinearly aligned layer concatenation + 2D average pooling.
3. Greedy k-Center Coreset Subsampling: Preserves minimax boundary coverage while reducing
   memory bank size by 90-99% using random linear projection.
4. Nearest-Neighbor Anomaly Scoring: Computes distance to nearest nominal prototype in memory bank.
5. Continuous Gaussian Smoothing: Produces spatially coherent pixel anomaly heatmaps.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader


class PatchCoreBackbone(nn.Module):
    """
    Frozen feature extractor extracting multi-scale representations from Layers 2 and 3.
    """

    SUPPORTED_BACKBONES = ("resnet18", "resnet50", "wide_resnet50_2")

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.device = torch.device(device)

        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: '{self.backbone_name}'. "
                f"Supported: {self.SUPPORTED_BACKBONES}"
            )

        # Load weights
        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.c2 = 128
            self.c3 = 256
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.c2 = 512
            self.c3 = 1024
        elif self.backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.wide_resnet50_2(weights=weights)
            self.c2 = 512
            self.c3 = 1024

        self.feature_dim = self.c2 + self.c3

        # Deconstruct layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Strictly freeze all parameters
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts intermediate feature activations from Layer 2 and Layer 3.

        Args:
            x: Input batch tensor of shape (B, 3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - feat2: (B, C2, H/8, W/8)
                - feat3: (B, C3, H/16, W/16)
        """
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        return feat2, feat3


class GreedyCoresetSampler:
    """
    Greedy k-Center Coreset Subsampler using Minimax Coverage Optimization.

    Finds a subset M_C subset of M that minimizes the maximum distance from any point
    in M to its nearest center in M_C:
        min_{M_C} max_{z in M} min_{c in M_C} ||z - c||_2

    Applies Johnson-Lindenstrauss random projection to speed up distance calculations
    during subsampling while preserving relative Euclidean geometries.
    """

    def __init__(
        self,
        coreset_ratio: float = 0.10,
        projection_dim: int = 128,
        seed: int = 42
    ) -> None:
        if not (0.0 < coreset_ratio <= 1.0):
            raise ValueError(f"coreset_ratio must be in (0.0, 1.0], got {coreset_ratio}")
        self.coreset_ratio = coreset_ratio
        self.projection_dim = projection_dim
        self.seed = seed

    def sample(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs greedy k-center coreset subsampling.

        Args:
            features: Tensor of shape (M, D) containing all candidate patch vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - coreset_features: Subsampled tensor of shape (K, D).
                - selected_indices: 1D Tensor of shape (K,) containing selected indices.
        """
        M, D = features.shape
        target_k = max(1, int(math.ceil(M * self.coreset_ratio)))

        if target_k >= M:
            indices = torch.arange(M, dtype=torch.long)
            return features, indices

        torch.manual_seed(self.seed)

        # 1. Dimension reduction via Random Linear Projection
        if self.projection_dim > 0 and self.projection_dim < D:
            # Gaussian random projection scaled by 1/sqrt(projection_dim)
            proj_matrix = torch.randn(D, self.projection_dim, device=features.device, dtype=features.dtype)
            proj_matrix /= math.sqrt(self.projection_dim)
            proj_features = torch.matmul(features, proj_matrix)
        else:
            proj_features = features

        # 2. Initialize first center (deterministic or seed-based)
        selected_indices: List[int] = []
        first_idx = int(torch.randint(0, M, (1,)).item())
        selected_indices.append(first_idx)

        # 3. Vectorized Minimax Greedy Selection
        # Track minimum squared distance from each point to the selected centers
        current_center = proj_features[first_idx:first_idx + 1]
        min_sq_dists = torch.sum((proj_features - current_center) ** 2, dim=1)

        for _ in range(1, target_k):
            # Select point furthest from current centers
            furthest_idx = int(torch.argmax(min_sq_dists).item())
            selected_indices.append(furthest_idx)

            # Update distance vector: min(d_old, d_to_new_center)
            new_center = proj_features[furthest_idx:furthest_idx + 1]
            new_sq_dists = torch.sum((proj_features - new_center) ** 2, dim=1)
            min_sq_dists = torch.minimum(min_sq_dists, new_sq_dists)

        selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=features.device)
        coreset_features = features[selected_tensor]
        return coreset_features, selected_tensor


class PatchCore:
    """
    PatchCore Industrial Anomaly Detection and Localization Model.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        patch_size: int = 3,
        coreset_ratio: float = 0.10,
        projection_dim: int = 128,
        device: Union[str, torch.device] = "cpu",
        seed: int = 42
    ) -> None:
        self.device = torch.device(device)
        self.backbone_name = backbone_name
        self.patch_size = patch_size
        self.coreset_ratio = coreset_ratio
        self.projection_dim = projection_dim
        self.seed = seed

        self.backbone = PatchCoreBackbone(
            backbone_name=backbone_name,
            pretrained=True,
            device=self.device
        )
        self.sampler = GreedyCoresetSampler(
            coreset_ratio=coreset_ratio,
            projection_dim=projection_dim,
            seed=seed
        )

        # Memory bank: shape (K, D) where K is subsampled patch count
        self.memory_bank: Optional[torch.Tensor] = None
        self.threshold: float = 0.0

    def extract_patch_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extracts neighborhood-pooled patch feature embeddings.

        Args:
            images: Tensor of shape (B, 3, H, W).

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]:
                - patch_features: Tensor of shape (B, H2*W2, D)
                - spatial_shape: (H2, W2) spatial resolution of Layer 2
        """
        with torch.no_grad():
            feat2, feat3 = self.backbone(images)
            # feat2: (B, C2, H2, W2)
            # feat3: (B, C3, H3, W3)
            H2, W2 = feat2.shape[2], feat2.shape[3]

            # Upsample Layer 3 to match Layer 2 spatial dimensions
            feat3_up = F.interpolate(feat3, size=(H2, W2), mode="bilinear", align_corners=False)

            # Concatenate along channel dimension: (B, C2+C3, H2, W2)
            concat_feats = torch.cat([feat2, feat3_up], dim=1)

            # Locally aware neighborhood pooling: AvgPool2d(kernel_size=3, stride=1, padding=1)
            pad = self.patch_size // 2
            pooled = F.avg_pool2d(
                concat_feats,
                kernel_size=self.patch_size,
                stride=1,
                padding=pad
            )  # (B, D, H2, W2)

            B, D, _, _ = pooled.shape
            # Permute and reshape to (B, H2*W2, D)
            patches = pooled.permute(0, 2, 3, 1).reshape(B, H2 * W2, D)
            return patches, (H2, W2)

    def fit(self, dataloader: DataLoader, verbose: bool = True) -> int:
        """
        Builds the nominal memory bank from defect-free training images.

        Args:
            dataloader: PyTorch DataLoader yielding normal training samples.
            verbose: Whether to print memory bank statistics.

        Returns:
            int: Number of coreset prototypes stored in the memory bank.
        """
        all_features: List[torch.Tensor] = []

        self.backbone.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                elif isinstance(batch, dict):
                    images = batch["image"]
                else:
                    images = batch

                images = images.to(self.device)
                patches, _ = self.extract_patch_features(images)
                # Flatten batch: (B * P, D)
                patches_flat = patches.reshape(-1, patches.shape[-1]).cpu()
                all_features.append(patches_flat)

        if not all_features:
            raise ValueError("Training DataLoader provided zero samples to build memory bank.")

        full_features = torch.cat(all_features, dim=0)  # (M, D)
        total_patches = full_features.shape[0]

        # Subsample using Greedy k-Center Coreset
        coreset, _ = self.sampler.sample(full_features)
        self.memory_bank = coreset.to(self.device)

        if verbose:
            reduction_pct = (1.0 - (coreset.shape[0] / total_patches)) * 100.0
            print(
                f"[PatchCore] Memory bank created: {coreset.shape[0]} prototypes "
                f"(reduced from {total_patches} patches, {reduction_pct:.1f}% compression). "
                f"Feature dim: {coreset.shape[1]}."
            )

        return self.memory_bank.shape[0]

    def predict_patches(
        self,
        images: torch.Tensor,
        chunk_size: int = 64,
        score_method: str = "max",
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Computes nearest-neighbor patch anomaly distances against the memory bank.

        Args:
            images: Tensor of shape (B, 3, H, W).
            chunk_size: Size for chunking distance matrix to prevent OOM.
            score_method: 'max' (maximum patch distance) or 'top_k' (mean of top-k largest patch distances).
            top_k: Number of highest-scoring patches to average when score_method='top_k'.

        Returns:
            Tuple containing:
                - patch_distances: (B, P) distance of each patch to nearest bank center
                - image_scores: (B,) aggregate image anomaly score
                - spatial_shape: (H2, W2) feature map grid dimensions
        """
        if self.memory_bank is None:
            raise RuntimeError("PatchCore memory bank is empty. Call fit() or load() before predict.")

        patches, (H2, W2) = self.extract_patch_features(images)
        B, P, D = patches.shape
        patches_flat = patches.reshape(B * P, D)  # (N_queries, D)

        # Compute nearest-neighbor distances in chunks: min_j ||q_i - m_j||_2
        # ||q - m||^2 = ||q||^2 + ||m||^2 - 2 <q, m>
        N_queries = patches_flat.shape[0]
        K = self.memory_bank.shape[0]

        min_dists = torch.empty(N_queries, device=self.device, dtype=torch.float32)

        mem_norm_sq = torch.sum(self.memory_bank ** 2, dim=1, keepdim=True).t()  # (1, K)

        for start in range(0, N_queries, chunk_size):
            end = min(start + chunk_size, N_queries)
            q_chunk = patches_flat[start:end]  # (chunk, D)
            q_norm_sq = torch.sum(q_chunk ** 2, dim=1, keepdim=True)  # (chunk, 1)

            # Dot products: (chunk, K)
            dot = torch.matmul(q_chunk, self.memory_bank.t())

            # Squared Euclidean distances: ||q||^2 + ||m||^2 - 2 * dot
            sq_dists = q_norm_sq + mem_norm_sq - 2.0 * dot
            sq_dists = torch.clamp(sq_dists, min=0.0)

            chunk_min_sq, _ = torch.min(sq_dists, dim=1)
            min_dists[start:end] = torch.sqrt(chunk_min_sq)

        patch_distances = min_dists.reshape(B, P)

        # Aggregate Image-level anomaly score
        if score_method.lower() == "top_k":
            k_val = min(max(1, top_k), P)
            topk_vals, _ = torch.topk(patch_distances, k=k_val, dim=1, largest=True)
            image_scores = torch.mean(topk_vals, dim=1)
        else:
            image_scores, _ = torch.max(patch_distances, dim=1)

        return patch_distances, image_scores, (H2, W2)

    def compute_anomaly_maps(
        self,
        patch_distances: torch.Tensor,
        spatial_shape: Tuple[int, int],
        output_hw: Tuple[int, int],
        sigma: float = 4.0
    ) -> np.ndarray:
        """
        Converts patch distances into continuous, Gaussian-smoothed pixel anomaly maps.

        Args:
            patch_distances: Tensor of shape (B, P) containing patch anomaly distances.
            spatial_shape: (H2, W2) patch grid dimensions.
            output_hw: (target_height, target_width) for output maps.
            sigma: Standard deviation for Gaussian smoothing filter (if <= 0.0, raw bilinear upsampling).

        Returns:
            np.ndarray: Anomaly heatmaps of shape (B, H, W) in float32.
        """
        B, P = patch_distances.shape
        H2, W2 = spatial_shape
        out_h, out_w = output_hw

        # Reshape to 2D grid: (B, 1, H2, W2)
        grid = patch_distances.reshape(B, 1, H2, W2)

        # Bilinear interpolation to target resolution
        upsampled = F.interpolate(grid, size=(out_h, out_w), mode="bilinear", align_corners=False)
        upsampled_np = upsampled.squeeze(1).cpu().numpy()  # (B, H, W)

        if sigma <= 0.0:
            return upsampled_np

        # Apply 2D Gaussian filter to each sample in batch
        smoothed_maps = np.empty_like(upsampled_np)
        for i in range(B):
            smoothed_maps[i] = ndimage.gaussian_filter(upsampled_np[i], sigma=sigma)

        return smoothed_maps

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Saves PatchCore state (memory bank, backbone config, threshold) to disk.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "backbone_name": self.backbone_name,
            "patch_size": self.patch_size,
            "coreset_ratio": self.coreset_ratio,
            "projection_dim": self.projection_dim,
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "threshold": self.threshold,
            "seed": self.seed
        }
        torch.save(state, filepath)
        return filepath

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Loads PatchCore state from disk.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PatchCore checkpoint not found at: {filepath}")

        state = torch.load(filepath, map_location=self.device)
        self.backbone_name = state.get("backbone_name", self.backbone_name)
        self.patch_size = state.get("patch_size", self.patch_size)
        self.coreset_ratio = state.get("coreset_ratio", self.coreset_ratio)
        self.projection_dim = state.get("projection_dim", self.projection_dim)
        self.threshold = state.get("threshold", 0.0)

        mb = state.get("memory_bank")
        if mb is not None:
            self.memory_bank = mb.to(self.device)

