"""
Semantic Segmentation Evaluation Metrics and Loss Functions.

Implements from first principles:
1. Pixel Accuracy
2. Intersection over Union (IoU / Jaccard Index)
3. Dice Coefficient (F1-Score)
4. Differentiable Soft Dice Loss module
5. Combined BCE + Dice Loss module for training under extreme class imbalance
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6
) -> float:
    """
    Computes Intersection over Union (IoU) for binary masks:
        IoU = |Pred ∩ Target| / (|Pred ∪ Target| + ε)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum().item()
    union = pred_flat.sum().item() + target_flat.sum().item() - intersection

    if union == 0.0:
        return 1.0  # Both empty -> perfect alignment

    return float((intersection + epsilon) / (union + epsilon))


def compute_dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6
) -> float:
    """
    Computes Dice Coefficient (F1-Score) for binary masks:
        Dice = 2 * |Pred ∩ Target| / (|Pred| + |Target| + ε)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum().item()
    cardinality = pred_flat.sum().item() + target_flat.sum().item()

    if cardinality == 0.0:
        return 1.0  # Both empty -> perfect alignment

    return float((2.0 * intersection + epsilon) / (cardinality + epsilon))


def compute_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Computes percentage of correctly classified pixels:
        Acc = (TP + TN) / Total
    """
    correct = (pred == target).sum().item()
    total = pred.numel()
    return float(correct / total) if total > 0 else 0.0


def compute_segmentation_metrics(
    pred_probs: torch.Tensor,
    target_masks: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculates comprehensive segmentation metrics for a batch of predictions.

    Args:
        pred_probs: Predicted probabilities in [0, 1] of shape (B, 1, H, W) or (B, H, W).
        target_masks: Ground truth masks with values in {0, 1}.
        threshold: Binarization decision threshold.

    Returns:
        Dict[str, float]: iou, dice, pixel_accuracy
    """
    binary_preds = (pred_probs >= threshold).float()
    targets = target_masks.float()

    iou = compute_iou(binary_preds, targets)
    dice = compute_dice_coefficient(binary_preds, targets)
    acc = compute_pixel_accuracy(binary_preds, targets)

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "pixel_accuracy": round(acc, 4)
    }


class SoftDiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss for gradient backpropagation.

    Loss = 1 - (2 * sum(p * g) + ε) / (sum(p^2) + sum(g^2) + ε)
    """

    def __init__(self, epsilon: float = 1e-6, squared_denom: bool = True) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.squared_denom = squared_denom

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_probs: Probabilities from sigmoid activation, shape (B, 1, H, W).
            targets: Binary ground truth {0, 1}, shape (B, 1, H, W).
        """
        pred_flat = pred_probs.contiguous().view(pred_probs.size(0), -1)
        target_flat = targets.contiguous().view(targets.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)

        if self.squared_denom:
            cardinality = (pred_flat ** 2).sum(dim=1) + (target_flat ** 2).sum(dim=1)
        else:
            cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)
        loss = 1.0 - dice
        return loss.mean()


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Soft Dice Loss.

    Loss = BCEWithLogits(logits, targets) + lambda_dice * SoftDiceLoss(sigmoid(logits), targets)

    Benefits for Industrial Inspection:
    - BCE provides smooth, stable gradients across all pixels (background and defect).
    - Dice Loss prevents the network from collapsing into trivial background prediction
      when defects occupy only 1-5% of total pixels.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw network output before sigmoid, shape (B, 1, H, W).
            targets: Ground-truth binary mask, shape (B, 1, H, W).
        """
        bce = self.bce_loss(logits, targets)
        probs = torch.sigmoid(logits)
        dice = self.dice_loss(probs, targets)

        return (self.bce_weight * bce) + (self.dice_weight * dice)
