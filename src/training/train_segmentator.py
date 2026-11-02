"""
U-Net Defect Segmentation Training Harness.

Features:
- Combined BCE + Soft Dice Loss for severe industrial class imbalance
- Mixed Precision (AMP) acceleration on CUDA
- Gradient clipping for stable training dynamics
- Validation tracking for Mean IoU, Dice Coefficient, and Pixel Accuracy
- Best checkpoint persistence based on validation Dice score
- Early stopping patience to avoid overfitting on limited defect sets
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.segmentation_metrics import (
    CombinedBCEDiceLoss,
    compute_iou,
    compute_dice_coefficient,
    compute_pixel_accuracy,
)

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """Production training harness for PyTorch U-Net defect segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
        checkpoint_dir: Union[str, Path] = "models/segmentation",
        model_name: str = "unet_bottle_defect",
        patience: int = 15,
        max_grad_norm: float = 1.0
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else CombinedBCEDiceLoss(1.0, 1.0)
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.patience = patience
        self.max_grad_norm = max_grad_norm

        self.best_checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "val_accuracy": [],
            "lr": []
        }

    def train_epoch(self) -> float:
        """Executes a single training epoch across all minibatches."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        use_cuda = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

        for images, masks, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance across validation or test split."""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_acc = 0.0
        num_batches = len(data_loader)

        for images, masks, _ in data_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            bin_preds = (probs >= 0.5).float()

            total_iou += compute_iou(bin_preds, masks)
            total_dice += compute_dice_coefficient(bin_preds, masks)
            total_acc += compute_pixel_accuracy(bin_preds, masks)

        return {
            "loss": total_loss / max(1, num_batches),
            "iou": total_iou / max(1, num_batches),
            "dice": total_dice / max(1, num_batches),
            "pixel_accuracy": total_acc / max(1, num_batches)
        }

    def train(self, num_epochs: int = 30) -> Dict[str, Any]:
        """Runs the full training loop with early stopping and best checkpoint persistence."""
        logger.info(f"Starting U-Net training for {num_epochs} epochs on device: {self.device}")
        best_val_dice = -1.0
        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_metrics["loss"], 4))
            self.history["val_iou"].append(round(val_metrics["iou"], 4))
            self.history["val_dice"].append(round(val_metrics["dice"], 4))
            self.history["val_accuracy"].append(round(val_metrics["pixel_accuracy"], 4))
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | Val Dice: {val_metrics['dice']:.4f} | "
                f"Val Acc: {val_metrics['pixel_accuracy']:.4f}"
            )

            # Checkpoint on best Dice
            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "base_channels": getattr(self.model, "base_channels", 32)
                    },
                    self.best_checkpoint_path
                )
                logger.info(f"Saved new best model checkpoint to: {self.best_checkpoint_path}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs without improvement.")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f} seconds. Best Val Dice: {best_val_dice:.4f}")

        # Save history telemetry
        history_path = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            "best_val_dice": round(best_val_dice, 4),
            "total_time_seconds": round(total_time, 2),
            "epochs_trained": len(self.history["train_loss"]),
            "best_checkpoint": str(self.best_checkpoint_path),
            "history": self.history
        }
