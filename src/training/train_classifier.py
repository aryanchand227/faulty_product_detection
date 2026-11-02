"""
Modular Training and Validation Pipeline for Classification Models.

Features:
- Automatic Mixed Precision (AMP) support for fast, memory-efficient GPU training
- Gradient clipping for numerical stability
- Learning rate scheduling with CosineAnnealingLR
- Patience-based Early Stopping with best checkpoint serialization
- Multi-metric tracking (Loss, Accuracy, Macro F1, Recall)
"""

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from src.evaluation.classification_metrics import compute_classification_metrics


class ClassificationTrainer:
    """
    Manages model training, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Optional[Path] = None,
        model_name: str = "classifier"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp and (device.type == "cuda")
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1_macro": [],
            "learning_rates": []
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Runs one full training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Scaled backprop
            self.scaler.scale(loss).backward()

            # Gradient clipping to prevent exploding gradients
            if self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Runs validation and computes full evaluation metric dictionary."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_classification_metrics(all_labels, all_preds, y_prob=all_probs)
        return val_loss, metrics

    def fit(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        checkpoint_prefix: str = "best"
    ) -> Dict[str, List[float]]:
        """
        Executes full training loop with early stopping on validation Macro-F1.
        """
        best_val_f1 = -1.0
        patience_counter = 0
        best_checkpoint_path = self.checkpoint_dir / f"{checkpoint_prefix}_{self.model_name}.pt"

        print(f"\n[Trainer] Starting training for {num_epochs} epochs on device: {self.device}")
        print(f"[Trainer] Mixed Precision (AMP): {self.use_amp}")
        print(f"[Trainer] Checkpoint path: {best_checkpoint_path}\n")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_metrics = self.validate()
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_macro"]

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1_macro"].append(val_f1)
            self.history["learning_rates"].append(current_lr)

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} F1-Macro: {val_f1:.4f} | LR: {current_lr:.6f}"
            )

            # Early Stopping Check (maximizing Macro F1)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "model_name": self.model_name
                }, best_checkpoint_path)
                print(f"  --> Saved new best checkpoint (F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[Trainer] Early stopping triggered after {epoch} epochs (Patience: {early_stopping_patience})")
                    break

        print(f"\n[Trainer] Training complete. Best Val F1-Macro: {best_val_f1:.4f}")
        return self.history

