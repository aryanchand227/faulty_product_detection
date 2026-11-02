"""
YOLO Object Detection Training Module for Industrial Defect Detection.

Automates training of Ultralytics YOLO models (e.g., YOLOv8n, YOLOv8s) on the
NEU surface defect dataset converted to YOLO format.

Supports:
- Anchor-free TaskAlignedAssigner training
- Dual CIoU + DFL bounding box regression losses
- Multi-scale training and data augmentation (HSV jitter, mosaic, mixup, fliplr)
- Automatic evaluation on validation and test partitions
- Exporting best model checkpoints and training telemetry
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_yolo_detector(
    data_yaml: Union[str, Path] = "datasets/annotations/yolo/data.yaml",
    model_variant: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "cpu",
    project: Union[str, Path] = "experiments/detection/yolo",
    name: str = "neu_yolov8n",
    patience: int = 15,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    seed: int = 42,
    run_test_eval: bool = True,
    exist_ok: bool = True
) -> Dict[str, Any]:
    """
    Executes YOLO training pipeline on the prepared dataset.

    Args:
        data_yaml: Path to data.yaml dataset descriptor.
        model_variant: Pretrained model weights (e.g. 'yolov8n.pt', 'yolov8s.pt').
        epochs: Number of training epochs.
        imgsz: Target square image dimension for network input (multiples of 32).
        batch_size: Mini-batch size.
        device: Execution hardware target ('cpu', '0', 'cuda').
        project: Directory where training runs and artifacts are saved.
        name: Subdirectory name for this specific training run.
        patience: Epochs to wait without validation improvement before early stopping.
        optimizer: Optimization algorithm ('AdamW', 'SGD', 'auto').
        lr0: Initial learning rate.
        seed: Random seed for deterministic weight initialization and shuffling.
        run_test_eval: Whether to run an explicit evaluation pass on the test split.
        exist_ok: Whether to allow writing into existing run directory without incrementing.

    Returns:
        Dict[str, Any]: Summary dictionary containing checkpoint paths, training duration,
                        and validation/test metrics.
    """
    data_yaml_path = Path(data_yaml).resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YOLO data configuration file not found at: {data_yaml_path}")

    project_path = Path(project).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INDUSTRIAL PRODUCT DEFECT DETECTOR: YOLO TRAINING")
    logger.info(f"Model Architecture : {model_variant}")
    logger.info(f"Dataset Config     : {data_yaml_path}")
    logger.info(f"Input Resolution   : {imgsz}x{imgsz}")
    logger.info(f"Epochs / Batch Size: {epochs} / {batch_size}")
    logger.info(f"Optimizer / LR     : {optimizer} / {lr0}")
    logger.info(f"Device Target      : {device}")
    logger.info(f"Output Project     : {project_path / name}")
    logger.info("=" * 60)

    # Load base model (downloads pretrained weights if not cached locally)
    model = YOLO(model_variant)

    start_time = time.time()

    # Launch Ultralytics training engine
    train_results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(project_path),
        name=name,
        patience=patience,
        optimizer=optimizer,
        lr0=lr0,
        seed=seed,
        exist_ok=exist_ok,
        verbose=True,
        save=True,
        plots=True
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time / 60:.2f} minutes.")

    # Locate generated best and last checkpoint weights
    run_dir = project_path / name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    summary: Dict[str, Any] = {
        "model_variant": model_variant,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": device,
        "training_time_seconds": round(elapsed_time, 2),
        "run_dir": str(run_dir),
        "best_weights": str(best_weights) if best_weights.exists() else None,
        "last_weights": str(last_weights) if last_weights.exists() else None,
        "val_metrics": {},
        "test_metrics": {}
    }

    # Evaluate best weights on validation split
    if best_weights.exists():
        logger.info("Evaluating best model checkpoint on validation set...")
        best_model = YOLO(str(best_weights))
        val_results = best_model.val(data=str(data_yaml_path), split="val", imgsz=imgsz, device=device)
        summary["val_metrics"] = {
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr)
        }
        logger.info(
            f"Validation Results: mAP@50={val_results.box.map50:.4f}, "
            f"mAP@50:95={val_results.box.map:.4f}, "
            f"Precision={val_results.box.mp:.4f}, Recall={val_results.box.mr:.4f}"
        )

        # Evaluate on test split if requested
        if run_test_eval:
            logger.info("Evaluating best model checkpoint on hold-out test set...")
            test_results = best_model.val(data=str(data_yaml_path), split="test", imgsz=imgsz, device=device)
            summary["test_metrics"] = {
                "mAP50": float(test_results.box.map50),
                "mAP50_95": float(test_results.box.map),
                "precision": float(test_results.box.mp),
                "recall": float(test_results.box.mr)
            }
            logger.info(
                f"Test Results: mAP@50={test_results.box.map50:.4f}, "
                f"mAP@50:95={test_results.box.map:.4f}, "
                f"Precision={test_results.box.mp:.4f}, Recall={test_results.box.mr:.4f}"
            )

    # Save summary report to JSON
    summary_path = run_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training telemetry summary to {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector on industrial surface defects.")
    parser.add_argument("--data", type=str, default="datasets/annotations/yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLO model (yolov8n.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu, 0, cuda)")
    parser.add_argument("--project", type=str, default="experiments/detection/yolo", help="Project output folder")
    parser.add_argument("--name", type=str, default="neu_yolov8n", help="Run experiment name")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (AdamW, SGD)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_yolo_detector(
        data_yaml=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr
    )
