"""
Classification Evaluation Metrics Module.

Computes:
- Confusion Matrix
- Per-class and macro/weighted Precision, Recall, F1-Score
- Overall Accuracy
- Multi-class ROC-AUC (One-vs-Rest)
- Generates formatted summary DataFrames and Matplotlib confusion matrix heatmaps
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
    """
    Computes a complete suite of classification evaluation metrics.

    Args:
        y_true: Ground-truth class labels (integers 0 to C-1).
        y_pred: Predicted class labels (integers 0 to C-1).
        y_prob: Optional predicted class probability matrix of shape (N, C).
        class_names: Optional human-readable class names.

    Returns:
        Dict containing:
            - 'accuracy': float
            - 'precision_macro': float
            - 'recall_macro': float
            - 'f1_macro': float
            - 'f1_weighted': float
            - 'roc_auc_ovr': Optional float (One-vs-Rest ROC-AUC)
            - 'confusion_matrix': np.ndarray of shape (C, C)
            - 'per_class_summary': pd.DataFrame
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class scores
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class support (sample count)
    support = [np.sum(y_true == i) for i in range(len(class_names))]

    summary_df = pd.DataFrame({
        "Class": class_names,
        "Support": support,
        "Precision": np.round(prec_per_class, 4),
        "Recall": np.round(rec_per_class, 4),
        "F1-Score": np.round(f1_per_class, 4)
    })

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_wt),
        "confusion_matrix": cm,
        "per_class_summary": summary_df
    }

    # Optional Multi-Class ROC-AUC (One-vs-Rest)
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            metrics["roc_auc_ovr"] = float(auc)
        except Exception:
            metrics["roc_auc_ovr"] = None
    else:
        metrics["roc_auc_ovr"] = None

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[str] = None
):
    """
    Renders and optionally saves an annotated Confusion Matrix heatmap using pure Matplotlib.
    """
    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate cells with numeric text
    threshold = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm_display[i, j]:.1%}" if normalize else str(int(cm[i, j]))
            color = "white" if cm_display[i, j] > threshold else "black"
            ax.text(j, i, val, ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()
