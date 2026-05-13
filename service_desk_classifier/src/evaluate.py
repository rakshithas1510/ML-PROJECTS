"""
src/evaluate.py
---------------
Model evaluation utilities:
  - Accuracy
  - Weighted & Macro F1-score
  - Confusion matrix (saved as PNG)
  - Full classification report
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


# ─── Inference Helper ─────────────────────────────────────────────────────────

def get_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference over *loader* and collect all predictions + ground-truth labels.

    Returns
    -------
    (y_true, y_pred) both as 1-D numpy int arrays
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device, non_blocking=True)
            logits    = model(sequences)
            preds     = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    class_names:  List[str],
) -> dict:
    """Compute and return a metrics dict."""
    acc        = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    cm         = confusion_matrix(y_true, y_pred)
    report     = classification_report(
        y_true, y_pred,
        target_names = class_names,
        zero_division = 0,
    )

    metrics = {
        "accuracy":    acc,
        "weighted_f1": weighted_f1,
        "macro_f1":    macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics to stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Weighted F1  : {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1     : {metrics['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])


# ─── Confusion Matrix Plot ────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: List[str],
    save_path:   str,
    title:       str = "Confusion Matrix",
) -> None:
    """
    Save a colour-coded confusion matrix as a PNG image.
    Falls back gracefully if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (safe for servers)
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not found – skipping confusion matrix plot.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks     = np.arange(len(class_names)),
        yticks     = np.arange(len(class_names)),
        xticklabels= class_names,
        yticklabels= class_names,
        ylabel     = "True Label",
        xlabel     = "Predicted Label",
        title      = title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", save_path)


# ─── Training Curve Plot ─────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str) -> None:
    """Save loss & accuracy training curves as a PNG image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not found – skipping training curves plot.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   marker="s")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved -> %s", save_path)


# ─── Full Evaluation Pipeline ─────────────────────────────────────────────────

def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    class_names: List[str],
    results_dir: str = "results/",
    split_name:  str = "test",
) -> dict:
    """
    End-to-end evaluation: predict -> compute metrics -> save plots.

    Parameters
    ----------
    model       : trained PyTorch model
    loader      : DataLoader for the evaluation split
    device      : torch device
    class_names : list of string class labels
    results_dir : directory where artefacts are saved
    split_name  : label used in filenames / titles (e.g. "test", "val")
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    y_true, y_pred = get_predictions(model, loader, device)
    metrics        = compute_metrics(y_true, y_pred, class_names)

    print_metrics(metrics)

    # Confusion matrix PNG
    cm_path = str(Path(results_dir) / f"confusion_matrix_{split_name}.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        save_path = cm_path,
        title     = f"Confusion Matrix ({split_name})",
    )

    # Save classification report as text
    report_path = Path(results_dir) / f"classification_report_{split_name}.txt"
    report_path.write_text(metrics["classification_report"])
    logger.info("Classification report saved -> %s", report_path)

    return metrics
