"""
utils/metrics.py
================
Metrics helpers used by train.py and evaluate.py
"""

from pathlib import Path
from typing import List

import numpy as np

# ── Optional imports (graceful degradation) ───────────────────────
try:
    from sklearn.metrics import (
        roc_auc_score, f1_score, confusion_matrix,
        precision_score, recall_score,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    print("[WARNING] scikit-learn not installed — AUC/F1 metrics unavailable")

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False
    print("[WARNING] matplotlib not installed — plots disabled")


# ── compute_metrics ───────────────────────────────────────────────
def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
) -> dict:
    """
    Parameters
    ----------
    y_true : ground-truth class indices
    y_pred : predicted class indices
    y_prob : probability of the positive class (malignant)

    Returns
    -------
    dict with keys: auc, f1, precision, recall/sensitivity, specificity,
                    conf_matrix
    """
    m: dict = {}

    if _SKLEARN:
        m["auc"]         = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0
        m["f1"]          = float(f1_score(y_true, y_pred, zero_division=0))
        m["precision"]   = float(precision_score(y_true, y_pred, zero_division=0))
        m["sensitivity"] = float(recall_score(y_true, y_pred, zero_division=0))  # = recall
        cm               = confusion_matrix(y_true, y_pred)
        m["conf_matrix"] = cm
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            m["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            m["specificity"] = 0.0
    else:
        # Minimal fallback
        m["auc"]         = 0.0
        m["f1"]          = 0.0
        m["precision"]   = 0.0
        m["sensitivity"] = 0.0
        m["specificity"] = 0.0
        m["conf_matrix"] = np.zeros((2, 2), dtype=int)

    return m


# ── plot_training_curves ──────────────────────────────────────────
def plot_training_curves(history: dict, save_path):
    """Save a 2×2 grid: loss, accuracy, AUC, F1 for train & val."""
    if not _MPL:
        return

    epochs  = range(1, len(history["train"]) + 1)
    metrics = [
        ("loss",     "Loss"),
        ("accuracy", "Accuracy"),
        ("auc",      "AUC-ROC"),
        ("f1",       "F1 Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        tr = [e.get(key, 0) for e in history["train"]]
        va = [e.get(key, 0) for e in history["val"]]
        ax.plot(epochs, tr, label="Train",      linewidth=2)
        ax.plot(epochs, va, label="Validation", linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Cancer Detection CNN",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves saved → {save_path}")


# ── save_confusion_matrix ─────────────────────────────────────────
def save_confusion_matrix(cm, class_names: List[str], save_path):
    """Save a colour-coded confusion matrix image."""
    if not _MPL:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    ax.set_ylabel("True Label",      fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved → {save_path}")


# ── plot_roc_curve ────────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob, save_path):
    """Save an ROC curve plot."""
    if not (_MPL and _SKLEARN):
        return
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0]);  ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Cancer Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ ROC curve saved → {save_path}")
