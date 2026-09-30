from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def save_confusion_matrix_figure(matrix: np.ndarray, path: str | Path) -> None:
    """Save a confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is not installed; skipping confusion matrix figure.")
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, path: str | Path) -> None:
    """Save ROC curve plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is not installed; skipping ROC figure.")
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
