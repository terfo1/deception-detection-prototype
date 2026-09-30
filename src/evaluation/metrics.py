from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Compute standard binary or multiclass metrics conservatively."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    unique_labels = np.unique(y_true)
    if unique_labels.size == 2 and y_score is not None:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def per_participant_metrics(y_true: np.ndarray, y_pred: np.ndarray, participant_ids: np.ndarray) -> pd.DataFrame:
    """Compute participant-level accuracy and sample counts."""
    frame = pd.DataFrame({"participant_id": participant_ids, "y_true": y_true, "y_pred": y_pred})
    return frame.groupby("participant_id").apply(
        lambda group: pd.Series({"n_samples": int(len(group)), "accuracy": float((group["y_true"] == group["y_pred"]).mean())})
    ).reset_index()


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Write metrics to disk as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ROC curve coordinates for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr
