from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.loaders import load_dataset
from src.data.synthetic import make_synthetic_eye_tracking_dataset
from src.evaluation.metrics import compute_classification_metrics, compute_roc, per_participant_metrics, save_metrics
from src.evaluation.plots import save_confusion_matrix_figure, save_roc_curve
from src.features.aggregated import create_aggregated_features
from src.features.preprocessing import preprocess_eye_tracking
from src.features.windowing import WindowedDataset, create_sliding_windows
from src.models.baselines import build_baseline_model
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _load_or_synthetic(config: dict, allow_synthetic_debug: bool) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    try:
        return load_dataset(
            dataset_name=dataset_cfg["name"],
            raw_dir=dataset_cfg["raw_dir"],
            recursive=bool(dataset_cfg.get("recursive", False)),
            allow_missing_schema=bool(dataset_cfg.get("allow_missing_schema", True)),
        )
    except FileNotFoundError:
        if allow_synthetic_debug:
            LOGGER.warning("Raw dataset missing. Falling back to synthetic debug data.")
            return make_synthetic_eye_tracking_dataset(seed=int(config.get("seed", 42)))
        raise


def split_by_participant(
    participant_ids: np.ndarray,
    train_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train, validation, and test without participant leakage."""
    indices = np.arange(len(participant_ids))
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(splitter.split(indices, groups=participant_ids))

    temp_ratio = val_size / (1.0 - train_size)
    temp_participants = participant_ids[temp_idx]
    second_splitter = GroupShuffleSplit(n_splits=1, train_size=temp_ratio, random_state=seed)
    rel_val_idx, rel_test_idx = next(second_splitter.split(temp_idx, groups=temp_participants))
    val_idx = temp_idx[rel_val_idx]
    test_idx = temp_idx[rel_test_idx]
    return train_idx, val_idx, test_idx


def split_within_subject(
    participant_ids: np.ndarray,
    train_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Secondary benchmark split that samples within each participant."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    all_indices = np.arange(len(participant_ids))
    for participant in np.unique(participant_ids):
        participant_idx = all_indices[participant_ids == participant]
        shuffled = rng.permutation(participant_idx)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_size)))
        n_val = max(1, int(round(n * val_size)))
        n_train = min(n_train, n - 2) if n >= 3 else max(1, n - 1)
        n_val = min(n_val, max(1, n - n_train - 1)) if n >= 3 else 0
        train_parts.append(shuffled[:n_train])
        val_parts.append(shuffled[n_train : n_train + n_val])
        test_parts.append(shuffled[n_train + n_val :])
    return np.concatenate(train_parts), np.concatenate(val_parts), np.concatenate(test_parts)


def make_split_indices(config: dict, participant_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create either subject-independent or within-subject splits."""
    split_cfg = config["split"]
    if bool(split_cfg.get("subject_independent", True)):
        return split_by_participant(
            participant_ids=participant_ids,
            train_size=float(split_cfg["train_size"]),
            val_size=float(split_cfg["val_size"]),
            seed=int(config["seed"]),
        )
    return split_within_subject(
        participant_ids=participant_ids,
        train_size=float(split_cfg["train_size"]),
        val_size=float(split_cfg["val_size"]),
        seed=int(config["seed"]),
    )


def train_baseline_pipeline(config: dict, allow_synthetic_debug: bool = False) -> dict:
    """Run aggregated-feature training and evaluation."""
    raw = _load_or_synthetic(config, allow_synthetic_debug=allow_synthetic_debug)
    preprocessed = preprocess_eye_tracking(raw, config["preprocessing"])
    features = create_aggregated_features(preprocessed)

    metadata_cols = ["participant_id", "trial_id", "label", "stimulus_id"]
    feature_cols = [column for column in features.columns if column not in metadata_cols]
    X = features[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = features["label"].to_numpy(dtype=int)
    participants = features["participant_id"].to_numpy()

    train_idx, val_idx, test_idx = make_split_indices(config, participants)

    model = build_baseline_model(config["model"]["name"])
    model.fit(X[train_idx], y[train_idx])

    y_pred = model.predict(X[test_idx])
    y_score = model.predict_proba(X[test_idx])[:, 1] if hasattr(model, "predict_proba") else None
    metrics = compute_classification_metrics(y[test_idx], y_pred, y_score)
    metrics["n_train"] = int(len(train_idx))
    metrics["n_val"] = int(len(val_idx))
    metrics["n_test"] = int(len(test_idx))
    metrics["per_participant"] = per_participant_metrics(y[test_idx], y_pred, participants[test_idx]).to_dict("records")

    output_dir = Path(config["training"]["output_dir"])
    save_metrics(metrics, output_dir / "metrics" / f"{config['model']['name']}_metrics.json")
    save_confusion_matrix_figure(
        np.asarray(metrics["confusion_matrix"]),
        output_dir / "figures" / f"{config['model']['name']}_confusion_matrix.png",
    )
    if y_score is not None and len(np.unique(y[test_idx])) == 2:
        fpr, tpr = compute_roc(y[test_idx], y_score)
        save_roc_curve(fpr, tpr, output_dir / "figures" / f"{config['model']['name']}_roc_curve.png")

    checkpoint_path = output_dir / "checkpoints" / f"{config['model']['name']}.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        pickle.dump({"model": model, "feature_columns": feature_cols}, handle)

    return metrics


def _select_window_subset(dataset: WindowedDataset, indices: np.ndarray) -> WindowedDataset:
    return WindowedDataset(
        X=dataset.X[indices],
        y=dataset.y[indices],
        participant_ids=dataset.participant_ids[indices],
        trial_ids=dataset.trial_ids[indices],
    )


def train_sequence_pipeline(config: dict, allow_synthetic_debug: bool = False) -> dict:
    """Run sequence-model training and evaluation on sliding windows."""
    import torch
    from torch.utils.data import DataLoader

    from src.models.sequence import build_sequence_model
    from src.training.datasets import SequenceDataset
    from src.training.trainer import train_sequence_model

    raw = _load_or_synthetic(config, allow_synthetic_debug=allow_synthetic_debug)
    preprocessed = preprocess_eye_tracking(raw, config["preprocessing"])
    features_cfg = config["features"]
    windowed = create_sliding_windows(
        preprocessed,
        sequence_columns=list(features_cfg["sequence_columns"]),
        window_size=int(features_cfg["window_size"]),
        stride=int(features_cfg["stride"]),
    )
    if len(windowed.y) == 0:
        raise ValueError("No sliding windows were produced. Check the window size and raw data.")

    train_idx, val_idx, test_idx = make_split_indices(config, windowed.participant_ids)

    train_subset = _select_window_subset(windowed, train_idx)
    val_subset = _select_window_subset(windowed, val_idx)
    test_subset = _select_window_subset(windowed, test_idx)

    train_set = SequenceDataset(train_subset.X, train_subset.y)
    val_set = SequenceDataset(val_subset.X, val_subset.y)
    test_set = SequenceDataset(test_subset.X, test_subset.y)

    batch_size = int(config["training"].get("batch_size", 32))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = build_sequence_model(config["model"])
    output_dir = Path(config["training"]["output_dir"])
    checkpoint_path = output_dir / "checkpoints" / f"{config['model']['name']}.pt"
    model, history = train_sequence_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(config["training"]["epochs"]),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
        patience=int(config["training"].get("patience", 5)),
        checkpoint_path=checkpoint_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).cpu()
            logits_list.append(logits)
            labels_list.append(batch_y)
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).numpy()
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    y_score = probabilities[:, 1] if probabilities.shape[1] == 2 else None

    metrics = compute_classification_metrics(labels, predictions, y_score)
    metrics["best_epoch"] = history.best_epoch
    metrics["best_val_loss"] = history.best_val_loss
    metrics["n_train"] = int(len(train_idx))
    metrics["n_val"] = int(len(val_idx))
    metrics["n_test"] = int(len(test_idx))
    metrics["per_participant"] = per_participant_metrics(labels, predictions, test_subset.participant_ids).to_dict("records")

    save_metrics(metrics, output_dir / "metrics" / f"{config['model']['name']}_metrics.json")
    save_confusion_matrix_figure(
        np.asarray(metrics["confusion_matrix"]),
        output_dir / "figures" / f"{config['model']['name']}_confusion_matrix.png",
    )
    if y_score is not None and len(np.unique(labels)) == 2:
        fpr, tpr = compute_roc(labels, y_score)
        save_roc_curve(fpr, tpr, output_dir / "figures" / f"{config['model']['name']}_roc_curve.png")

    return metrics
